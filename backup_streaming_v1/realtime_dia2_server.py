"""Dia2 Streaming TTS Server - Stable Low Latency.

Architecture:
1. Cached Prefix: Pre-calculates state for 'prefix.wav' at startup.
2. Soundfile Loading: Forces robust audio loading to prevent truncation.
3. Full Buffer Decoding: Stable decoding strategy.
4. Prefix Skipping: Correctly initializes output pointer to skip prefix audio.
"""
import argparse
import asyncio
import base64
import json
import struct
import tempfile
import os
import time
import copy
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

# Force local import of dia2 to ensure patches are applied
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dia2
print(f"[Dia2] Loaded dia2 package from: {dia2.__file__}")

import torch
import torch.nn.functional as F
import soundfile as sf  # Explicit import for robustness
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Parse args early
def _parse_args():
    parser = argparse.ArgumentParser(description="Dia2 Streaming TTS Server")
    parser.add_argument("--port", type=int, default=3030, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--prefix-audio", type=str, default=os.path.abspath("prefix.wav"), help="Path to prefix audio for voice cloning")
    parser.add_argument("--prefix-speaker-2", type=str, default=None, help="Path to prefix audio for Speaker 2")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation")
    args, _ = parser.parse_known_args()
    return args

_args = _parse_args()

if _args.seed is not None:
    print(f"[Dia2] Setting random seed to {_args.seed}")
    torch.manual_seed(_args.seed)
    torch.cuda.manual_seed(_args.seed)
    np.random.seed(_args.seed)

from dia2 import Dia2, GenerationConfig, SamplingConfig
from dia2.runtime.generator import (
    build_initial_state, 
    GenerationState,
    _allocate_network_buffers,
    warmup_with_prefix,
    _ensure_graph_cublas_ready,
    _execute_transformer_graph,
    _execute_depformer_graph,
    _fill_audio_channels,
    NetworkBuffers,
)
from dia2.runtime.streaming_generator import CachedGraphs
from dia2.runtime.script_parser import parse_script
from dia2.runtime.voice_clone import build_prefix_plan, PrefixPlan
from dia2.generation import PrefixConfig
from dia2.generation import normalize_script
from dia2.runtime.guidance import apply_classifier_guidance, sample_audio_logits
from dia2.runtime.sampler import sample_token
from dia2.audio.grid import mask_audio_logits, undelay_frames, delay_frames

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_REPO = "nari-labs/Dia2-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"

print(f"[Dia2] Initializing Dia2 from {MODEL_REPO} on {DEVICE} ({DTYPE})...")
dia = Dia2.from_repo(MODEL_REPO, device=DEVICE, dtype=DTYPE)

_cuda_executor = ThreadPoolExecutor(max_workers=1)


# =============================================================================
# No-Prefix VoiceSession - CUDA graphs captured once, reused for all requests
# =============================================================================

@dataclass
class StateSnapshot:
    """Holds a copy of the generation state (KV cache, buffers) after prefix warmup."""
    transformer_kv: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]  # (k, v, len)
    audio_buf: torch.Tensor
    step_tokens: torch.Tensor
    start_step: int

@dataclass
class VoiceSession:
    """Holds pre-warmed state for fast TTS - no voice prefix required."""
    gen_state: GenerationState
    buffers: NetworkBuffers
    positions: torch.Tensor
    transformer_capture: Any
    dep_captures: List[Dict]
    prefix_plan: Optional[PrefixPlan] = None
    snapshot: Optional[StateSnapshot] = None
    
_session: Optional[VoiceSession] = None


def _create_session() -> VoiceSession:
    """Create session and capture CUDA graphs during warmup."""
    print("[Dia2] Creating no-prefix VoiceSession...")
    runtime = dia._ensure_runtime()
    
    # Debug vocab sizes
    vocab_size = runtime.config.data.audio_vocab_size
    pad_id = runtime.constants.audio_pad
    print(f"[Dia2] Vocab: audio={vocab_size}, pad={pad_id}")
    
    # CRITICAL FIX: Ensure audio_pad is within vocab bounds
    if pad_id >= vocab_size:
        print(f"[Dia2] WARNING: audio_pad {pad_id} >= vocab {vocab_size}. Clamping to {vocab_size - 1}.")
        runtime.constants.audio_pad = vocab_size - 1
    
    # Create generation state (no prefix)
    gen_state = build_initial_state(runtime, prefix=None)
    
    # Allocate buffers
    branches = gen_state.step_tokens.shape[0]
    buffers = _allocate_network_buffers(runtime, branches)
    
    # Position tensor
    positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
    
    # Ensure CUDA ready
    _ensure_graph_cublas_ready(runtime.device)
    
    # Capture CUDA graphs by running warmup
    print("[Dia2] Capturing CUDA graphs...")
    t_start = time.perf_counter()
    
    step_tokens = gen_state.step_tokens
    audio_buf = gen_state.audio_buf
    token_ids = runtime.constants
    delay_tensor = runtime.audio_delay_tensor
    positions_view = positions.expand(branches, -1)
    
    transformer_capture = None
    dep_captures = None
    
    # --- 1. Warmup for Graph Capture (Dummy Data) ---
    
    # Run 30 frames to capture all graphs
    for t in range(30):
        gen_state.reset_dep_cache()
        positions.fill_(t)
        _fill_audio_channels(step_tokens, audio_buf, delay_tensor, t, token_ids.audio_bos)
        
        if branches > 1:
            step_tokens[1:, 0, 0] = token_ids.zero
            step_tokens[1:, 1, 0] = token_ids.pad
        
        # Transformer with graph capture
        transformer_capture, dep_captures = _execute_transformer_graph(
            runtime=runtime,
            step_tokens=step_tokens,
            positions_view=positions_view,
            branches=branches,
            generation=gen_state,
            transformer_step=runtime.transformer_step,
            buffers=buffers,
            transformer_capture=transformer_capture,
            dep_captures=dep_captures,
        )
        hidden_t = transformer_capture[1]
        
        # Dummy sampling
        guided_text = apply_classifier_guidance(buffers.text, False, 1.0, 50)
        text_token = sample_token(guided_text[:1], temp=0.8, top_k=50).item()
        
        step_tokens[:, 0, 0] = token_ids.pad
        step_tokens[:, 1, 0] = token_ids.pad
        
        guided_cb0 = apply_classifier_guidance(buffers.cb0, False, 1.0, 50)
        masked_cb0 = mask_audio_logits(guided_cb0[:1], token_ids.audio_pad, token_ids.audio_bos)
        codebook_token = sample_audio_logits(masked_cb0, 0.8, 50)
        audio_buf[:, 0, t + 1] = codebook_token
        
        prev_audio = codebook_token.expand(branches)
        main_tokens = torch.full((branches,), token_ids.pad, dtype=torch.long, device=runtime.device)
        aux_tokens = torch.full((branches,), token_ids.pad, dtype=torch.long, device=runtime.device)
        
        # Depformer with graph capture
        for stage in range(runtime.model.depformer.num_depth):
            dep_captures[stage] = _execute_depformer_graph(
                stage=stage,
                prev_audio=prev_audio,
                hidden_t=hidden_t,
                generation=gen_state,
                depformer_step=runtime.depformer_step,
                main_tokens=main_tokens,
                aux_tokens=aux_tokens,
                buffers=buffers,
                capture=dep_captures[stage],
            )
            
            dep_logits = apply_classifier_guidance(buffers.dep[stage], False, 1.0, 50)
            stage_token = sample_audio_logits(dep_logits[:1], 0.8, 50)
            audio_buf[:, stage + 1, t + 1] = stage_token
            prev_audio = stage_token.expand(branches)
    
    t_end = time.perf_counter()
    print(f"[Dia2] Graph capture complete: {(t_end - t_start)*1000:.0f}ms")
    
    # --- 2. Warmup for Voice Prefix (Real Data) ---
    prefix_plan = None
    snapshot = None
    
    if _args.prefix_audio and os.path.exists(_args.prefix_audio):
        print(f"[Dia2] Processing voice prefix: {_args.prefix_audio}...")
        t_prefix = time.perf_counter()
        
        # Reset state before prefix warmup
        for slot in gen_state.decode.transformer.slots:
            slot.length.fill_(0)
        # Initialize with audio_bos instead of ungenerated (-2) to prevent crashes
        gen_state.audio_buf.fill_(token_ids.audio_bos)
        
        # Build plan (runs Whisper)
        # We inject a custom load_audio_fn to force soundfile usage
        def safe_load_audio(path, sr):
            print(f"[Dia2] Loading audio with soundfile: {path}")
            audio, _ = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            print(f"[Dia2] Audio loaded: shape={audio.shape}, max={audio.max()}, min={audio.min()}")
            return audio

        s2_path = _args.prefix_speaker_2
        if s2_path and not os.path.exists(s2_path):
            print(f"[Dia2] Warning: Speaker 2 prefix not found at {s2_path}, ignoring.")
            s2_path = None

        prefix_cfg = PrefixConfig(speaker_1=_args.prefix_audio, speaker_2=s2_path)
        prefix_plan = build_prefix_plan(runtime, prefix_cfg, load_audio_fn=safe_load_audio)
        
        # Safety: Truncate prefix if it exceeds context limits
        max_allowed = 1000 # ~20 seconds
        if prefix_plan.aligned_frames > max_allowed:
            print(f"[Dia2] Warning: Prefix too long ({prefix_plan.aligned_frames} frames), truncating to {max_allowed}")
            prefix_plan.aligned_frames = max_allowed
            prefix_plan.aligned_tokens = prefix_plan.aligned_tokens[:, :max_allowed]
        
        # Create temporary state machine for warmup
        state = runtime.machine.new_state(prefix_plan.entries)
        
        # Populate audio_buf with delayed prefix (CRITICAL for stability)
        delayed = delay_frames(
            prefix_plan.aligned_tokens, 
            runtime.audio_delays, 
            token_ids.audio_pad
        ).to(runtime.device)
        
        # Safety clamp: Ensure no tokens exceed vocab size (prevent CUDA assert)
        if token_ids.audio_pad >= vocab_size:
             delayed = torch.clamp(delayed, 0, vocab_size - 1)
        
        length = min(delayed.shape[1], gen_state.audio_buf.shape[-1])
        gen_state.audio_buf[0, :, :length] = delayed[:, :length]
        if branches > 1:
            gen_state.audio_buf[1:, :, :length] = delayed[:, :length]
        
        # Run warmup (fills KV cache)
        start_step = warmup_with_prefix(runtime, prefix_plan, state, gen_state)
        print(f"[Dia2] Warmup complete. Aligned frames: {prefix_plan.aligned_frames}, Start step: {start_step}")
        
        # Save snapshot
        kv_snapshot = []
        for slot in gen_state.decode.transformer.slots:
            # Clone keys, values, and length
            kv_snapshot.append((
                slot.keys.clone(),
                slot.values.clone(),
                slot.length.clone()
            ))
            
        snapshot = StateSnapshot(
            transformer_kv=kv_snapshot,
            audio_buf=gen_state.audio_buf.clone(),
            step_tokens=gen_state.step_tokens.clone(),
            start_step=start_step + 1
        )
        
        print(f"[Dia2] Prefix processed: {(time.perf_counter() - t_prefix)*1000:.0f}ms, {start_step} frames")
    else:
        print(f"[Dia2] WARNING: Prefix audio not found at {_args.prefix_audio}. Starting without voice clone.")
    
    return VoiceSession(
        gen_state=gen_state,
        buffers=buffers,
        positions=positions,
        transformer_capture=transformer_capture,
        dep_captures=dep_captures,
        prefix_plan=prefix_plan,
        snapshot=snapshot,
    )


def _reset_session(session: VoiceSession) -> None:
    """Reset session for new TTS request - restores snapshot if available."""
    runtime = dia._ensure_runtime()
    
    if session.snapshot:
        # Restore from snapshot
        snap = session.snapshot
        
        # Restore KV cache
        for i, slot in enumerate(session.gen_state.decode.transformer.slots):
            k, v, l = snap.transformer_kv[i]
            slot.keys.copy_(k)
            slot.values.copy_(v)
            slot.length.copy_(l)
            
        # Restore buffers
        session.gen_state.audio_buf.copy_(snap.audio_buf)
        session.gen_state.step_tokens.copy_(snap.step_tokens)
        
    else:
        # Reset to empty
        for slot in session.gen_state.decode.transformer.slots:
            slot.length.fill_(0)
        session.gen_state.audio_buf.fill_(runtime.constants.ungenerated)

    session.gen_state.decode.depformer.reset()


# Initialize session at startup
print("[Dia2] Initializing session...")
_session = _cuda_executor.submit(_create_session).result()
print("[Dia2] Session ready!")


# =============================================================================
# Streaming Generation
# =============================================================================

@dataclass
class AudioChunk:
    pcm16: bytes
    sample_rate: int
    is_last: bool


def waveform_to_pcm16(waveform: torch.Tensor) -> bytes:
    wav_np = waveform.detach().cpu().numpy().astype(np.float32)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    return (wav_np * 32767.0).astype(np.int16).tobytes()


def _run_tts(
    text: str,
    output_queue: Queue,
    session: VoiceSession,
    temperature: float = 0.8,
    top_k: int = 50,
) -> None:
    """Run TTS using pre-warmed session with cached CUDA graphs."""
    try:
        t_start = time.perf_counter()
        print(f"[Dia2] TTS: {text[:60]}...")
        
        runtime = dia._ensure_runtime()
        
        # Reset session for new request
        _reset_session(session)
        
        # Parse text
        normalized = normalize_script(text)
        new_entries = list(parse_script(
            [normalized],
            runtime.tokenizer,
            runtime.constants,
            runtime.frame_rate
        ))
        
        # ONLY use new entries. The prefix is already in the model's KV cache.
        entries = new_entries
        print(f"Entries:", entries) 
        # Create state machine
        runtime.machine.initial_padding = 0
        state = runtime.machine.new_state(entries)
        
        # Get session components
        gen = session.gen_state
        buffers = session.buffers
        positions = session.positions
        transformer_capture = session.transformer_capture
        dep_captures = session.dep_captures
        
        step_tokens = gen.step_tokens
        audio_buf = gen.audio_buf
        branches = step_tokens.shape[0]
        token_ids = runtime.constants
        delay_tensor = runtime.audio_delay_tensor
        max_delay = int(delay_tensor.max().item()) if delay_tensor.numel() else 0
        flush_tail = max_delay + getattr(runtime.machine, "max_padding", 0)
        print(f"[Dia2] max_delay: {max_delay} frames")
        
        # Determine start step
        start_step = 0
        if session.snapshot:
            start_step = session.snapshot.start_step
            print(f"[Dia2] Resuming from cached state at step {start_step}")
        
        positions_view = positions.expand(branches, -1)
        
        # Generation state
        eos_cutoff = None
        frames_generated = 0
        total_samples_output = 0
        chunks_sent = 0
        
        # If we have a prefix, calculate its length in samples so we can skip it in output
        if session.prefix_plan:
            # Initialize total_samples_output based on start_step
            # This ensures we skip exactly what was pre-calculated in the state
            # We assume 320 samples per frame (24000 / 75)
            samples_per_frame = 320
            total_samples_output = start_step * samples_per_frame
            print(f"[Dia2] Prefix length: {total_samples_output} samples. Skipping these in output.")
        
        # Reset seed for consistent voice if specified
        if _args.seed is not None:
            torch.manual_seed(_args.seed)
            torch.cuda.manual_seed(_args.seed)

        sample_rate = runtime.mimi.sample_rate
        
        t_setup = time.perf_counter()
        print(f"[Dia2] Setup: {(t_setup - t_start)*1000:.0f}ms")
        
        # Allow generation up to 1500 frames beyond start
        max_frames = start_step + 1500
        print(f"[Dia2] Starting at step {start_step}, max_frames {max_frames}")
        
        prefix_frames = session.prefix_plan.aligned_frames if session.prefix_plan else 0

        for t in range(start_step, max_frames):
            if eos_cutoff is not None and t >= eos_cutoff:
                break
            if t + 1 >= audio_buf.shape[-1]:
                break
            
            gen.reset_dep_cache()
            positions.fill_(t)
            _fill_audio_channels(step_tokens, audio_buf, delay_tensor, t, token_ids.audio_bos)
            
            if branches > 1:
                step_tokens[1:, 0, 0] = token_ids.zero
                step_tokens[1:, 1, 0] = token_ids.pad
            
            # Replay transformer graph
            transformer_capture[0].replay()
            hidden_t = transformer_capture[1]
            
            # Text sampling
            guided_text = apply_classifier_guidance(buffers.text, False, 1.0, 50)
            text_token = sample_token(guided_text[:1], temp=temperature, top_k=top_k).item()
            
            # Force new_word at the start to ensure we switch to the new text immediately
            is_forced = False
            if t == start_step:
                text_token = token_ids.new_word
                is_forced = True

            # State machine
            main_token, aux_token, _ = runtime.machine.process(t, state, text_token, is_forced=is_forced)
            step_tokens[:, 0, 0] = main_token
            step_tokens[:, 1, 0] = aux_token if aux_token != -1 else token_ids.pad
            
            # Audio sampling
            guided_cb0 = apply_classifier_guidance(buffers.cb0, False, 1.0, 50)
            masked_cb0 = mask_audio_logits(guided_cb0[:1], token_ids.audio_pad, token_ids.audio_bos)
            
            # Force ground truth if within prefix
            if t < prefix_frames:
                codebook_token = audio_buf[:, 0, t + 1]
            else:
                codebook_token = sample_audio_logits(masked_cb0, temperature, top_k)
            
            audio_buf[:, 0, t + 1] = codebook_token
            
            prev_audio = codebook_token.expand(branches)
            main_tokens = torch.full((branches,), main_token, dtype=torch.long, device=runtime.device)
            aux_tokens_t = torch.full((branches,), step_tokens[0, 1, 0].item(), dtype=torch.long, device=runtime.device)
            
            # Replay depformer graphs
            for stage in range(runtime.model.depformer.num_depth):
                dep_captures[stage]["prev_audio"].copy_(prev_audio)
                if stage == 0:
                    dep_captures[stage]["main_tokens"].copy_(main_tokens)
                    dep_captures[stage]["second_tokens"].copy_(aux_tokens_t)
                dep_captures[stage]["graph"].replay()
                
                dep_logits = apply_classifier_guidance(buffers.dep[stage], False, 1.0, 50)
                
                # Force ground truth if within prefix (accounting for delay)
                # Channel k (stage+1) is at audio frame t - delay
                delay = runtime.audio_delays[stage + 1]
                if (t - delay) < prefix_frames:
                    stage_token = audio_buf[:, stage + 1, t + 1]
                else:
                    stage_token = sample_audio_logits(dep_logits[:1], temperature, top_k)
                
                audio_buf[:, stage + 1, t + 1] = stage_token
                prev_audio = stage_token.expand(branches)
            
            frames_generated += 1
            
            if frames_generated == 1:
                print(f"[Dia2] First frame: {(time.perf_counter() - t_start)*1000:.0f}ms")
            
            if eos_cutoff is None and state.end_step is not None:
                eos_cutoff = state.end_step + flush_tail
            
            is_final = (eos_cutoff is not None and t + 1 >= eos_cutoff)
            
            # Decode when we have enough frames
            # Strategy: If we have prefix, we can decode immediately. Otherwise wait for buffer.
            frames_before_decode = 1 if start_step > 0 else (max_delay + 1)
            should_decode = (frames_generated >= frames_before_decode) or is_final
            
            if should_decode:
                # Undelay and decode
                # Use full buffer decoding (stable) instead of sliding window
                end_pos = t + 2
                delayed_tokens = audio_buf[0, :, :end_pos].clone()
                
                aligned = undelay_frames(
                    delayed_tokens,
                    runtime.audio_delays,
                    token_ids.audio_pad
                ).unsqueeze(0)
                
                # Safety clamp to valid codebook range (0-2047)
                # This prevents CUDA errors if audio_pad (2048) leaks into Mimi
                aligned = torch.clamp(aligned, 0, 2047)
                
                torch.cuda.synchronize()
                pcm = runtime.mimi.decode(aligned)
                full_waveform = torch.clamp(pcm[0, 0], -1.0, 1.0)
                
                if full_waveform.shape[0] > total_samples_output:
                    waveform = full_waveform[total_samples_output:]
                    
                    if waveform.shape[0] > 0:
                        if chunks_sent == 0:
                            print(f"[Dia2] First audio: {(time.perf_counter() - t_start)*1000:.0f}ms")
                        
                        chunks_sent += 1
                        total_samples_output += waveform.shape[0]
                       
                        if chunks_sent>150: 
                            output_queue.put(AudioChunk(
                                pcm16=waveform_to_pcm16(waveform),
                                sample_rate=sample_rate,
                                is_last=is_final,
                            ))
                    
                if is_final:
                    break
        
        duration = total_samples_output / sample_rate if total_samples_output > 0 else 0
        print(f"[Dia2] Done: {chunks_sent} chunks, {duration:.2f}s audio")
        
    except Exception as e:
        print(f"[Dia2] TTS error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        output_queue.put(None)


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Send ready message
    runtime = dia._ensure_runtime()
    await websocket.send_json({
        "event": "ready",
        "sample_rate": runtime.mimi.sample_rate,
        "version": "1.0.0"
    })
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"event": "pong"})
                continue
                
            if data.get("type") == "tts":
                text = data.get("text")
                if not text:
                    continue
                    
                # Run TTS in background thread
                output_queue = Queue()
                
                if _session is None:
                    await websocket.send_json({"error": "Server not ready (no session)"})
                    continue
                
                loop = asyncio.get_event_loop()
                # Use the single-threaded CUDA executor to ensure serial access to GPU
                future = loop.run_in_executor(
                    _cuda_executor, 
                    _run_tts, 
                    text, 
                    output_queue, 
                    _session,
                    data.get("temperature", 0.8),
                    data.get("top_k", 50)
                )
                
                chunks_sent = 0
                while True:
                    # Wait for queue item in default executor to avoid blocking event loop
                    chunk = await loop.run_in_executor(None, output_queue.get)
                    
                    if chunk is None:
                        break
                        
                    header = struct.pack("!?", chunk.is_last)
                    await websocket.send_bytes(header + chunk.pcm16)
                    chunks_sent += 1
                
                await websocket.send_json({"event": "done", "chunks": chunks_sent})
                
    except WebSocketDisconnect:
        print("[Dia2] Client disconnected")
    except Exception as e:
        print(f"[Dia2] WebSocket error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    print(f"[Dia2] Starting server on {_args.host}:{_args.port}")
    uvicorn.run(app, host=_args.host, port=_args.port)
