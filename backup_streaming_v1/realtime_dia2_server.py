"""Dia2 Streaming TTS Server - No-Prefix VoiceSession for Low Latency.

This version creates a VoiceSession at startup WITHOUT voice prefix.
CUDA graphs are captured once during warmup and reused for all requests.
KV cache is reset to empty between requests.
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Parse args early
def _parse_args():
    parser = argparse.ArgumentParser(description="Dia2 Streaming TTS Server")
    parser.add_argument("--port", type=int, default=3030, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--prefix-audio", type=str, default="prefix.wav", help="Path to prefix audio for voice cloning")
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
        gen_state.audio_buf.fill_(token_ids.ungenerated)
        
        # Build plan (runs Whisper)
        prefix_cfg = PrefixConfig(speaker_1=_args.prefix_audio)
        prefix_plan = build_prefix_plan(runtime, prefix_cfg)
        
        # Safety: Truncate prefix if it exceeds context limits
        max_allowed = 1000 # ~20 seconds
        if prefix_plan.aligned_frames > max_allowed:
            print(f"[Dia2] Warning: Prefix too long ({prefix_plan.aligned_frames} frames), truncating to {max_allowed}")
            prefix_plan.aligned_frames = max_allowed
            prefix_plan.aligned_tokens = prefix_plan.aligned_tokens[:, :max_allowed]
            # Also truncate entries/new_word_steps if possible, but for warmup just limiting frames is key
        
        # Create temporary state machine for warmup
        state = runtime.machine.new_state(prefix_plan.entries)
        
        # Populate audio_buf with delayed prefix (CRITICAL: This was missing!)
        delayed = delay_frames(
            prefix_plan.aligned_tokens, 
            runtime.audio_delays, 
            token_ids.audio_pad
        ).to(runtime.device)
        length = min(delayed.shape[1], gen_state.audio_buf.shape[-1])
        gen_state.audio_buf[0, :, :length] = delayed[:, :length]
        if branches > 1:
            gen_state.audio_buf[1:, :, :length] = delayed[:, :length]
        
        # Run warmup (fills KV cache)
        start_step = warmup_with_prefix(runtime, prefix_plan, state, gen_state)
        
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
        
        # Combine with prefix entries if available
        entries = []
        if session.prefix_plan:
            entries.extend(session.prefix_plan.entries)
        entries.extend(new_entries)
        
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
            # Fast-forward state machine through prefix
            plan = session.prefix_plan
            for t in range(plan.aligned_frames):
                forced = token_ids.new_word if t in plan.new_word_steps else token_ids.pad
                runtime.machine.process(t, state, forced, is_forced=True)
        
        positions_view = positions.expand(branches, -1)
        
        # Generation state
        eos_cutoff = None
        frames_generated = 0
        total_samples_output = 0
        chunks_sent = 0
        fade_samples = 4800  # ~200ms at 24kHz
        
        # Reset seed for consistent voice if specified
        if _args.seed is not None:
            torch.manual_seed(_args.seed)
            torch.cuda.manual_seed(_args.seed)

        sample_rate = runtime.mimi.sample_rate
        
        t_setup = time.perf_counter()
        print(f"[Dia2] Setup: {(t_setup - t_start)*1000:.0f}ms")
        
        max_frames = 500  # Safety limit
        
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
            
            # State machine
            main_token, aux_token, _ = runtime.machine.process(t, state, text_token)
            step_tokens[:, 0, 0] = main_token
            step_tokens[:, 1, 0] = aux_token if aux_token != -1 else token_ids.pad
            
            # Audio sampling
            guided_cb0 = apply_classifier_guidance(buffers.cb0, False, 1.0, 50)
            masked_cb0 = mask_audio_logits(guided_cb0[:1], token_ids.audio_pad, token_ids.audio_bos)
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
            # Strategy: Wait for full alignment (max_delay + 1) to ensure clean audio.
            frames_before_decode = max_delay + 1
            should_decode = (frames_generated >= frames_before_decode) or is_final
            
            if should_decode:
                # Undelay and decode
                end_pos = frames_generated + 1
                delayed_tokens = audio_buf[0, :, :end_pos].clone()
                
                aligned = undelay_frames(
                    delayed_tokens,
                    runtime.audio_delays,
                    token_ids.audio_pad
                ).unsqueeze(0)
                
                aligned[aligned >= 2048] = 0
                aligned[aligned < 0] = 0
                
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


# =============================================================================
# WebSocket Handler
# =============================================================================

@app.websocket("/ws/stream_tts")
async def stream_tts(ws: WebSocket):
    await ws.accept()
    print("[Dia2] WebSocket connected")
    
    try:
        await ws.send_text(json.dumps({
            "event": "ready",
            "sample_rate": dia.sample_rate,
        }))
        
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=120)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"event": "ping"}))
                continue
            except Exception:
                break
            
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"error": "Invalid JSON"}))
                continue
            
            msg_type = payload.get("type", "tts")
            
            if msg_type == "pong":
                continue
            
            if msg_type == "close":
                break
            
            if msg_type == "tts" or "text" in payload:
                text = payload.get("text")
                if not text:
                    await ws.send_text(json.dumps({"error": "Missing 'text'"}))
                    continue
                
                chunk_queue: Queue = Queue()
                
                loop = asyncio.get_running_loop()
                gen_future = loop.run_in_executor(
                    _cuda_executor,
                    lambda: _run_tts(
                        text=text,
                        output_queue=chunk_queue,
                        session=_session,
                        temperature=float(payload.get("temperature", 0.8)),
                        top_k=int(payload.get("top_k", 50)),
                    )
                )
                
                chunks_sent = 0
                while True:
                    try:
                        chunk = await loop.run_in_executor(
                            None,
                            lambda: chunk_queue.get(timeout=0.05)
                        )
                    except Empty:
                        if gen_future.done():
                            while True:
                                try:
                                    chunk = chunk_queue.get_nowait()
                                    if chunk is None:
                                        break
                                    header = struct.pack("!?", chunk.is_last)
                                    await ws.send_bytes(header + chunk.pcm16)
                                    chunks_sent += 1
                                except Empty:
                                    break
                            break
                        continue
                    
                    if chunk is None:
                        break
                    
                    header = struct.pack("!?", chunk.is_last)
                    await ws.send_bytes(header + chunk.pcm16)
                    chunks_sent += 1
                    
                    if chunk.is_last:
                        break
                
                await ws.send_text(json.dumps({"event": "done", "chunks": chunks_sent}))
                print(f"[Dia2] Sent {chunks_sent} chunks")
                continue
            
            await ws.send_text(json.dumps({"error": f"Unknown type: {msg_type}"}))
    
    except Exception as e:
        print(f"[Dia2] WebSocket error: {e}")
    finally:
        print("[Dia2] Connection closed")


if __name__ == "__main__":
    import uvicorn
    print(f"[Dia2] Starting server on {_args.host}:{_args.port}")
    uvicorn.run(app, host=_args.host, port=_args.port)
