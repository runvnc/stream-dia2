"""Dia2 TTS Server with pre-warmed voice state for minimal latency.

This version pre-warms the transformer cache when voice is set, then reuses
that state for each TTS request. Single conversation at a time.

The warmup happens ONCE when set_voice is called, not per TTS request.
"""
import asyncio
import base64
import json
import struct
import tempfile
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue, Empty
import time
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Import dia2 components
from dia2 import Dia2, GenerationConfig, SamplingConfig
from dia2.runtime import voice_clone
from dia2 import engine as dia2_engine
from dia2.runtime.voice_clone import WhisperWord, build_prefix_plan, PrefixPlan
from dia2.runtime.audio_io import load_mono_audio, encode_audio_tokens
from dia2.generation import PrefixConfig, normalize_script
from dia2.runtime.generator import build_initial_state, warmup_with_prefix, GenerationState
from dia2.runtime.streaming_generator import run_streaming_generation, StreamingChunk
from dia2.runtime.script_parser import parse_script
from dia2.audio.grid import delay_frames
from dia2.core.cache import KVCache, CacheSlot
from dia2.core.model import DecodeState


app = FastAPI()

MODEL_REPO = "nari-labs/Dia2-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"

# ============================================================================
# Pre-warmed Voice State - Single conversation mode
# ============================================================================

@dataclass
class PrewarmedVoiceState:
    """Cached state after voice warmup - ready for instant TTS."""
    prefix_plan: PrefixPlan
    start_step: int
    # Snapshot of the warmed KV cache (keys/values tensors)
    transformer_cache_snapshot: List[Tuple[torch.Tensor, torch.Tensor]]
    # Original generation state for reference
    audio_buf_template: torch.Tensor
    step_tokens_template: torch.Tensor


# Global pre-warmed state (single conversation mode)
_prewarmed_state: Optional[PrewarmedVoiceState] = None
_voice_speaker_1_path: Optional[str] = None


# ============================================================================
# Whisper Model Caching
# ============================================================================

_whisper_model = None
_audio_token_cache: Dict[str, torch.Tensor] = {}
_audio_data_cache: Dict[str, np.ndarray] = {}
_transcription_cache: Dict[str, List[WhisperWord]] = {}
_current_encoding_path: Optional[str] = None


def _get_whisper_model():
    """Get or load the cached Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        import whisper_timestamped as wts
        print("[Dia2] Loading Whisper model (one-time)...")
        _whisper_model = wts.load_model("openai/whisper-large-v3", device=DEVICE)
        print("[Dia2] Whisper model loaded.")
    return _whisper_model


def _hash_file(path: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_transcribe_words(
    audio_path: str,
    device: torch.device,
    language: Optional[str] = None,
) -> List[WhisperWord]:
    """Cached version of transcribe_words."""
    file_hash = _hash_file(audio_path)
    
    if file_hash in _transcription_cache:
        print(f"[Dia2] Using cached transcription")
        return _transcription_cache[file_hash]
    
    print(f"[Dia2] Transcribing {os.path.basename(audio_path)}...")
    import whisper_timestamped as wts
    model = _get_whisper_model()
    result = wts.transcribe(model, audio_path, language=language)
    
    words: List[WhisperWord] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            text = (word.get("text") or word.get("word") or "").strip()
            if not text:
                continue
            words.append(WhisperWord(
                text=text,
                start=float(word.get("start", 0.0)),
                end=float(word.get("end", 0.0)),
            ))
    
    _transcription_cache[file_hash] = words
    print(f"[Dia2] Transcription cached ({len(words)} words)")
    return words


voice_clone.transcribe_words = _cached_transcribe_words


def _cached_load_audio(audio_path: str, sample_rate: int) -> np.ndarray:
    """Load audio with caching."""
    global _current_encoding_path
    file_hash = _hash_file(audio_path)
    _current_encoding_path = audio_path
    
    if file_hash in _audio_data_cache:
        return _audio_data_cache[file_hash]
    
    print(f"[Dia2] Loading audio...")
    audio = load_mono_audio(audio_path, sample_rate)
    _audio_data_cache[file_hash] = audio
    return audio


def _cached_encode_audio(mimi, audio: np.ndarray) -> torch.Tensor:
    """Encode audio with caching."""
    global _current_encoding_path
    
    if _current_encoding_path:
        file_hash = _hash_file(_current_encoding_path)
        if file_hash in _audio_token_cache:
            return _audio_token_cache[file_hash]
    
    print(f"[Dia2] Encoding audio...")
    tokens = encode_audio_tokens(mimi, audio)
    
    if _current_encoding_path:
        file_hash = _hash_file(_current_encoding_path)
        _audio_token_cache[file_hash] = tokens
    
    return tokens


_original_build_prefix_plan = build_prefix_plan

def _cached_build_prefix_plan(runtime, prefix: Optional[PrefixConfig], **kwargs):
    """Wrapper that injects cached functions."""
    if prefix is None:
        return None
    
    def cached_encode_fn(audio: np.ndarray) -> torch.Tensor:
        return _cached_encode_audio(runtime.mimi, audio)
    
    return _original_build_prefix_plan(
        runtime,
        prefix,
        transcribe_fn=_cached_transcribe_words,
        load_audio_fn=_cached_load_audio,
        encode_fn=cached_encode_fn,
        **kwargs
    )

voice_clone.build_prefix_plan = _cached_build_prefix_plan
dia2_engine.build_prefix_plan = _cached_build_prefix_plan


# ============================================================================
# Dia2 Model Initialization
# ============================================================================

print(f"[Dia2] Initializing Dia2 from {MODEL_REPO} on {DEVICE} ({DTYPE})...")
dia = Dia2.from_repo(MODEL_REPO, device=DEVICE, dtype=DTYPE)

print("[Dia2] Pre-loading Whisper model...")
_get_whisper_model()

_cuda_executor = ThreadPoolExecutor(max_workers=1)


def _warmup_model() -> None:
    """Warm up the model."""
    try:
        cfg = GenerationConfig(
            cfg_scale=1.0,
            text=SamplingConfig(temperature=0.6, top_k=50),
            audio=SamplingConfig(temperature=0.8, top_k=50),
            use_cuda_graph=True,
        )
        print("[Dia2] Running warm-up generation...")
        _ = dia.generate("[S1] Warm up.", config=cfg, output_wav=None, verbose=False)
        print("[Dia2] Warm-up complete.")
    except Exception as e:
        print(f"[Dia2] Warm-up failed: {e}")


_cuda_executor.submit(_warmup_model).result()


# ============================================================================
# Voice Pre-warming
# ============================================================================

def _snapshot_kv_cache(cache: KVCache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create a snapshot of KV cache tensors."""
    snapshots = []
    for slot in cache.slots:
        # Clone the actual data up to current length
        length = slot.length.item()
        keys_snapshot = slot.keys[:, :, :length, :].clone()
        values_snapshot = slot.values[:, :, :length, :].clone()
        snapshots.append((keys_snapshot, values_snapshot))
    return snapshots


def _restore_kv_cache(cache: KVCache, snapshots: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Restore KV cache from snapshot."""
    for slot, (keys_snap, values_snap) in zip(cache.slots, snapshots):
        length = keys_snap.shape[2]
        slot.keys[:, :, :length, :].copy_(keys_snap)
        slot.values[:, :, :length, :].copy_(values_snap)
        slot.length.fill_(length)


def _prewarm_voice(speaker_1_path: str) -> PrewarmedVoiceState:
    """Pre-warm the transformer with voice conditioning.
    
    This runs the full warmup once and saves the state for reuse.
    """
    global _prewarmed_state, _voice_speaker_1_path
    
    t0 = time.perf_counter()
    print(f"[Dia2] Pre-warming voice from {speaker_1_path}...")
    
    runtime = dia._ensure_runtime()
    
    # Build prefix plan
    prefix_config = PrefixConfig(speaker_1=speaker_1_path)
    prefix_plan = _cached_build_prefix_plan(runtime, prefix_config)
    
    if prefix_plan is None:
        raise ValueError("Failed to build prefix plan")
    
    # Build initial generation state
    gen_state = build_initial_state(runtime, prefix=prefix_plan)
    
    # Create a dummy state machine state for warmup
    # We just need the prefix entries for warmup
    runtime.machine.initial_padding = 0
    state = runtime.machine.new_state(prefix_plan.entries)
    
    # Run warmup
    start_step = warmup_with_prefix(runtime, prefix_plan, state, gen_state)
    
    # Snapshot the warmed KV cache
    cache_snapshot = _snapshot_kv_cache(gen_state.decode.transformer)
    
    # Save templates
    audio_buf_template = gen_state.audio_buf.clone()
    step_tokens_template = gen_state.step_tokens.clone()
    
    elapsed = time.perf_counter() - t0
    print(f"[Dia2] Voice pre-warmed in {elapsed:.2f}s (start_step={start_step})")
    
    _prewarmed_state = PrewarmedVoiceState(
        prefix_plan=prefix_plan,
        start_step=start_step,
        transformer_cache_snapshot=cache_snapshot,
        audio_buf_template=audio_buf_template,
        step_tokens_template=step_tokens_template,
    )
    _voice_speaker_1_path = speaker_1_path
    
    return _prewarmed_state


# ============================================================================
# Audio Processing
# ============================================================================

@dataclass
class AudioChunk:
    pcm16: bytes
    sample_rate: int
    is_last: bool


def waveform_to_pcm16(waveform: torch.Tensor) -> bytes:
    """Convert waveform tensor to PCM16 bytes."""
    if waveform.ndim != 1:
        waveform = waveform.view(-1)
    wav_np = waveform.detach().cpu().numpy().astype(np.float32)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    pcm16 = (wav_np * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def save_audio_from_base64(audio_b64: str, suffix: str = ".wav") -> str:
    """Decode base64 audio and save to temp file."""
    audio_bytes = base64.b64decode(audio_b64)
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, audio_bytes)
    os.close(fd)
    return path


# ============================================================================
# Streaming TTS with Pre-warmed State
# ============================================================================

def _run_streaming_tts_prewarmed(
    text: str,
    output_queue: Queue,
    prewarmed: PrewarmedVoiceState,
    cfg_scale: float = 1.0,
    temperature: float = 0.8,
    top_k: int = 50,
    chunk_frames: int = 12,
) -> None:
    """Run streaming TTS using pre-warmed voice state."""
    try:
        t_start = time.perf_counter()
        print(f"[Dia2] TTS: {text[:60]}...")
        
        runtime = dia._ensure_runtime()
        
        # Config
        sampling = SamplingConfig(temperature=temperature, top_k=top_k)
        config = GenerationConfig(
            cfg_scale=cfg_scale,
            text=sampling,
            audio=sampling,
            use_cuda_graph=False,
        )
        
        # Build fresh generation state
        gen_state = build_initial_state(runtime, prefix=prewarmed.prefix_plan)
        
        # Restore the pre-warmed transformer cache
        _restore_kv_cache(gen_state.decode.transformer, prewarmed.transformer_cache_snapshot)
        
        # Copy the audio buffer template (has prefix audio tokens)
        gen_state.audio_buf.copy_(prewarmed.audio_buf_template)
        gen_state.step_tokens.copy_(prewarmed.step_tokens_template)
        
        # Parse new text and create state machine
        normalized_text = normalize_script(text)
        entries = list(prewarmed.prefix_plan.entries)  # Start with prefix entries
        entries.extend(parse_script(
            [normalized_text],
            runtime.tokenizer,
            runtime.constants,
            runtime.frame_rate
        ))
        
        runtime.machine.initial_padding = config.initial_padding
        state = runtime.machine.new_state(entries)
        
        # Fast-forward state machine to match warmed position
        # The prefix entries are already "processed" by warmup
        for t in range(prewarmed.start_step):
            # Mark prefix frames as processed in state machine
            if t in prewarmed.prefix_plan.new_word_steps:
                runtime.machine.process(t, state, runtime.constants.new_word, is_forced=True)
            else:
                runtime.machine.process(t, state, runtime.constants.pad, is_forced=True)
        
        t_setup = time.perf_counter()
        print(f"[Dia2] Setup: {(t_setup - t_start)*1000:.0f}ms")
        
        # Run streaming generation from pre-warmed position
        chunk_count = 0
        total_samples = 0
        
        for chunk in run_streaming_generation(
            runtime,
            state=state,
            generation=gen_state,
            config=config,
            start_step=prewarmed.start_step,
            chunk_frames=chunk_frames,
            include_prefix_audio=False,
        ):
            chunk_count += 1
            pcm16_bytes = waveform_to_pcm16(chunk.waveform)
            total_samples += chunk.waveform.shape[0]
            
            output_queue.put(AudioChunk(
                pcm16=pcm16_bytes,
                sample_rate=chunk.sample_rate,
                is_last=chunk.is_final,
            ))
            
            if chunk.is_final:
                break
        
        t_end = time.perf_counter()
        duration = total_samples / runtime.mimi.sample_rate if total_samples > 0 else 0
        print(f"[Dia2] Done: {chunk_count} chunks, {duration:.2f}s audio in {t_end - t_start:.2f}s")
                
    except Exception as e:
        print(f"[Dia2] TTS error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        output_queue.put(None)


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/stream_tts")
async def stream_tts(ws: WebSocket):
    """WebSocket endpoint - single conversation mode.
    
    Protocol:
    1. Connect -> {"event": "ready", "sample_rate": 24000}
    2. Set voice (ONCE, does pre-warming): {"type": "set_voice", "speaker_1": "<base64>"}
       Response: {"event": "voice_ready"} (after warmup completes)
    3. TTS (instant start): {"type": "tts", "text": "[S1] Hello!"}
       Response: Binary audio chunks, then {"event": "done"}
    """
    await ws.accept()
    print("[Dia2] WebSocket connected")
    
    temp_files: List[str] = []
    
    async def keepalive():
        try:
            while True:
                await asyncio.sleep(10)
                try:
                    await ws.send_text(json.dumps({"event": "ping"}))
                except:
                    break
        except asyncio.CancelledError:
            pass
    
    keepalive_task = asyncio.create_task(keepalive())
    
    try:
        await ws.send_text(json.dumps({
            "event": "ready",
            "sample_rate": dia.sample_rate,
        }))
        
        while True:
            try:
                msg = await ws.receive_text()
            except WebSocketDisconnect:
                break
            
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"error": "Invalid JSON"}))
                continue
            
            msg_type = payload.get("type", "tts")
            
            if msg_type == "pong":
                continue
            
            if msg_type == "set_voice":
                global _prewarmed_state
                
                if payload.get("speaker_1"):
                    suffix = payload.get("format_1", ".wav")
                    if not suffix.startswith("."):
                        suffix = "." + suffix
                    voice_path = save_audio_from_base64(payload["speaker_1"], suffix)
                    temp_files.append(voice_path)
                    
                    # Pre-warm in executor (this takes ~3s but only happens once)
                    await ws.send_text(json.dumps({"event": "warming", "message": "Pre-warming voice..."}))
                    
                    loop = asyncio.get_running_loop()
                    try:
                        await loop.run_in_executor(_cuda_executor, _prewarm_voice, voice_path)
                        await ws.send_text(json.dumps({"event": "voice_ready"}))
                    except Exception as e:
                        await ws.send_text(json.dumps({"error": f"Voice warmup failed: {e}"}))
                else:
                    await ws.send_text(json.dumps({"error": "Missing speaker_1"}))
                continue
            
            if msg_type == "close":
                break
            
            if msg_type == "tts" or "text" in payload:
                text = payload.get("text")
                if not text:
                    await ws.send_text(json.dumps({"error": "Missing text"}))
                    continue
                
                if _prewarmed_state is None:
                    await ws.send_text(json.dumps({"error": "Voice not set. Call set_voice first."}))
                    continue
                
                chunk_queue: Queue = Queue()
                
                loop = asyncio.get_running_loop()
                gen_future = loop.run_in_executor(
                    _cuda_executor,
                    lambda: _run_streaming_tts_prewarmed(
                        text=text,
                        output_queue=chunk_queue,
                        prewarmed=_prewarmed_state,
                        cfg_scale=float(payload.get("cfg_scale", 1.0)),
                        temperature=float(payload.get("temperature", 0.8)),
                        top_k=int(payload.get("top_k", 50)),
                        chunk_frames=int(payload.get("chunk_frames", 12)),
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
    
    except WebSocketDisconnect:
        print("[Dia2] Client disconnected")
    except Exception as e:
        print(f"[Dia2] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        keepalive_task.cancel()
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass
        print("[Dia2] Connection cleaned up")
