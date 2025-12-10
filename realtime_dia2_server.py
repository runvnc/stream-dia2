"""Dia2 TTS Server with TRUE streaming - yields audio chunks during generation.

This version uses run_streaming_generation() to yield audio chunks as they're
generated, rather than waiting for full generation to complete.

Voice samples are cached to avoid re-transcription and re-encoding.
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
from typing import Optional, List, Dict, Any

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
from dia2.runtime.generator import build_initial_state, warmup_with_prefix
from dia2.runtime.streaming_generator import run_streaming_generation, StreamingChunk
from dia2.runtime.script_parser import parse_script


app = FastAPI()

MODEL_REPO = "nari-labs/Dia2-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"

# ============================================================================
# Whisper Model Caching - Load once, use forever
# ============================================================================

_whisper_model = None
_audio_token_cache: Dict[str, torch.Tensor] = {}  # hash -> encoded audio tokens
_prefix_plan_cache: Dict[str, PrefixPlan] = {}  # hash -> full prefix plan
_audio_data_cache: Dict[str, np.ndarray] = {}  # hash -> loaded audio numpy array

# Track file path during encoding for caching
_current_encoding_path: Optional[str] = None

_transcription_cache: Dict[str, List[WhisperWord]] = {}


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
    """Cached version of transcribe_words that reuses the Whisper model."""
    file_hash = _hash_file(audio_path)
    
    if file_hash in _transcription_cache:
        print(f"[Dia2] Using cached transcription for {os.path.basename(audio_path)}")
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


# Monkey-patch the voice_clone module to use our cached version
print("[Dia2] Patching voice_clone.transcribe_words with cached version...")
voice_clone.transcribe_words = _cached_transcribe_words


def _cached_load_audio(audio_path: str, sample_rate: int) -> np.ndarray:
    """Load audio with caching."""
    global _current_encoding_path
    file_hash = _hash_file(audio_path)
    
    # Store path for the encode function to use
    _current_encoding_path = audio_path
    
    if file_hash in _audio_data_cache:
        print(f"[Dia2] Using cached audio data for {os.path.basename(audio_path)}")
        return _audio_data_cache[file_hash]
    
    print(f"[Dia2] Loading audio {os.path.basename(audio_path)}...")
    t0 = time.perf_counter()
    audio = load_mono_audio(audio_path, sample_rate)
    elapsed = time.perf_counter() - t0
    print(f"[Dia2] Audio loaded in {elapsed:.2f}s")
    
    _audio_data_cache[file_hash] = audio
    return audio


def _cached_encode_audio(mimi, audio: np.ndarray) -> torch.Tensor:
    """Encode audio with caching based on current path."""
    global _current_encoding_path
    
    if _current_encoding_path:
        file_hash = _hash_file(_current_encoding_path)
        if file_hash in _audio_token_cache:
            print(f"[Dia2] Using cached audio tokens")
            return _audio_token_cache[file_hash]
    
    print(f"[Dia2] Encoding audio...")
    t0 = time.perf_counter()
    tokens = encode_audio_tokens(mimi, audio)
    elapsed = time.perf_counter() - t0
    print(f"[Dia2] Audio encoded in {elapsed:.2f}s, shape={tokens.shape}")
    
    if _current_encoding_path:
        file_hash = _hash_file(_current_encoding_path)
        _audio_token_cache[file_hash] = tokens
    
    return tokens


# Store original build_prefix_plan
_original_build_prefix_plan = build_prefix_plan

def _cached_build_prefix_plan(runtime, prefix: Optional[PrefixConfig], **kwargs):
    """Wrapper around build_prefix_plan that injects cached functions."""
    if prefix is None:
        return None
    
    # Create encode function that uses our cache
    def cached_encode_fn(audio: np.ndarray) -> torch.Tensor:
        return _cached_encode_audio(runtime.mimi, audio)
    
    result = _original_build_prefix_plan(
        runtime,
        prefix,
        transcribe_fn=_cached_transcribe_words,
        load_audio_fn=_cached_load_audio,
        encode_fn=cached_encode_fn,
        **kwargs
    )
    
    return result

# Monkey-patch build_prefix_plan
print("[Dia2] Patching voice_clone.build_prefix_plan with cached version...")
voice_clone.build_prefix_plan = _cached_build_prefix_plan

# ALSO patch the reference in engine.py (it imports build_prefix_plan directly)
print("[Dia2] Patching dia2.engine.build_prefix_plan...")
dia2_engine.build_prefix_plan = _cached_build_prefix_plan


def _preload_voice(audio_path: str) -> None:
    """Pre-transcribe AND pre-encode audio file to warm all caches.
    
    Call this when voice is set, not when generation starts.
    """
    global _current_encoding_path
    try:
        t0 = time.perf_counter()
        
        # Pre-transcribe (uses Whisper)
        _cached_transcribe_words(audio_path, torch.device(DEVICE))
        t1 = time.perf_counter()
        print(f"[Dia2] Transcription took {t1-t0:.2f}s")
        
        # Pre-load and encode audio tokens (uses Mimi)
        runtime = dia._ensure_runtime()
        
        # Load audio (caches it)
        audio = _cached_load_audio(audio_path, runtime.mimi.sample_rate)
        
        # Encode audio (caches it)
        _cached_encode_audio(runtime.mimi, audio)
        t2 = time.perf_counter()
        print(f"[Dia2] Audio encoding took {t2-t1:.2f}s")
        
        print(f"[Dia2] Total preload time: {t2-t0:.2f}s")
    except Exception as e:
        print(f"[Dia2] Warning: Failed to preload {audio_path}: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Dia2 Model Initialization
# ============================================================================

print(f"[Dia2] Initializing Dia2 from {MODEL_REPO} on {DEVICE} ({DTYPE})...")
dia = Dia2.from_repo(MODEL_REPO, device=DEVICE, dtype=DTYPE)

# Also preload Whisper model at startup
print("[Dia2] Pre-loading Whisper model...")
_get_whisper_model()

_cuda_executor = ThreadPoolExecutor(max_workers=1)


def _warmup_model() -> None:
    """Warm up the model with a simple generation."""
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
# Streaming TTS Generation
# ============================================================================

def _run_streaming_generation(
    text: str,
    output_queue: Queue,
    prefix_speaker_1: Optional[str] = None,
    prefix_speaker_2: Optional[str] = None,
    include_prefix: bool = False,
    cfg_scale: float = 1.0,
    temperature: float = 0.8,
    top_k: int = 50,
    use_cuda_graph: bool = False,
    chunk_frames: int = 12,  # ~1 second at 12.5 Hz
) -> None:
    """Run streaming generation and put chunks into the queue."""
    try:
        t_start = time.perf_counter()
        print(f"[Dia2] Streaming: {text[:80]}...")
        print(f"[Dia2] cfg_scale={cfg_scale}, temp={temperature}, chunk_frames={chunk_frames}")
        
        runtime = dia._ensure_runtime()
        
        # Use same temperature for both text and audio (like CLI)
        sampling = SamplingConfig(temperature=temperature, top_k=top_k)
        base_config = GenerationConfig(
            cfg_scale=cfg_scale,
            text=sampling,
            audio=sampling,
            use_cuda_graph=use_cuda_graph,
        )
        
        # Build prefix plan if voice samples provided
        prefix_plan = None
        if prefix_speaker_1 or prefix_speaker_2:
            prefix_config = PrefixConfig(
                speaker_1=prefix_speaker_1,
                speaker_2=prefix_speaker_2,
            )
            prefix_plan = _cached_build_prefix_plan(runtime, prefix_config)
        
        # Normalize and parse the script
        normalized_text = normalize_script(text)
        
        # Build entries list
        entries = []
        if prefix_plan is not None:
            entries.extend(prefix_plan.entries)
        entries.extend(parse_script(
            [normalized_text],
            runtime.tokenizer,
            runtime.constants,
            runtime.frame_rate
        ))
        
        # Set up state machine
        runtime.machine.initial_padding = base_config.initial_padding
        state = runtime.machine.new_state(entries)
        
        # Build generation state
        gen_state = build_initial_state(runtime, prefix=prefix_plan)
        
        # Warmup with prefix if needed
        start_step = 0
        if prefix_plan is not None:
            start_step = warmup_with_prefix(runtime, prefix_plan, state, gen_state)
            print(f"[Dia2] Prefix warmup done, start_step={start_step}")
        
        include_prefix_audio = include_prefix
        
        t_setup = time.perf_counter()
        print(f"[Dia2] Setup took {t_setup - t_start:.2f}s")
        
        # Run streaming generation
        chunk_count = 0
        total_samples = 0
        for chunk in run_streaming_generation(
            runtime,
            state=state,
            generation=gen_state,
            config=base_config,
            start_step=start_step,
            chunk_frames=chunk_frames,
            include_prefix_audio=include_prefix_audio,
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
        print(f"[Dia2] Generation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        output_queue.put(None)  # Signal completion


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/stream_tts")
async def stream_tts(ws: WebSocket):
    """WebSocket endpoint with persistent connection and TRUE streaming.
    
    Protocol:
    1. Connect -> Server sends {"event": "ready", "sample_rate": 24000}
    2. Set voice: {"type": "set_voice", "speaker_1": "<base64>", "format_1": ".mp3"}
       Response: {"event": "voice_set", "speaker_1": true, "speaker_2": false}
    3. TTS: {"type": "tts", "text": "[S1] Hello!"}
       Response: Binary audio chunks streamed AS THEY'RE GENERATED, then {"event": "done"}
    4. Close: {"type": "close"}
    """
    await ws.accept()
    print("[Dia2] WebSocket connected")
    
    prefix_speaker_1: Optional[str] = None
    prefix_speaker_2: Optional[str] = None
    temp_files: List[str] = []
    
    # Keepalive task
    async def keepalive():
        try:
            while True:
                await asyncio.sleep(10)  # More aggressive for RunPod proxy
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
                if payload.get("clear"):
                    for f in temp_files:
                        try:
                            os.unlink(f)
                        except:
                            pass
                    temp_files = []
                    prefix_speaker_1 = None
                    prefix_speaker_2 = None
                
                if payload.get("speaker_1"):
                    if prefix_speaker_1 and prefix_speaker_1 in temp_files:
                        try:
                            os.unlink(prefix_speaker_1)
                            temp_files.remove(prefix_speaker_1)
                        except:
                            pass
                    suffix = payload.get("format_1", ".wav")
                    if not suffix.startswith("."):
                        suffix = "." + suffix
                    prefix_speaker_1 = save_audio_from_base64(payload["speaker_1"], suffix)
                    temp_files.append(prefix_speaker_1)
                    print(f"[Dia2] Set speaker 1: {prefix_speaker_1}")
                    # Pre-transcribe immediately in background
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(_cuda_executor, _preload_voice, prefix_speaker_1)
                
                if payload.get("speaker_2"):
                    if prefix_speaker_2 and prefix_speaker_2 in temp_files:
                        try:
                            os.unlink(prefix_speaker_2)
                            temp_files.remove(prefix_speaker_2)
                        except:
                            pass
                    suffix = payload.get("format_2", ".wav")
                    if not suffix.startswith("."):
                        suffix = "." + suffix
                    prefix_speaker_2 = save_audio_from_base64(payload["speaker_2"], suffix)
                    temp_files.append(prefix_speaker_2)
                    print(f"[Dia2] Set speaker 2: {prefix_speaker_2}")
                    # Pre-transcribe immediately in background
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(_cuda_executor, _preload_voice, prefix_speaker_2)
                
                await ws.send_text(json.dumps({
                    "event": "voice_set",
                    "speaker_1": prefix_speaker_1 is not None,
                    "speaker_2": prefix_speaker_2 is not None,
                }))
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
                    lambda: _run_streaming_generation(
                        text=text,
                        output_queue=chunk_queue,
                        prefix_speaker_1=prefix_speaker_1,
                        prefix_speaker_2=prefix_speaker_2,
                        include_prefix=bool(payload.get("include_prefix", False)),
                        cfg_scale=float(payload.get("cfg_scale", 1.0)),
                        temperature=float(payload.get("temperature", 0.8)),
                        top_k=int(payload.get("top_k", 50)),
                        use_cuda_graph=bool(payload.get("use_cuda_graph", False)),
                        chunk_frames=int(payload.get("chunk_frames", 12)),
                    )
                )
                
                # Stream chunks as they arrive
                chunks_sent = 0
                while True:
                    try:
                        # Poll queue with timeout
                        chunk = await loop.run_in_executor(
                            None,
                            lambda: chunk_queue.get(timeout=0.05)
                        )
                    except Empty:
                        # Check if generation is done
                        if gen_future.done():
                            # Drain any remaining chunks
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
                    
                    # Send chunk immediately
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
