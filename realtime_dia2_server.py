"""Dia2 TTS Server with persistent WebSocket connections and voice conditioning.

This version uses the standard dia2.generate() for reliability, then streams
the result in chunks. Voice samples are cached to avoid re-transcription
and re-encoding.
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
from dia2.generation import PrefixConfig
from dia2.runtime.voice_clone import WhisperWord


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
    
    # Debug: Check speaker token mapping
    convert = getattr(runtime.tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        s1_id = convert("[S1]")
        s2_id = convert("[S2]")
        print(f"[Dia2] DEBUG: Token IDs - [S1]={s1_id}, [S2]={s2_id}")
        print(f"[Dia2] DEBUG: Constants - spk1={runtime.constants.spk1}, spk2={runtime.constants.spk2}")
        if s1_id != runtime.constants.spk1:
            print(f"[Dia2] WARNING: [S1] token ID mismatch! {s1_id} != {runtime.constants.spk1}")
        if s2_id != runtime.constants.spk2:
            print(f"[Dia2] WARNING: [S2] token ID mismatch! {s2_id} != {runtime.constants.spk2}")
    
    # Debug: Show prefix config
    print(f"[Dia2] DEBUG: PrefixConfig - speaker_1={prefix.speaker_1}, speaker_2={prefix.speaker_2}")
    
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
    
    # Debug: Show what entries were created
    if result:
        print(f"[Dia2] DEBUG: PrefixPlan has {len(result.entries)} entries, {result.aligned_frames} frames")
        for i, entry in enumerate(result.entries[:5]):  # Show first 5
            print(f"[Dia2] DEBUG:   Entry {i}: tokens={entry.tokens[:3]}... text='{entry.text}'")
    
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
            cfg_scale=2.0,
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


def waveform_to_pcm16_chunks(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_ms: int = 100,
) -> List[AudioChunk]:
    """Convert waveform to PCM16 chunks."""
    if waveform.ndim != 1:
        waveform = waveform.view(-1)
    
    wav_np = waveform.detach().cpu().numpy().astype(np.float32)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    pcm16 = (wav_np * 32767.0).astype(np.int16)
    
    samples_per_chunk = int(sample_rate * chunk_ms / 1000)
    num_samples = len(pcm16)
    
    chunks = []
    for start in range(0, num_samples, samples_per_chunk):
        end = min(start + samples_per_chunk, num_samples)
        piece = pcm16[start:end]
        chunks.append(AudioChunk(
            pcm16=piece.tobytes(),
            sample_rate=sample_rate,
            is_last=(end >= num_samples),
        ))
    
    if not chunks:
        chunks.append(AudioChunk(pcm16=b"", sample_rate=sample_rate, is_last=True))
    
    return chunks


def save_audio_from_base64(audio_b64: str, suffix: str = ".wav") -> str:
    """Decode base64 audio and save to temp file."""
    audio_bytes = base64.b64decode(audio_b64)
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, audio_bytes)
    os.close(fd)
    return path


# ============================================================================
# TTS Generation
# ============================================================================

def _generate_tts(
    text: str,
    prefix_speaker_1: Optional[str] = None,
    prefix_speaker_2: Optional[str] = None,
    include_prefix: bool = False,
    cfg_scale: float = 6.0,
    text_temperature: float = 0.6,
    audio_temperature: float = 0.8,
    top_k: int = 50,
) -> List[AudioChunk]:
    """Generate TTS using dia2.generate() and return chunks."""
    t_start = time.perf_counter()
    print(f"[Dia2] Generating: {text[:80]}...")
    print(f"[Dia2] DEBUG _generate_tts: prefix_speaker_1={prefix_speaker_1}")
    print(f"[Dia2] DEBUG _generate_tts: prefix_speaker_2={prefix_speaker_2}")
    print(f"[Dia2] DEBUG _generate_tts: cfg_scale={cfg_scale}")
    print(f"[Dia2] DEBUG _generate_tts: text_temp={text_temperature}, audio_temp={audio_temperature}")
    
    cfg = GenerationConfig(
        cfg_scale=cfg_scale,
        text=SamplingConfig(temperature=text_temperature, top_k=top_k),
        audio=SamplingConfig(temperature=audio_temperature, top_k=top_k),
        use_cuda_graph=True,
    )
    
    t_before_gen = time.perf_counter()
    result = dia.generate(
        text,
        config=cfg,
        output_wav=None,
        prefix_speaker_1=prefix_speaker_1,
        prefix_speaker_2=prefix_speaker_2,
        include_prefix=include_prefix,
        verbose=False,
    )
    t_after_gen = time.perf_counter()
    
    chunks = waveform_to_pcm16_chunks(result.waveform, result.sample_rate, chunk_ms=100)
    t_end = time.perf_counter()
    
    print(f"[Dia2] Timing: setup={t_before_gen-t_start:.2f}s, "
          f"generate={t_after_gen-t_before_gen:.2f}s, "
          f"chunk={t_end-t_after_gen:.2f}s, total={t_end-t_start:.2f}s")
    print(f"[Dia2] Generated {len(chunks)} chunks ({result.waveform.shape[-1]/result.sample_rate:.2f}s audio)")
    return chunks


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/stream_tts")
async def stream_tts(ws: WebSocket):
    """WebSocket endpoint with persistent connection.
    
    Protocol:
    1. Connect -> Server sends {"event": "ready", "sample_rate": 24000}
    2. Set voice: {"type": "set_voice", "speaker_1": "<base64>", "format_1": ".mp3"}
       Response: {"event": "voice_set", "speaker_1": true, "speaker_2": false}
    3. TTS: {"type": "tts", "text": "[S1] Hello!"}
       Response: Binary audio chunks, then {"event": "done"}
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
                await asyncio.sleep(30)  # Increased from 15 to 30 seconds
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
                
                loop = asyncio.get_running_loop()
                try:
                    chunks = await loop.run_in_executor(
                        _cuda_executor,
                        lambda: _generate_tts(
                            text=text,
                            prefix_speaker_1=prefix_speaker_1,
                            prefix_speaker_2=prefix_speaker_2,
                            include_prefix=bool(payload.get("include_prefix", False)),
                            cfg_scale=float(payload.get("cfg_scale", 6.0)),
                            text_temperature=float(payload.get("text_temperature", 0.6)),
                            audio_temperature=float(payload.get("audio_temperature", 0.8)),
                            top_k=int(payload.get("top_k", 50)),
                        )
                    )
                except Exception as e:
                    print(f"[Dia2] Generation error: {e}")
                    import traceback
                    traceback.print_exc()
                    await ws.send_text(json.dumps({"error": str(e)}))
                    continue
                
                # Stream chunks
                for chunk in chunks:
                    header = struct.pack("!?", chunk.is_last)
                    await ws.send_bytes(header + chunk.pcm16)
                
                await ws.send_text(json.dumps({"event": "done", "chunks": len(chunks)}))
                print(f"[Dia2] Sent {len(chunks)} chunks")
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
