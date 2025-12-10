"""Dia2 TTS Server with pre-warmed voice state for minimal latency.

Single conversation mode - pre-warms once, reuses state for each TTS request.
Requires BOTH speaker_1 and speaker_2 for voice cloning to work.
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
from dia2.runtime.generator import build_initial_state, warmup_with_prefix, GenerationState
from dia2.runtime.streaming_generator import run_streaming_generation, StreamingChunk
from dia2.runtime.script_parser import parse_script


app = FastAPI()

MODEL_REPO = "nari-labs/Dia2-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"

# ============================================================================
# Pre-warmed Voice State - Single conversation mode
# ============================================================================

@dataclass 
class VoiceSession:
    """Holds the warmed state for a voice - reused across TTS requests."""
    prefix_plan: PrefixPlan
    gen_state: GenerationState
    start_step: int
    prefix_cache_length: int
    prefix_audio_tokens: torch.Tensor


_voice_session: Optional[VoiceSession] = None


# ============================================================================
# Whisper Model Caching
# ============================================================================

_whisper_model = None
_audio_token_cache: Dict[str, torch.Tensor] = {}
_audio_data_cache: Dict[str, np.ndarray] = {}
_transcription_cache: Dict[str, List[WhisperWord]] = {}
_current_encoding_path: Optional[str] = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper_timestamped as wts
        print("[Dia2] Loading Whisper model (one-time)...")
        _whisper_model = wts.load_model("openai/whisper-large-v3", device=DEVICE)
        print("[Dia2] Whisper model loaded.")
    return _whisper_model


def _hash_file(path: str) -> str:
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
    global _current_encoding_path
    file_hash = _hash_file(audio_path)
    _current_encoding_path = audio_path
    
    if file_hash in _audio_data_cache:
        return _audio_data_cache[file_hash]
    
    print(f"[Dia2] Loading audio {os.path.basename(audio_path)}...")
    audio = load_mono_audio(audio_path, sample_rate)
    _audio_data_cache[file_hash] = audio
    return audio


def _cached_encode_audio(mimi, audio: np.ndarray) -> torch.Tensor:
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
# Voice Session Management
# ============================================================================

def _create_voice_session(speaker_1_path: str, speaker_2_path: str) -> VoiceSession:
    """Create and warm up a voice session with BOTH speakers."""
    global _voice_session
    
    t0 = time.perf_counter()
    print(f"[Dia2] Creating voice session...")
    print(f"[Dia2]   Speaker 1: {speaker_1_path}")
    print(f"[Dia2]   Speaker 2: {speaker_2_path}")
    
    runtime = dia._ensure_runtime()
    
    # Build prefix plan with BOTH speakers
    prefix_config = PrefixConfig(
        speaker_1=speaker_1_path,
        speaker_2=speaker_2_path,
    )
    prefix_plan = _cached_build_prefix_plan(runtime, prefix_config)
    
    if prefix_plan is None:
        raise ValueError("Failed to build prefix plan")
    
    # Build generation state
    gen_state = build_initial_state(runtime, prefix=prefix_plan)
    
    # Create state machine for warmup
    runtime.machine.initial_padding = 0
    warmup_state = runtime.machine.new_state(prefix_plan.entries)
    
    # Run warmup
    start_step = warmup_with_prefix(runtime, prefix_plan, warmup_state, gen_state)
    
    # Get cache length after warmup
    prefix_cache_length = gen_state.decode.transformer.slots[0].length.item()
    
    # Save prefix audio tokens
    prefix_len = prefix_plan.aligned_frames + 10
    prefix_audio_tokens = gen_state.audio_buf[:, :, :prefix_len].clone()
    
    elapsed = time.perf_counter() - t0
    print(f"[Dia2] Voice session ready in {elapsed:.2f}s (start_step={start_step}, cache_len={prefix_cache_length})")
    
    _voice_session = VoiceSession(
        prefix_plan=prefix_plan,
        gen_state=gen_state,
        start_step=start_step,
        prefix_cache_length=prefix_cache_length,
        prefix_audio_tokens=prefix_audio_tokens,
    )
    
    return _voice_session


def _reset_session_for_new_tts(session: VoiceSession) -> None:
    """Reset the session state for a new TTS request."""
    runtime = dia._ensure_runtime()
    
    # Reset transformer cache length
    for slot in session.gen_state.decode.transformer.slots:
        slot.length.fill_(session.prefix_cache_length)
    
    # Reset depformer cache
    session.gen_state.decode.depformer.reset()
    
    # Restore prefix audio tokens
    prefix_len = session.prefix_audio_tokens.shape[2]
    session.gen_state.audio_buf[:, :, :prefix_len].copy_(session.prefix_audio_tokens)
    session.gen_state.audio_buf[:, :, prefix_len:].fill_(runtime.constants.ungenerated)


def _clear_voice_session() -> None:
    global _voice_session
    _voice_session = None
    print("[Dia2] Voice session cleared")


# ============================================================================
# Audio Processing
# ============================================================================

@dataclass
class AudioChunk:
    pcm16: bytes
    sample_rate: int
    is_last: bool


def waveform_to_pcm16(waveform: torch.Tensor) -> bytes:
    if waveform.ndim != 1:
        waveform = waveform.view(-1)
    wav_np = waveform.detach().cpu().numpy().astype(np.float32)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    pcm16 = (wav_np * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def save_audio_from_base64(audio_b64: str, suffix: str = ".wav") -> str:
    audio_bytes = base64.b64decode(audio_b64)
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, audio_bytes)
    os.close(fd)
    return path


# ============================================================================
# Streaming TTS
# ============================================================================

def _run_streaming_tts(
    text: str,
    output_queue: Queue,
    session: VoiceSession,
    cfg_scale: float = 1.0,
    temperature: float = 0.8,
    top_k: int = 50,
    chunk_frames: int = 1,  # Send every frame immediately
) -> None:
    """Run streaming TTS using the pre-warmed voice session."""
    try:
        t_start = time.perf_counter()
        print(f"[Dia2] TTS: {text[:60]}...")
        
        first_chunk_sent = False
        
        runtime = dia._ensure_runtime()
        
        # Reset session
        _reset_session_for_new_tts(session)
        
        # Config
        sampling = SamplingConfig(temperature=temperature, top_k=top_k)
        config = GenerationConfig(
            cfg_scale=cfg_scale,
            text=sampling,
            audio=sampling,
            use_cuda_graph=False,  # Disabled to test if graph capture is causing first-chunk delay
        )
        
        # Parse text and create state machine
        normalized_text = normalize_script(text)
        entries = list(session.prefix_plan.entries)
        entries.extend(parse_script(
            [normalized_text],
            runtime.tokenizer,
            runtime.constants,
            runtime.frame_rate
        ))
        
        runtime.machine.initial_padding = config.initial_padding
        state = runtime.machine.new_state(entries)
        
        # Fast-forward state machine through prefix
        new_word_steps = set(session.prefix_plan.new_word_steps)
        for t in range(session.start_step + 1):
            forced = runtime.constants.new_word if t in new_word_steps else runtime.constants.pad
            runtime.machine.process(t, state, forced, is_forced=True)
        
        t_setup = time.perf_counter()
        print(f"[Dia2] Setup: {(t_setup - t_start)*1000:.0f}ms")
        
        # Run streaming generation
        chunk_count = 0
        total_samples = 0
        
        for chunk in run_streaming_generation(
            runtime,
            state=state,
            generation=session.gen_state,
            config=config,
            start_step=session.start_step,
            chunk_frames=chunk_frames,
            include_prefix_audio=False,
        ):
            chunk_count += 1
            pcm16_bytes = waveform_to_pcm16(chunk.waveform)
            total_samples += chunk.waveform.shape[0]
            
            if not first_chunk_sent:
                first_chunk_sent = True
                print(f"[Dia2] First chunk ready: {(time.perf_counter() - t_start)*1000:.0f}ms")
            
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
    2. Set voice (requires BOTH speakers): 
       {"type": "set_voice", "speaker_1": "<base64>", "speaker_2": "<base64>"}
       Response: {"event": "voice_ready"}
    3. TTS: {"type": "tts", "text": "[S1] Hello!"}
       Response: Binary audio chunks, then {"event": "done"}
    4. Clear: {"type": "clear"}
    5. Close: {"type": "close"}
    """
    await ws.accept()
    print("[Dia2] WebSocket connected")
    
    temp_files: List[str] = []
    
    async def keepalive():
        try:
            while True:
                await asyncio.sleep(5)  # More frequent for RunPod
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
                msg = await asyncio.wait_for(ws.receive_text(), timeout=60)
            except asyncio.TimeoutError:
                # Send ping on timeout to keep connection alive
                try:
                    await ws.send_text(json.dumps({"event": "ping"}))
                except:
                    break
                continue
            except WebSocketDisconnect:
                print("[Dia2] Client disconnected")
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
                speaker_1 = payload.get("speaker_1")
                speaker_2 = payload.get("speaker_2")
                
                if not speaker_1 or not speaker_2:
                    await ws.send_text(json.dumps({
                        "error": "Both speaker_1 and speaker_2 are required for voice cloning"
                    }))
                    continue
                
                # Save audio files
                suffix_1 = payload.get("format_1", ".wav")
                suffix_2 = payload.get("format_2", ".wav")
                if not suffix_1.startswith("."): suffix_1 = "." + suffix_1
                if not suffix_2.startswith("."): suffix_2 = "." + suffix_2
                
                voice_path_1 = save_audio_from_base64(speaker_1, suffix_1)
                voice_path_2 = save_audio_from_base64(speaker_2, suffix_2)
                temp_files.extend([voice_path_1, voice_path_2])
                
                await ws.send_text(json.dumps({"event": "warming", "message": "Pre-warming voices..."}))
                
                loop = asyncio.get_running_loop()
                try:
                    await loop.run_in_executor(
                        _cuda_executor, 
                        _create_voice_session, 
                        voice_path_1, 
                        voice_path_2
                    )
                    await ws.send_text(json.dumps({"event": "voice_ready"}))
                except Exception as e:
                    print(f"[Dia2] Voice setup error: {e}")
                    import traceback
                    traceback.print_exc()
                    await ws.send_text(json.dumps({"error": f"Voice setup failed: {e}"}))
                continue
            
            if msg_type == "clear":
                _clear_voice_session()
                await ws.send_text(json.dumps({"event": "cleared"}))
                continue
            
            if msg_type == "close":
                break
            
            if msg_type == "tts" or "text" in payload:
                text = payload.get("text")
                if not text:
                    await ws.send_text(json.dumps({"error": "Missing text"}))
                    continue
                
                if _voice_session is None:
                    await ws.send_text(json.dumps({"error": "Voice not set. Call set_voice with both speakers first."}))
                    continue
                
                chunk_queue: Queue = Queue()
                
                loop = asyncio.get_running_loop()
                gen_future = loop.run_in_executor(
                    _cuda_executor,
                    lambda: _run_streaming_tts(
                        text=text,
                        output_queue=chunk_queue,
                        session=_voice_session,
                        cfg_scale=float(payload.get("cfg_scale", 1.0)),
                        temperature=float(payload.get("temperature", 0.8)),
                        top_k=int(payload.get("top_k", 50)),
                        chunk_frames=int(payload.get("chunk_frames", 12)),
                    )
                )
                
                chunks_sent = 0
                try:
                    while True:
                        try:
                            chunk = await loop.run_in_executor(
                                None,
                                lambda: chunk_queue.get(timeout=0.05)
                            )
                        except Empty:
                            if gen_future.done():
                                # Drain remaining
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
                except Exception as e:
                    print(f"[Dia2] Error sending chunks: {e}")
                
                await ws.send_text(json.dumps({"event": "done", "chunks": chunks_sent}))
                print(f"[Dia2] Sent {chunks_sent} chunks")
                continue
            
            await ws.send_text(json.dumps({"error": f"Unknown type: {msg_type}"}))
    
    except WebSocketDisconnect:
        print("[Dia2] WebSocket disconnected")
    except Exception as e:
        print(f"[Dia2] WebSocket error: {e}")
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
