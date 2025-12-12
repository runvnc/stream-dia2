"""Dia2 Streaming TTS Server with persistent connections and voice conditioning."""
import argparse
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
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Parse args early so seed can be set before model load
def _parse_args():
    parser = argparse.ArgumentParser(description="Dia2 Streaming TTS Server")
    parser.add_argument("--port", type=int, default=3030, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation")
    parser.add_argument("--prefix-audio", type=str, default=None, help="Path to prefix audio for voice cloning (Speaker 1)")
    parser.add_argument("--prefix-speaker-2", type=str, default=None, help="Path to prefix audio for Speaker 2")
    parser.add_argument("--max-prefix-duration", type=float, default=0, help="Max prefix duration in seconds (0 to disable truncation)")
    # Only parse known args to avoid conflicts with uvicorn
    args, _ = parser.parse_known_args()
    return args

_args = _parse_args()

# Set seed BEFORE model loading
if _args.seed is not None:
    print(f"[Dia2] Setting random seed to {_args.seed}")
    torch.manual_seed(_args.seed)
    torch.cuda.manual_seed(_args.seed)
    np.random.seed(_args.seed)

from dia2 import Dia2, GenerationConfig, SamplingConfig
from dia2.runtime.generator import build_initial_state, warmup_with_prefix
from dia2.runtime.streaming_generator import run_streaming_generation, StreamingChunk, CachedGraphs
from dia2.runtime.script_parser import parse_script
from dia2.runtime.voice_clone import build_prefix_plan
from dia2.runtime.state_machine import Entry
from dia2.generation import merge_generation_config, normalize_script, PrefixConfig


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

# Cache for prefix plans (keyed by hash of voice file paths)
_prefix_cache: Dict[str, Any] = {}

# Store seed for use in generation
_global_seed = _args.seed

# Store prefix paths from command line
_default_prefix_speaker_1 = _args.prefix_audio
_default_prefix_speaker_2 = _args.prefix_speaker_2
_default_max_prefix_duration = _args.max_prefix_duration


def _set_seed_if_needed():
    """Reset seed before each generation for reproducibility."""
    if _global_seed is not None:
        torch.manual_seed(_global_seed)
        torch.cuda.manual_seed(_global_seed)
        np.random.seed(_global_seed)


def _warmup_model() -> None:
    """Warmup with CUDA graphs to compile kernels (graphs themselves can't be reused)."""
    try:
        _set_seed_if_needed()
        cfg = GenerationConfig(
            cfg_scale=1.0,
            text=SamplingConfig(temperature=0.6, top_k=50),
            audio=SamplingConfig(temperature=0.8, top_k=50),
            use_cuda_graph=True,  # Enable for faster generation
        )
        print("[Dia2] Running STREAMING warm-up generation...")
        
        # Use the streaming generator for warmup so those kernels get compiled
        runtime = dia._ensure_runtime()
        normalized_text = normalize_script("[S1] Hello.")
        entries = list(parse_script([normalized_text], runtime.tokenizer, runtime.constants, runtime.frame_rate))
        runtime.machine.initial_padding = cfg.initial_padding
        state = runtime.machine.new_state(entries)
        gen_state = build_initial_state(runtime, prefix=None)
        
        # Run through streaming generator to compile CUDA graphs
        for chunk in run_streaming_generation(
            runtime,
            state=state,
            generation=gen_state,
            config=cfg,
            start_step=0,
            chunk_frames=1,
        ):
            pass  # Just run through to compile kernels
        
        print("[Dia2] Streaming warm-up complete.")
    except Exception as e:
        print(f"[Dia2] Warm-up failed: {e}")


_cuda_executor.submit(_warmup_model).result()


@dataclass
class AudioChunk:
    pcm16: bytes
    sample_rate: int
    is_last: bool


def waveform_to_pcm16(waveform: torch.Tensor) -> bytes:
    wav_np = waveform.detach().cpu().numpy().astype(np.float32)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    pcm16 = (wav_np * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def save_audio_from_base64(audio_b64: str, suffix: str = ".wav") -> str:
    """Decode base64 audio and save to temp file. Returns path."""
    audio_bytes = base64.b64decode(audio_b64)
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, audio_bytes)
    os.close(fd)
    return path


def _get_cache_key(speaker_1: Optional[str], speaker_2: Optional[str]) -> str:
    """Generate cache key from voice file paths."""
    key = f"{speaker_1 or ''}|{speaker_2 or ''}"
    return hashlib.md5(key.encode()).hexdigest()


def _truncate_prefix_plan(prefix_plan, max_duration_sec: float, frame_rate: float = 12.5):
    """
    Truncate a prefix plan to a maximum duration.
    This reduces text context pollution while keeping voice characteristics.
    
    Args:
        prefix_plan: The original PrefixPlan object
        max_duration_sec: Maximum duration in seconds
        frame_rate: Frames per second (default 12.5 for Mimi)
    
    Returns:
        A new PrefixPlan with truncated audio and entries
    """
    if prefix_plan is None:
        return None
    
    max_frames = int(max_duration_sec * frame_rate)
    
    # If already short enough, return as-is
    if prefix_plan.aligned_frames <= max_frames:
        print(f"[Dia2] Prefix already <= {max_duration_sec}s ({prefix_plan.aligned_frames} frames)")
        return prefix_plan
    
    # Truncate audio tokens
    truncated_tokens = prefix_plan.aligned_tokens[:, :max_frames]
    
    # Filter entries to only those that fit within max_frames
    # We need to estimate which entries fit based on new_word_steps
    truncated_entries = []
    truncated_steps = []
    
    for entry, step in zip(prefix_plan.entries, prefix_plan.new_word_steps):
        if step < max_frames:
            truncated_entries.append(entry)
            truncated_steps.append(step)
    
    # Create a new PrefixPlan-like object
    from dataclasses import dataclass
    from typing import List
    
    class TruncatedPrefixPlan:
        def __init__(self, entries, new_word_steps, aligned_tokens, aligned_frames):
            self.entries = entries
            self.new_word_steps = new_word_steps
            self.aligned_tokens = aligned_tokens
            self.aligned_frames = aligned_frames
    
    print(f"[Dia2] Truncated prefix: {prefix_plan.aligned_frames} -> {max_frames} frames, {len(prefix_plan.entries)} -> {len(truncated_entries)} entries")
    
    return TruncatedPrefixPlan(truncated_entries, truncated_steps, truncated_tokens, max_frames)


def _build_and_cache_prefix(
    runtime,
    prefix_speaker_1: Optional[str],
    prefix_speaker_2: Optional[str],
) -> Any:
    """Build prefix plan, using cache if available."""
    cache_key = _get_cache_key(prefix_speaker_1, prefix_speaker_2)
    
    if cache_key in _prefix_cache:
        print(f"[Dia2] Using cached prefix plan")
        return _prefix_cache[cache_key]
    
    # Build prefix config
    prefix_config = PrefixConfig(
        speaker_1=prefix_speaker_1,
        speaker_2=prefix_speaker_2,
    )
    
    print(f"[Dia2] Building prefix plan (will be cached)...")
    prefix_plan = build_prefix_plan(runtime, prefix_config)
    
    if prefix_plan is not None:
        _prefix_cache[cache_key] = prefix_plan
        print(f"[Dia2] Prefix plan cached")
    
    return prefix_plan


def _run_streaming_generation(
    text: str,
    output_queue: Queue,
    prefix_speaker_1: Optional[str] = None,
    prefix_speaker_2: Optional[str] = None,
    include_prefix: bool = False,
    max_prefix_duration: float = 1.0,
    cfg_scale: float = 6.0,
    temperature: float = 0.8,
    top_k: int = 50,
    use_cuda_graph: bool = True,
    chunk_frames: int = 15,
) -> None:
    """Run streaming generation and put chunks into the queue."""
    try:
        _set_seed_if_needed()  # Reset seed for reproducible generation
        
        print(f"[Dia2] Generating: {text[:50]}...")
        runtime = dia._ensure_runtime()
        
        base_config = GenerationConfig(
            cfg_scale=cfg_scale,
            text=SamplingConfig(temperature=temperature, top_k=top_k),
            audio=SamplingConfig(temperature=temperature, top_k=top_k),
            use_cuda_graph=True,  # Enable for faster generation
        )
        
        # Get cached prefix plan
        prefix_plan = None
        if prefix_speaker_1 or prefix_speaker_2:
            prefix_plan = _build_and_cache_prefix(runtime, prefix_speaker_1, prefix_speaker_2)
        
        # Truncate prefix to minimize text context pollution
        if prefix_plan is not None and max_prefix_duration > 0 and prefix_plan.aligned_frames > int(max_prefix_duration * runtime.frame_rate):
            prefix_plan = _truncate_prefix_plan(prefix_plan, max_prefix_duration, runtime.frame_rate) 
            print(f"[Dia2] Using truncated prefix: {prefix_plan.aligned_frames} frames ({prefix_plan.aligned_frames / runtime.frame_rate:.2f}s)")

        normalized_text = normalize_script(text)
        
        # Parse new text entries
        new_entries = list(parse_script(
            [normalized_text], 
            runtime.tokenizer, 
            runtime.constants, 
            runtime.frame_rate
        ))
        
        # FIX: Use ONE state machine with ALL entries (prefix + new text)
        # This matches the original Dia2 CLI behavior
        all_entries = []
        if prefix_plan is not None:
            all_entries.extend(prefix_plan.entries)
        all_entries.extend(new_entries)
        
        runtime.machine.initial_padding = base_config.initial_padding
        state = runtime.machine.new_state(all_entries)  # SINGLE state machine
        
        print(f"[Dia2] DEBUG: Total entries = {len(all_entries)}")
        print(f"[Dia2] DEBUG: Prefix entries = {len(prefix_plan.entries) if prefix_plan else 0}")
        print(f"[Dia2] DEBUG: New entries = {len(new_entries)}")
        
        gen_state = build_initial_state(runtime, prefix=prefix_plan)

        start_step = 0
        if prefix_plan is not None:
            # Warmup with the SAME state - prefix entries will be consumed
            start_step = warmup_with_prefix(runtime, prefix_plan, state, gen_state)
            print(f"[Dia2] Prefix warmup done, start_step={start_step}")
            print(f"[Dia2] DEBUG: State after warmup - entries remaining: {len(state.entries)}")
            print(f"[Dia2] DEBUG: State transcript so far: {state.transcript[:5]}...")  # First 5 words
        
        include_prefix_audio = include_prefix
        
        print(f"[Dia2] Starting streaming generation: chunk_frames=1, cfg_scale={cfg_scale}, temp={temperature}")
        chunk_count = 0
        for chunk in run_streaming_generation(
            runtime,
            state=state,  # Use the SAME state that was warmed up!
            generation=gen_state,
            config=base_config,
            start_step=start_step,
            chunk_frames=1,  # Decode every frame for lowest latency
            include_prefix_audio=include_prefix_audio,
        ):
            chunk_count += 1
            pcm16_bytes = waveform_to_pcm16(chunk.waveform)
            output_queue.put(AudioChunk(
                pcm16=pcm16_bytes,
                sample_rate=chunk.sample_rate,
                is_last=chunk.is_final,
            ))
            if chunk.is_final:
                break
        
        print(f"[Dia2] Done: {chunk_count} chunks")
                
    except Exception as e:
        print(f"[Dia2] Generation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        output_queue.put(None)


@app.websocket("/ws/stream_tts")
async def stream_tts(ws: WebSocket):
    """WebSocket endpoint with persistent connection support."""
    await ws.accept()
    print("[Dia2] WebSocket connected")
    
    # Session state
    # Use command-line defaults if provided
    prefix_speaker_1: Optional[str] = _default_prefix_speaker_1
    prefix_speaker_2: Optional[str] = _default_prefix_speaker_2
    temp_files: List[str] = []
    
    # Keepalive task
    async def keepalive():
        try:
            while True:
                await asyncio.sleep(30)
                await ws.send_text(json.dumps({"event": "ping"}))
        except:
            pass
    
    keepalive_task = asyncio.create_task(keepalive())
    
    try:
        await ws.send_text(json.dumps({
            "event": "ready",
            "sample_rate": dia.sample_rate,
        }))
        
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=120)
            except asyncio.TimeoutError:
                print("[Dia2] Connection timeout, sending ping")
                await ws.send_text(json.dumps({"event": "ping"}))
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
            
            # Handle pong
            if msg_type == "pong":
                continue
            
            # Handle set_voice command
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
                    # Clear cache for this session's voices
                    print("[Dia2] Cleared all voices")
                
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
                    path = save_audio_from_base64(payload["speaker_1"], suffix)
                    prefix_speaker_1 = path
                    temp_files.append(path)
                    print(f"[Dia2] Set speaker 1 voice: {path}")
                
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
                    path = save_audio_from_base64(payload["speaker_2"], suffix)
                    prefix_speaker_2 = path
                    temp_files.append(path)
                    print(f"[Dia2] Set speaker 2 voice: {path}")
                
                await ws.send_text(json.dumps({
                    "event": "voice_set",
                    "speaker_1": prefix_speaker_1 is not None,
                    "speaker_2": prefix_speaker_2 is not None,
                }))
                continue
            
            if msg_type == "close":
                break
            
            # Handle TTS request
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
                        max_prefix_duration=float(payload.get("max_prefix_duration", _default_max_prefix_duration)),
                        cfg_scale=float(payload.get("cfg_scale", 6.0)),
                        temperature=float(payload.get("temperature", 0.8)),
                        top_k=int(payload.get("top_k", 50)),
                        chunk_frames=int(payload.get("chunk_frames", 15)),
                    )
                )
                
                chunks_sent = 0
                while True:
                    try:
                        chunk = await loop.run_in_executor(
                            None,
                            lambda: chunk_queue.get(timeout=0.1)
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
        print("[Dia2] Connection closed, cleaned up temp files")


if __name__ == "__main__":
    import uvicorn
    print(f"[Dia2] Starting server on {_args.host}:{_args.port}")
    uvicorn.run(app, host=_args.host, port=_args.port)
