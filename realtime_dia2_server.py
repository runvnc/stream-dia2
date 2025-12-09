"""Dia2 Streaming TTS Server with persistent connections and voice conditioning."""
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
from fastapi.responses import JSONResponse

from dia2 import Dia2, GenerationConfig, SamplingConfig
from dia2.runtime.generator import build_initial_state, warmup_with_prefix
from dia2.runtime.streaming_generator import run_streaming_generation, StreamingChunk
from dia2.runtime.script_parser import parse_script
from dia2.runtime.voice_clone import build_prefix_plan
from dia2.generation import merge_generation_config, normalize_script, PrefixConfig


app = FastAPI()

MODEL_REPO = "nari-labs/Dia2-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"

print(f"[Dia2] Initializing Dia2 from {MODEL_REPO} on {DEVICE} ({DTYPE})...")
dia = Dia2.from_repo(MODEL_REPO, device=DEVICE, dtype=DTYPE)

_cuda_executor = ThreadPoolExecutor(max_workers=1)

# Cache for prefix plans (keyed by hash of voice file paths)
_prefix_cache: Dict[str, Any] = {}


def _warmup_model() -> None:
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
    cfg_scale: float = 6.0,
    temperature: float = 0.8,
    top_k: int = 50,
    use_cuda_graph: bool = True,
    chunk_frames: int = 15,
) -> None:
    """Run streaming generation and put chunks into the queue."""
    try:
        print(f"[Dia2] Generating: {text[:50]}...")
        runtime = dia._ensure_runtime()
        
        base_config = GenerationConfig(
            cfg_scale=cfg_scale,
            text=SamplingConfig(temperature=temperature, top_k=top_k),
            audio=SamplingConfig(temperature=temperature, top_k=top_k),
            use_cuda_graph=use_cuda_graph,
        )
        
        # Get cached prefix plan
        prefix_plan = None
        if prefix_speaker_1 or prefix_speaker_2:
            prefix_plan = _build_and_cache_prefix(runtime, prefix_speaker_1, prefix_speaker_2)
        
        normalized_text = normalize_script(text)
        
        entries = []
        if prefix_plan is not None:
            entries.extend(prefix_plan.entries)
        entries.extend(parse_script(
            [normalized_text], 
            runtime.tokenizer, 
            runtime.constants, 
            runtime.frame_rate
        ))
        
        runtime.machine.initial_padding = base_config.initial_padding
        state = runtime.machine.new_state(entries)
        gen_state = build_initial_state(runtime, prefix=prefix_plan)
        
        start_step = 0
        if prefix_plan is not None:
            start_step = warmup_with_prefix(runtime, prefix_plan, state, gen_state)
            print(f"[Dia2] Prefix warmup done, start_step={start_step}")
        
        include_prefix_audio = include_prefix
        
        chunk_count = 0
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
    prefix_speaker_1: Optional[str] = None
    prefix_speaker_2: Optional[str] = None
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
