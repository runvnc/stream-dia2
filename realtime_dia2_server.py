"""Dia2 TTS Server with persistent WebSocket connections and voice conditioning.

This version uses the standard dia2.generate() for reliability, then streams
the result in chunks. Voice samples are cached to avoid re-transcription.
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
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from dia2 import Dia2, GenerationConfig, SamplingConfig


app = FastAPI()

MODEL_REPO = "nari-labs/Dia2-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"

print(f"[Dia2] Initializing Dia2 from {MODEL_REPO} on {DEVICE} ({DTYPE})...")
dia = Dia2.from_repo(MODEL_REPO, device=DEVICE, dtype=DTYPE)

_cuda_executor = ThreadPoolExecutor(max_workers=1)


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


def _generate_tts(
    text: str,
    prefix_speaker_1: Optional[str] = None,
    prefix_speaker_2: Optional[str] = None,
    include_prefix: bool = False,
    cfg_scale: float = 6.0,
    temperature: float = 0.8,
    top_k: int = 50,
) -> List[AudioChunk]:
    """Generate TTS using dia2.generate() and return chunks."""
    print(f"[Dia2] Generating: {text[:50]}...")
    
    cfg = GenerationConfig(
        cfg_scale=cfg_scale,
        text=SamplingConfig(temperature=temperature, top_k=top_k),
        audio=SamplingConfig(temperature=temperature, top_k=top_k),
        use_cuda_graph=True,
    )
    
    result = dia.generate(
        text,
        config=cfg,
        output_wav=None,
        prefix_speaker_1=prefix_speaker_1,
        prefix_speaker_2=prefix_speaker_2,
        include_prefix=include_prefix,
        verbose=False,
    )
    
    chunks = waveform_to_pcm16_chunks(result.waveform, result.sample_rate, chunk_ms=100)
    print(f"[Dia2] Generated {len(chunks)} chunks")
    return chunks


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
    
    # Keepalive
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
                msg = await asyncio.wait_for(ws.receive_text(), timeout=300)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"event": "ping"}))
                continue
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
                    if prefix_speaker_1 in temp_files:
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
                
                if payload.get("speaker_2"):
                    if prefix_speaker_2 in temp_files:
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
                            temperature=float(payload.get("temperature", 0.8)),
                            top_k=int(payload.get("top_k", 50)),
                        )
                    )
                except Exception as e:
                    print(f"[Dia2] Generation error: {e}")
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
        print("[Dia2] Disconnected")
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
        print("[Dia2] Cleaned up")
