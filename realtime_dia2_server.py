"""Dia2 Streaming TTS Server with true streaming support."""
import asyncio
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread
from typing import Optional, List, Iterator

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from dia2 import Dia2, GenerationConfig, SamplingConfig
from dia2.runtime.generator import build_initial_state, warmup_with_prefix
from dia2.runtime.streaming_generator import run_streaming_generation, StreamingChunk
from dia2.runtime.script_parser import parse_script
from dia2.runtime.voice_clone import build_prefix_plan
from dia2.generation import merge_generation_config, normalize_script


# -------------------------------
# Global Dia2 instance & warm-up
# -------------------------------

app = FastAPI()

MODEL_REPO = "nari-labs/Dia2-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"

print(f"[Dia2] Initializing Dia2 from {MODEL_REPO} on {DEVICE} ({DTYPE})...")
dia = Dia2.from_repo(MODEL_REPO, device=DEVICE, dtype=DTYPE)

# Single-thread executor for CUDA operations
_cuda_executor = ThreadPoolExecutor(max_workers=1)


def _warmup_model() -> None:
    """Warm up the model in the CUDA thread."""
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


# -------------------------------
# Utility functions
# -------------------------------

@dataclass
class AudioChunk:
    pcm16: bytes
    sample_rate: int
    is_last: bool


def waveform_to_pcm16(waveform: torch.Tensor) -> bytes:
    """Convert waveform tensor to PCM16 bytes."""
    wav_np = waveform.detach().cpu().numpy().astype(np.float32)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    pcm16 = (wav_np * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def waveform_to_chunks(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_ms: int = 80,
) -> List[AudioChunk]:
    """Split waveform into PCM16 chunks."""
    if waveform.ndim != 1:
        waveform = waveform.view(-1)

    num_samples = waveform.shape[0]
    samples_per_chunk = int(sample_rate * chunk_ms / 1000)
    if samples_per_chunk <= 0:
        samples_per_chunk = max(sample_rate // 50, 1)

    wav_np = waveform.detach().cpu().numpy().astype(np.float32)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    pcm16 = (wav_np * 32767.0).astype(np.int16)

    chunks: List[AudioChunk] = []
    for start in range(0, num_samples, samples_per_chunk):
        end = min(start + samples_per_chunk, num_samples)
        piece = pcm16[start:end]
        if piece.size == 0:
            continue
        chunks.append(
            AudioChunk(
                pcm16=piece.tobytes(),
                sample_rate=sample_rate,
                is_last=(end >= num_samples),
            )
        )

    if not chunks:
        chunks.append(AudioChunk(pcm16=b"", sample_rate=sample_rate, is_last=True))
    return chunks


# -------------------------------
# Streaming generation
# -------------------------------

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
    chunk_frames: int = 15,  # ~1.2 seconds per chunk
) -> None:
    """
    Run streaming generation and put chunks into the queue.
    This runs in the CUDA executor thread.
    """
    try:
        runtime = dia._ensure_runtime()
        
        # Build config
        base_config = GenerationConfig(
            cfg_scale=cfg_scale,
            text=SamplingConfig(temperature=temperature, top_k=top_k),
            audio=SamplingConfig(temperature=temperature, top_k=top_k),
            use_cuda_graph=use_cuda_graph,
        )
        
        # Handle prefix/voice cloning
        overrides = {}
        if prefix_speaker_1:
            overrides["prefix_speaker_1"] = prefix_speaker_1
        if prefix_speaker_2:
            overrides["prefix_speaker_2"] = prefix_speaker_2
        if include_prefix:
            overrides["include_prefix"] = include_prefix
            
        merged = merge_generation_config(base=base_config, overrides=overrides)
        
        # Parse script
        normalized_text = normalize_script(text)
        prefix_plan = build_prefix_plan(runtime, merged.prefix)
        
        entries = []
        if prefix_plan is not None:
            entries.extend(prefix_plan.entries)
        entries.extend(parse_script(
            [normalized_text], 
            runtime.tokenizer, 
            runtime.constants, 
            runtime.frame_rate
        ))
        
        # Initialize state
        runtime.machine.initial_padding = merged.initial_padding
        state = runtime.machine.new_state(entries)
        gen_state = build_initial_state(runtime, prefix=prefix_plan)
        
        # Warmup with prefix if needed
        start_step = 0
        if prefix_plan is not None:
            start_step = warmup_with_prefix(runtime, prefix_plan, state, gen_state)
        
        # Run streaming generation
        for chunk in run_streaming_generation(
            runtime,
            state=state,
            generation=gen_state,
            config=merged,
            start_step=start_step,
            chunk_frames=chunk_frames,
        ):
            # Convert to PCM16 and queue
            pcm16_bytes = waveform_to_pcm16(chunk.waveform)
            output_queue.put(AudioChunk(
                pcm16=pcm16_bytes,
                sample_rate=chunk.sample_rate,
                is_last=chunk.is_final,
            ))
            
            if chunk.is_final:
                break
                
    except Exception as e:
        print(f"[Dia2] Streaming generation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Signal completion
        output_queue.put(None)


# -------------------------------
# Non-streaming generation (for REST API)
# -------------------------------

def _generate_audio(
    text: str,
    prefix_speaker_1: Optional[str] = None,
    prefix_speaker_2: Optional[str] = None,
    include_prefix: bool = False,
    cfg_scale: float = 6.0,
    temperature: float = 0.8,
    top_k: int = 50,
    use_cuda_graph: bool = True,
) -> List[AudioChunk]:
    """Generate audio (non-streaming). Runs in CUDA executor."""
    cfg = GenerationConfig(
        cfg_scale=cfg_scale,
        text=SamplingConfig(temperature=temperature, top_k=top_k),
        audio=SamplingConfig(temperature=temperature, top_k=top_k),
        use_cuda_graph=use_cuda_graph,
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

    return waveform_to_chunks(result.waveform, result.sample_rate, chunk_ms=80)


# -------------------------------
# REST endpoint
# -------------------------------

@app.post("/tts_once")
async def tts_once(payload: dict):
    text = payload.get("text")
    if not text:
        return JSONResponse({"error": "Missing 'text'"}, status_code=400)

    loop = asyncio.get_running_loop()
    
    try:
        chunks = await loop.run_in_executor(
            _cuda_executor,
            lambda: _generate_audio(
                text=text,
                prefix_speaker_1=payload.get("prefix_speaker_1"),
                prefix_speaker_2=payload.get("prefix_speaker_2"),
                include_prefix=bool(payload.get("include_prefix", False)),
                cfg_scale=float(payload.get("cfg_scale", 6.0)),
                temperature=float(payload.get("temperature", 0.8)),
                top_k=int(payload.get("top_k", 50)),
            )
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    all_pcm = b"".join(ch.pcm16 for ch in chunks)
    sample_rate = chunks[0].sample_rate if chunks else 44100

    return {
        "sample_rate": sample_rate,
        "pcm16_hex": all_pcm.hex(),
    }


# -------------------------------
# WebSocket streaming endpoint
# -------------------------------

@app.websocket("/ws/stream_tts")
async def stream_tts(ws: WebSocket):
    await ws.accept()
    try:
        msg = await ws.receive_text()
        try:
            payload = json.loads(msg)
        except json.JSONDecodeError:
            await ws.send_text(json.dumps({"error": "Invalid JSON"}))
            await ws.close(code=1003)
            return

        text = payload.get("text")
        if not text:
            await ws.send_text(json.dumps({"error": "Missing 'text'"}))
            await ws.close(code=1003)
            return

        # Send config
        await ws.send_text(json.dumps({
            "event": "config", 
            "sample_rate": dia.sample_rate,
            "streaming": True,
        }))

        # Create queue for streaming chunks
        chunk_queue: Queue = Queue()
        
        # Start generation in executor
        loop = asyncio.get_running_loop()
        gen_future = loop.run_in_executor(
            _cuda_executor,
            lambda: _run_streaming_generation(
                text=text,
                output_queue=chunk_queue,
                prefix_speaker_1=payload.get("prefix_speaker_1"),
                prefix_speaker_2=payload.get("prefix_speaker_2"),
                include_prefix=bool(payload.get("include_prefix", False)),
                cfg_scale=float(payload.get("cfg_scale", 6.0)),
                temperature=float(payload.get("temperature", 0.8)),
                top_k=int(payload.get("top_k", 50)),
                chunk_frames=int(payload.get("chunk_frames", 15)),
            )
        )

        # Stream chunks to client as they arrive
        while True:
            # Poll queue with timeout to allow async cooperation
            try:
                chunk = await loop.run_in_executor(
                    None,  # Default executor
                    lambda: chunk_queue.get(timeout=0.1)
                )
            except Empty:
                # Check if generation is done
                if gen_future.done():
                    # Drain remaining items
                    while True:
                        try:
                            chunk = chunk_queue.get_nowait()
                            if chunk is None:
                                break
                            header = struct.pack("!?", chunk.is_last)
                            await ws.send_bytes(header + chunk.pcm16)
                        except Empty:
                            break
                    break
                continue
            
            if chunk is None:
                break
                
            # Send chunk to client
            header = struct.pack("!?", chunk.is_last)
            await ws.send_bytes(header + chunk.pcm16)
            
            if chunk.is_last:
                break

        await ws.send_text(json.dumps({"event": "done"}))

    except WebSocketDisconnect:
        print("[Dia2] WebSocket disconnected")
    except Exception as e:
        print(f"[Dia2] WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
        await ws.close(code=1011)
