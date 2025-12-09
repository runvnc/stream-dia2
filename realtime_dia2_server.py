import asyncio
import json
import struct
import threading
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from dia2 import Dia2, GenerationConfig, SamplingConfig


# -------------------------------
# Global Dia2 instance & warm-up
# -------------------------------

app = FastAPI()

# Adjust model repo / device / dtype as needed
MODEL_REPO = "nari-labs/Dia2-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"  # or "auto" / "float32"


print(f"[Dia2] Initializing Dia2 from {MODEL_REPO} on {DEVICE} ({DTYPE})...")
dia = Dia2.from_repo(MODEL_REPO, device=DEVICE, dtype=DTYPE)


def _warmup_model() -> None:
    """One-time warm-up to load model, Mimi, tokenizer & CUDA graphs."""
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
    except Exception as e:  # pragma: no cover
        print(f"[Dia2] Warm-up failed: {e}")


_warmup_model()


# -------------------------------
# Utility: waveform -> PCM16 chunks
# -------------------------------

@dataclass
class AudioChunk:
    pcm16: bytes
    sample_rate: int
    is_last: bool


def waveform_to_chunks(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_ms: int = 80,
) -> List[AudioChunk]:
    """Split a 1D float waveform into PCM16 chunks of ~chunk_ms.

    - waveform: 1D torch tensor in [-1, 1]
    - sample_rate: int
    - chunk_ms: size of each chunk in milliseconds
    """
    if waveform.ndim != 1:
        waveform = waveform.view(-1)

    num_samples = waveform.shape[0]
    samples_per_chunk = int(sample_rate * chunk_ms / 1000)
    if samples_per_chunk <= 0:
        samples_per_chunk = max(sample_rate // 50, 1)  # fallback: ~20 ms

    # Clamp and convert to 16-bit PCM
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


# -------------------------------------
# Worker: blocking generate in a thread
# -------------------------------------

class GenerationWorker:
    def __init__(
        self,
        text: str,
        prefix_speaker_1: Optional[str] = None,
        prefix_speaker_2: Optional[str] = None,
        include_prefix: bool = False,
        cfg_scale: float = 6.0,
        temperature: float = 0.8,
        top_k: int = 50,
        use_cuda_graph: bool = True,
    ) -> None:
        self.text = text
        self.prefix_speaker_1 = prefix_speaker_1
        self.prefix_speaker_2 = prefix_speaker_2
        self.include_prefix = include_prefix
        self.cfg_scale = cfg_scale
        self.temperature = temperature
        self.top_k = top_k
        self.use_cuda_graph = use_cuda_graph

        self.queue: asyncio.Queue[Optional[AudioChunk]] = asyncio.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        """Blocking Dia2.generate, then push chunks into async queue."""
        try:
            base_cfg = GenerationConfig(
                cfg_scale=self.cfg_scale,
                text=SamplingConfig(temperature=self.temperature, top_k=self.top_k),
                audio=SamplingConfig(temperature=self.temperature, top_k=self.top_k),
                use_cuda_graph=self.use_cuda_graph,
            )

            result = dia.generate(
                self.text,
                config=base_cfg,
                output_wav=None,
                prefix_speaker_1=self.prefix_speaker_1,
                prefix_speaker_2=self.prefix_speaker_2,
                include_prefix=self.include_prefix,
                verbose=False,
            )

            waveform = result.waveform
            sr = result.sample_rate
            chunks = waveform_to_chunks(waveform, sr, chunk_ms=80)
            loop = asyncio.get_event_loop()
            for ch in chunks:
                loop.call_soon_threadsafe(self.queue.put_nowait, ch)
        except Exception as e:  # pragma: no cover
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(self.queue.put_nowait, None)
            print(f"[Dia2] Generation error: {e}")
        finally:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(self.queue.put_nowait, None)


# -------------------------------
# REST endpoint for one-shot TTS
# -------------------------------

@app.post("/tts_once")
async def tts_once(payload: dict):
    text = payload.get("text")
    if not text:
        return JSONResponse({"error": "Missing 'text'"}, status_code=400)

    prefix1 = payload.get("prefix_speaker_1")
    prefix2 = payload.get("prefix_speaker_2")
    include_prefix = bool(payload.get("include_prefix", False))

    base_cfg = GenerationConfig(
        cfg_scale=payload.get("cfg_scale", 6.0),
        text=SamplingConfig(
            temperature=payload.get("temperature", 0.8),
            top_k=payload.get("top_k", 50),
        ),
        audio=SamplingConfig(
            temperature=payload.get("temperature", 0.8),
            top_k=payload.get("top_k", 50),
        ),
        use_cuda_graph=True,
    )

    result = dia.generate(
        text,
        config=base_cfg,
        output_wav=None,
        prefix_speaker_1=prefix1,
        prefix_speaker_2=prefix2,
        include_prefix=include_prefix,
        verbose=False,
    )

    waveform = result.waveform
    sr = result.sample_rate
    wav_np = waveform.detach().cpu().numpy().astype(np.float32)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    pcm16 = (wav_np * 32767.0).astype(np.int16)

    return {
        "sample_rate": sr,
        "pcm16_hex": pcm16.tobytes().hex(),
        "timestamps": result.timestamps,
    }


# ---------------------------------
# WebSocket endpoint for streaming
# ---------------------------------

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

        worker = GenerationWorker(
            text=text,
            prefix_speaker_1=payload.get("prefix_speaker_1"),
            prefix_speaker_2=payload.get("prefix_speaker_2"),
            include_prefix=bool(payload.get("include_prefix", False)),
            cfg_scale=float(payload.get("cfg_scale", 6.0)),
            temperature=float(payload.get("temperature", 0.8)),
            top_k=int(payload.get("top_k", 50)),
            use_cuda_graph=True,
        )
        worker.start()

        await ws.send_text(json.dumps({"event": "config", "sample_rate": dia.sample_rate}))

        while True:
            chunk = await worker.queue.get()
            if chunk is None:
                break
            header = struct.pack("!?", chunk.is_last)
            await ws.send_bytes(header + chunk.pcm16)

        await ws.send_text(json.dumps({"event": "done"}))

    except WebSocketDisconnect:
        print("[Dia2] WebSocket disconnected")
    except Exception as e:  # pragma: no cover
        print(f"[Dia2] WebSocket error: {e}")
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
        await ws.close(code=1011)
