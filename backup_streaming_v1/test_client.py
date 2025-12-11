#!/usr/bin/env python3
"""Simple test client for the backup Dia2 streaming server."""
import asyncio
import json
import struct
import sys
import time

import numpy as np
import websockets

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("[warn] sounddevice not available, audio playback disabled")

WS_URL = "wss://i9981txmurahrf-3030.proxy.runpod.net/ws/stream_tts"
SAMPLE_RATE = 24000


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(pcm, dtype="<i2")
    return (arr.astype(np.float32) / 32767.0).clip(-1.0, 1.0)


async def tts_request(text: str, ws_url: str = WS_URL):
    """Send a TTS request and play the audio."""
    print(f"[client] Connecting to {ws_url}...")
    
    async with websockets.connect(ws_url, max_size=16*1024*1024) as ws:
        # Wait for ready
        msg = await ws.recv()
        data = json.loads(msg)
        if data.get("event") != "ready":
            print(f"[client] Unexpected: {data}")
            return
        
        sample_rate = data.get("sample_rate", SAMPLE_RATE)
        print(f"[client] Connected, sample_rate={sample_rate}")
        
        # Send TTS request (no voice cloning)
        request_time = time.time()
        await ws.send(json.dumps({
            "type": "tts",
            "text": text,
            "cfg_scale": 1.0,  # No CFG for faster generation
            "temperature": 0.8,
            "top_k": 50,
            "chunk_frames": 10,
        }))
        print(f"[client] Sent: {text[:50]}...")
        
        # Collect audio chunks
        audio_chunks = []
        first_chunk_time = None
        
        while True:
            msg = await ws.recv()
            
            if isinstance(msg, bytes):
                # Binary: audio chunk
                is_last = struct.unpack("!?", msg[:1])[0]
                pcm_data = msg[1:]
                audio_chunks.append(pcm_data)
                
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = (first_chunk_time - request_time) * 1000
                    print(f"[client] *** FIRST CHUNK LATENCY: {latency:.0f}ms ***")
                
                if is_last:
                    break
            else:
                # JSON message
                data = json.loads(msg)
                event = data.get("event")
                
                if event == "done":
                    print(f"[client] Done, {data.get('chunks', 0)} chunks received")
                    break
                elif event == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                elif "error" in data:
                    print(f"[client] Error: {data['error']}")
                    break
                else:
                    print(f"[client] Event: {data}")
        
        # Play audio
        if audio_chunks and HAS_AUDIO:
            all_pcm = b"".join(audio_chunks)
            audio = pcm16_to_float(all_pcm)
            print(f"[client] Playing {len(audio)/sample_rate:.2f}s of audio...")
            sd.play(audio, sample_rate)
            sd.wait()
        elif audio_chunks:
            all_pcm = b"".join(audio_chunks)
            print(f"[client] Received {len(all_pcm)} bytes of audio (playback disabled)")


async def interactive_mode(ws_url: str = WS_URL):
    """Interactive mode - keep connection open for multiple requests."""
    print(f"[client] Connecting to {ws_url}...")
    
    async with websockets.connect(ws_url, max_size=16*1024*1024) as ws:
        # Wait for ready
        msg = await ws.recv()
        data = json.loads(msg)
        if data.get("event") != "ready":
            print(f"[client] Unexpected: {data}")
            return
        
        sample_rate = data.get("sample_rate", SAMPLE_RATE)
        print(f"[client] Connected, sample_rate={sample_rate}")
        print("[client] Enter text to synthesize (Ctrl+C to quit):")
        
        while True:
            try:
                text = input("> ").strip()
                if not text:
                    continue
                if text.lower() in ("quit", "exit", "q"):
                    break
                
                # Send TTS request
                request_time = time.time()
                await ws.send(json.dumps({
                    "type": "tts",
                    "text": text,
                    "cfg_scale": 1.0,
                    "temperature": 0.8,
                    "top_k": 50,
                    "chunk_frames": 10,
                }))
                
                # Collect audio
                audio_chunks = []
                first_chunk_time = None
                
                while True:
                    msg = await ws.recv()
                    
                    if isinstance(msg, bytes):
                        is_last = struct.unpack("!?", msg[:1])[0]
                        pcm_data = msg[1:]
                        audio_chunks.append(pcm_data)
                        
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                            latency = (first_chunk_time - request_time) * 1000
                            print(f"  First chunk: {latency:.0f}ms")
                        
                        if is_last:
                            break
                    else:
                        data = json.loads(msg)
                        if data.get("event") == "done":
                            break
                        elif data.get("event") == "ping":
                            await ws.send(json.dumps({"type": "pong"}))
                
                # Play audio
                if audio_chunks and HAS_AUDIO:
                    all_pcm = b"".join(audio_chunks)
                    audio = pcm16_to_float(all_pcm)
                    print(f"  Playing {len(audio)/sample_rate:.2f}s...")
                    sd.play(audio, sample_rate)
                    sd.wait()
                    
            except KeyboardInterrupt:
                print("\n[client] Bye!")
                break
            except Exception as e:
                print(f"[client] Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dia2 TTS Test Client")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--url", default=WS_URL, help="WebSocket URL")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    if args.interactive:
        asyncio.run(interactive_mode(args.url))
    elif args.text:
        asyncio.run(tts_request(args.text, args.url))
    else:
        # Default test
        asyncio.run(tts_request("[S1] Hello! This is a test of the Dia2 streaming server.", args.url))
