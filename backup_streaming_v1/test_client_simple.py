#!/usr/bin/env python3
"""Test client for realtime_dia2_server_simple.py - uses /ws/stream_tts endpoint."""
import asyncio
import json
import struct
import sys
import time

import numpy as np
import websockets

# Try multiple audio backends
AUDIO_BACKEND = None

# Try pyaudio first (more reliable on many systems)
try:
    import pyaudio
    AUDIO_BACKEND = "pyaudio"
    print("[audio] Using pyaudio backend")
except ImportError:
    pass

# Try sounddevice as fallback
if AUDIO_BACKEND is None:
    try:
        import sounddevice as sd
        AUDIO_BACKEND = "sounddevice"
        print("[audio] Using sounddevice backend")
    except (ImportError, OSError) as e:
        print(f"[warn] sounddevice not available: {e}")

# Try pygame as another fallback
if AUDIO_BACKEND is None:
    try:
        import pygame
        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)
        AUDIO_BACKEND = "pygame"
        print("[audio] Using pygame backend")
    except (ImportError, pygame.error) as e:
        print(f"[warn] pygame not available: {e}")

if AUDIO_BACKEND is None:
    print("[warn] No audio backend available! Install pyaudio, sounddevice, or pygame")
    print("[warn] To install: pip install pyaudio  OR  pip install sounddevice  OR  pip install pygame")

HAS_AUDIO = AUDIO_BACKEND is not None

# Note: The simple server uses /ws/stream_tts endpoint
WS_URL = "wss://gl54bysgz2dl7s-3030.proxy.runpod.net//ws/stream_tts"
SAMPLE_RATE = 24000


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(pcm, dtype="<i2")
    return (arr.astype(np.float32) / 32767.0).clip(-1.0, 1.0)


class AudioPlayer:
    """Simple audio player that works with multiple backends."""
    
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.backend = AUDIO_BACKEND
        self._stream = None
        self._pa = None
        self._buffer = []
        
    def start_stream(self):
        """Start streaming playback."""
        if self.backend == "pyaudio":
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=1024,
            )
        elif self.backend == "sounddevice":
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=1024,
            )
            self._stream.start()
        elif self.backend == "pygame":
            # Pygame doesn't support true streaming, so we buffer
            self._buffer = []
    
    def write(self, audio_float: np.ndarray):
        """Write audio samples to the stream."""
        if self.backend == "pyaudio":
            self._stream.write(audio_float.astype(np.float32).tobytes())
        elif self.backend == "sounddevice":
            self._stream.write(audio_float)
        elif self.backend == "pygame":
            # Buffer for pygame
            self._buffer.append(audio_float)
    
    def stop_stream(self):
        """Stop streaming playback."""
        if self.backend == "pyaudio":
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
            if self._pa:
                self._pa.terminate()
        elif self.backend == "sounddevice":
            if self._stream:
                self._stream.stop()
                self._stream.close()
        elif self.backend == "pygame":
            # Play buffered audio
            if self._buffer:
                all_audio = np.concatenate(self._buffer)
                # Convert to int16 for pygame
                audio_int16 = (all_audio * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(audio_int16)
                sound.play()
                pygame.time.wait(int(len(all_audio) / self.sample_rate * 1000))
    
    def play_all(self, audio_float: np.ndarray):
        """Play all audio at once (non-streaming)."""
        if self.backend == "pyaudio":
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
            )
            stream.write(audio_float.astype(np.float32).tobytes())
            stream.stop_stream()
            stream.close()
            pa.terminate()
        elif self.backend == "sounddevice":
            sd.play(audio_float, self.sample_rate)
            sd.wait()
        elif self.backend == "pygame":
            audio_int16 = (audio_float * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(audio_int16)
            sound.play()
            pygame.time.wait(int(len(audio_float) / self.sample_rate * 1000))


async def tts_request(text: str, ws_url: str = WS_URL, stream_playback: bool = True):
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
        
        # Setup streaming playback
        player = None
        if stream_playback and HAS_AUDIO:
            player = AudioPlayer(sample_rate)
            player.start_stream()
        
        # Send TTS request (no voice cloning)
        request_time = time.time()
        await ws.send(json.dumps({
            "type": "tts",
            "text": text,
            "cfg_scale": 1.0,  # No CFG for faster generation
            "temperature": 0.8,
            "top_k": 50,
            "chunk_frames": 1,  # Low latency - decode every frame
        }))
        print(f"[client] Sent: {text[:50]}...")
        
        # Collect audio chunks
        audio_chunks = []
        first_chunk_time = None
        chunk_count = 0
        
        while True:
            msg = await ws.recv()
            
            if isinstance(msg, bytes):
                # Binary: audio chunk
                is_last = struct.unpack("!?", msg[:1])[0]
                pcm_data = msg[1:]
                audio_chunks.append(pcm_data)
                chunk_count += 1
                
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = (first_chunk_time - request_time) * 1000
                    print(f"[client] *** FIRST CHUNK LATENCY: {latency:.0f}ms ***")
                
                # Play immediately if streaming
                if player is not None:
                    audio_float = pcm16_to_float(pcm_data)
                    player.write(audio_float)
                    if chunk_count <= 5 or chunk_count % 10 == 0:
                        print(f"[client] Played chunk {chunk_count}: {len(audio_float)} samples")
                
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
        
        # Stop streaming playback
        if player is not None:
            player.stop_stream()
            total_time = time.time() - request_time
            print(f"[client] Streaming playback complete, total time: {total_time*1000:.0f}ms")
        
        # Play all at once if not streaming
        elif audio_chunks and HAS_AUDIO:
            all_pcm = b"".join(audio_chunks)
            audio = pcm16_to_float(all_pcm)
            print(f"[client] Playing {len(audio)/sample_rate:.2f}s of audio...")
            player = AudioPlayer(sample_rate)
            player.play_all(audio)
        
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
                    "chunk_frames": 1,
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
                    player = AudioPlayer(sample_rate)
                    player.play_all(audio)
                    
            except KeyboardInterrupt:
                print("\n[client] Bye!")
                break
            except Exception as e:
                print(f"[client] Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dia2 TTS Test Client (Simple Server)")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--url", default=WS_URL, help="WebSocket URL (default: ws://localhost:3030/ws/stream_tts)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming playback (play all at end)")
    args = parser.parse_args()
    
    if args.interactive:
        asyncio.run(interactive_mode(args.url))
    elif args.text:
        asyncio.run(tts_request(args.text, args.url, stream_playback=not args.no_stream))
    else:
        # Default test
        asyncio.run(tts_request("[S1] Hello! This is a test of the Dia2 streaming server.", args.url, stream_playback=not args.no_stream))
