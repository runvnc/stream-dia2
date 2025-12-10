import asyncio
import base64
import json
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets

# Increase max message size to 16MB for large audio chunks
WS_MAX_SIZE = 16 * 1024 * 1024

# Default CFG scale (1.0 = no guidance, higher = stronger voice matching)
DEFAULT_CFG_SCALE = 1.0
DEFAULT_TEMPERATURE = 0.8


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """Convert 16-bit little-endian PCM bytes to float32 [-1, 1]."""
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(pcm, dtype="<i2")
    return (arr.astype(np.float32) / 32767.0).clip(-1.0, 1.0)


def load_audio_as_base64(path: str) -> tuple[str, str]:
    """Load audio file and return (base64_data, format_suffix)."""
    p = Path(path)
    suffix = p.suffix.lower()
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8"), suffix


class Dia2Client:
    """Persistent WebSocket client for Dia2 TTS with auto-reconnect."""
    
    def __init__(self, ws_url: str = "ws://localhost:8000/ws/stream_tts"):
        self.ws_url = ws_url
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.sample_rate: int = 24000
        self.stream: Optional[sd.OutputStream] = None
        self.cfg_scale: float = DEFAULT_CFG_SCALE
        self.temperature: float = DEFAULT_TEMPERATURE
        # For auto-reconnect
        self._speaker_1_path: Optional[str] = None
        self._speaker_2_path: Optional[str] = None
        self._connected: bool = False
        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 5
    
    async def connect(self) -> None:
        """Connect to the server."""
        self.ws = await websockets.connect(self.ws_url, max_size=WS_MAX_SIZE)
        
        # Wait for ready message, handling any pings that arrive first
        while True:
            msg = await self.ws.recv()
            data = json.loads(msg)
            event = data.get("event")
            
            if event == "ready":
                self.sample_rate = data.get("sample_rate", 24000)
                print(f"[client] Connected, sample_rate={self.sample_rate}")
                break
            elif event == "ping":
                await self.ws.send(json.dumps({"type": "pong"}))
                continue
            else:
                raise RuntimeError(f"Unexpected response while connecting: {data}")
        self._connected = True
        self._reconnect_attempts = 0
    
    async def set_voice(
        self,
        speaker_1_path: Optional[str] = None,
        speaker_2_path: Optional[str] = None,
    ) -> None:
        """Set voice conditioning audio for speakers."""
        if not self.ws:
            raise RuntimeError("Not connected")
        
        if not speaker_1_path or not speaker_2_path:
            raise ValueError("Both speaker_1_path and speaker_2_path are required for voice cloning")
        
        # Store paths for reconnection
        self._speaker_1_path = speaker_1_path
        self._speaker_2_path = speaker_2_path
        
        payload: dict = {"type": "set_voice"}
        
        b64_1, fmt_1 = load_audio_as_base64(speaker_1_path)
        payload["speaker_1"] = b64_1
        payload["format_1"] = fmt_1
        print(f"[client] Setting speaker 1 voice from {speaker_1_path}")
        
        b64_2, fmt_2 = load_audio_as_base64(speaker_2_path)
        payload["speaker_2"] = b64_2
        payload["format_2"] = fmt_2
        print(f"[client] Setting speaker 2 voice from {speaker_2_path}")
        
        await self.ws.send(json.dumps(payload))
        
        # Wait for confirmation, handling pings
        while True:
            msg = await self.ws.recv()
            data = json.loads(msg)
            event = data.get("event")
            
            if event == "warming":
                print(f"[client] {data.get('message', 'Warming up voice...')}")
                continue
            elif event == "voice_ready":
                print(f"[client] Voice ready!")
                break
            elif event == "voice_set":
                # Legacy event
                print(f"[client] Voice set")
                break
            elif event == "ping":
                await self.ws.send(json.dumps({"type": "pong"}))
                continue
            elif "error" in data:
                print(f"[client] Error setting voice: {data['error']}")
                break
    
    async def ensure_connected(self) -> None:
        """Ensure we're connected, reconnecting if necessary."""
        while self._reconnect_attempts < self._max_reconnect_attempts:
            if self.ws and self._connected:
                try:
                    await self.ws.ping()
                    return
                except:
                    self._connected = False
            
            self._reconnect_attempts += 1
            wait_time = min(2 ** self._reconnect_attempts, 30)  # Exponential backoff, max 30s
            
            if self._reconnect_attempts > 1:
                print(f"[client] Reconnect attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print("[client] Reconnecting...")
            
            try:
                await self.connect()
                if self._speaker_1_path and self._speaker_2_path:
                    print("[client] Restoring voice settings...")
                    await self.set_voice(self._speaker_1_path, self._speaker_2_path)
                return
            except Exception as e:
                print(f"[client] Reconnection failed: {e}")
                if self._reconnect_attempts >= self._max_reconnect_attempts:
                    raise RuntimeError(f"Failed to reconnect after {self._max_reconnect_attempts} attempts")
        
        raise RuntimeError(f"Failed to reconnect after {self._max_reconnect_attempts} attempts")
    
    async def speak(
        self,
        text: str,
        include_prefix: bool = False,
        temperature: Optional[float] = None,
        cfg_scale: Optional[float] = None,
    ) -> None:
        """Generate and play TTS for the given text."""
        await self.ensure_connected()
        
        payload = {
            "type": "tts",
            "text": text,
            "include_prefix": include_prefix,
            "temperature": temperature if temperature is not None else self.temperature,
            "cfg_scale": cfg_scale if cfg_scale is not None else self.cfg_scale,
        }
        
        try:
            await self.ws.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed:
            print("[client] Connection lost, reconnecting...")
            await self.ensure_connected()
            await self.ws.send(json.dumps(payload))
        
        # Start audio stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        self.stream.start()
        
        chunks_received = 0
        
        try:
            while True:
                try:
                    msg = await self.ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    print("[client] Connection lost during streaming")
                    self._connected = False
                    break
                
                if isinstance(msg, str):
                    data = json.loads(msg)
                    event = data.get("event")
                    
                    if event == "done":
                        print(f"[client] Done, received {chunks_received} chunks (server reported {data.get('chunks', '?')})")
                        break
                    elif event == "ping":
                        await self.ws.send(json.dumps({"type": "pong"}))
                        continue
                    elif "error" in data:
                        print(f"[client] Error: {data['error']}")
                        break
                    else:
                        print(f"[client] Ignoring message: {data}")
                        continue
                
                # Binary frame - audio data
                if len(msg) < 1:
                    continue
                
                is_last = struct.unpack("!?", msg[:1])[0]
                pcm16_bytes = msg[1:]
                audio = pcm16_to_float(pcm16_bytes)
                
                if audio.size > 0:
                    self.stream.write(audio)
                    chunks_received += 1
        finally:
            if self.stream:
                await asyncio.sleep(0.5)
                self.stream.stop()
                self.stream.close()
                self.stream = None
    
    async def close(self) -> None:
        """Close the connection."""
        if self.ws:
            self._connected = False
            try:
                await self.ws.send(json.dumps({"type": "close"}))
            except:
                pass
            await self.ws.close()
            self.ws = None
            print("[client] Disconnected")


async def interactive_mode(
    ws_url: str = "ws://localhost:8000/ws/stream_tts",
    speaker_1_path: Optional[str] = None,
    speaker_2_path: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    cfg_scale: float = DEFAULT_CFG_SCALE,
) -> None:
    """Interactive mode with persistent connection."""
    print("\n=== Dia2 Interactive Mode ===")
    print("Connecting...")
    
    client = Dia2Client(ws_url)
    client.cfg_scale = cfg_scale
    client.temperature = temperature
    await client.connect()
    
    if speaker_1_path or speaker_2_path:
        await client.set_voice(speaker_1_path, speaker_2_path)
    
    print("\nCommands:")
    print("  /voices <p1> <p2> - Set both voices at once")
    print("  /cfg <value>      - Set CFG scale (default: 3.0)")
    print("  /temp <value>     - Set temperature (default: 0.8)")
    print("  /s2 <text>        - Speak as Speaker 2")
    print("  /both <s1> | <s2> - Dialogue")
    print("  /quit             - Exit")
    print("  <text>            - Speak as Speaker 1")
    print(f"\nCurrent settings: cfg={client.cfg_scale}, temp={client.temperature}")
    if not (speaker_1_path and speaker_2_path):
        print("\nWARNING: Both --voice and --voice2 are required for voice cloning!")
    print("==============================\n")
    
    try:
        while True:
            try:
                user_input = input("[S1]> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ("/quit", "/q", "quit", "exit"):
                break
            
            if user_input.lower().startswith("/cfg "):
                try:
                    val = float(user_input[5:].strip())
                    client.cfg_scale = val
                    print(f"  CFG scale set to {val}")
                except ValueError:
                    print("  Usage: /cfg <number> (e.g., /cfg 3.0)")
                continue
            
            if user_input.lower().startswith("/temp "):
                try:
                    val = float(user_input[6:].strip())
                    client.temperature = val
                    print(f"  Temperature set to {val}")
                except ValueError:
                    print("  Usage: /temp <number> (e.g., /temp 0.8)")
                continue
            
            if user_input.lower().startswith("/voices "):
                parts = user_input[8:].strip().split()
                if len(parts) >= 2:
                    await client.set_voice(speaker_1_path=parts[0], speaker_2_path=parts[1])
                else:
                    print("  Usage: /voices <speaker1_path> <speaker2_path>")
                continue
            
            if user_input.lower().startswith("/s2 "):
                text = "[S2] " + user_input[4:].strip()
            elif user_input.lower().startswith("/both "):
                parts = user_input[6:].split("|")
                if len(parts) >= 2:
                    text = f"[S1] {parts[0].strip()} [S2] {parts[1].strip()}"
                else:
                    text = "[S1] " + user_input[6:].strip()
            else:
                text = "[S1] " + user_input
            
            print(f"  -> {text}")
            
            try:
                await client.speak(text)
            except Exception as e:
                print(f"  [error] {e}")
                import traceback
                traceback.print_exc()
    
    finally:
        await client.close()


async def single_shot(
    text: str,
    ws_url: str,
    speaker_1_path: Optional[str] = None,
    speaker_2_path: Optional[str] = None,
    include_prefix: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    cfg_scale: float = DEFAULT_CFG_SCALE,
) -> None:
    """Single TTS request."""
    client = Dia2Client(ws_url)
    client.cfg_scale = cfg_scale
    await client.connect()
    
    if speaker_1_path or speaker_2_path:
        await client.set_voice(speaker_1_path, speaker_2_path)
    
    await client.speak(text, include_prefix=include_prefix, temperature=temperature, cfg_scale=cfg_scale)
    await client.close()


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Dia2 streaming TTS client")
    parser.add_argument("text", nargs="?", help="Text to speak (with [S1]/[S2] tags)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--url", default="ws://localhost:8000/ws/stream_tts", help="WebSocket URL")
    parser.add_argument("--voice", "--voice1", dest="voice1", help="Speaker 1 voice audio file")
    parser.add_argument("--voice2", help="Speaker 2 voice audio file")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMPERATURE, help=f"Temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG_SCALE, help=f"CFG scale (default: {DEFAULT_CFG_SCALE})")
    parser.add_argument("--include-prefix", action="store_true", help="Include prefix audio in output")
    
    args = parser.parse_args()
    
    if args.interactive:
        asyncio.run(interactive_mode(
            ws_url=args.url,
            speaker_1_path=args.voice1,
            speaker_2_path=args.voice2,
            temperature=args.temp,
            cfg_scale=args.cfg,
        ))
        return
    
    if not args.text:
        if sys.stdin.isatty():
            parser.print_help()
            sys.exit(1)
        text = sys.stdin.read().strip()
    else:
        text = args.text
    
    asyncio.run(single_shot(
        text=text,
        ws_url=args.url,
        speaker_1_path=args.voice1,
        speaker_2_path=args.voice2,
        include_prefix=args.include_prefix,
        temperature=args.temp,
        cfg_scale=args.cfg,
    ))


if __name__ == "__main__":
    main()
