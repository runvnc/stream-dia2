import asyncio
import json
import base64
import struct
import sys
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """Convert 16-bit little-endian PCM bytes to float32 [-1, 1]."""
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(pcm, dtype='<i2')  # little-endian int16
    return (arr.astype(np.float32) / 32767.0).clip(-1.0, 1.0)


async def stream_tts(
    text: str,
    ws_url: str = "ws://localhost:8000/ws/stream_tts",
    interactive: bool = False,
    prefix_speaker_1: Optional[str] = None,
    prefix_speaker_2: Optional[str] = None,
    include_prefix: bool = False,
):
    """Connect to the Dia2 streaming server, set voice, and play audio."""
    
    async with websockets.connect(ws_url) as ws:
        # 1. Wait for ready
        msg = await ws.recv()
        print(f"[client] Server says: {msg}")
        
        # 2. Set voice if provided
        if prefix_speaker_1 and prefix_speaker_2:
            print("[client] Sending voice configuration...")
            with open(prefix_speaker_1, "rb") as f:
                s1_b64 = base64.b64encode(f.read()).decode("utf-8")
            with open(prefix_speaker_2, "rb") as f:
                s2_b64 = base64.b64encode(f.read()).decode("utf-8")
            
            await ws.send(json.dumps({
                "type": "set_voice",
                "speaker_1": s1_b64,
                "speaker_2": s2_b64,
                "format_1": ".wav",
                "format_2": ".wav"
            }))
            
            # Wait for voice_ready
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                if data.get("event") == "voice_ready":
                    print("[client] Voice ready.")
                    break
                elif data.get("event") == "warming":
                    print(f"[client] {data.get('message')}")
                elif "error" in data:
                    print(f"[client] Error setting voice: {data['error']}")
                    return

        # 3. Start Input Listener if interactive
        if interactive:
            print("[client] Interactive mode. Type text to speak, or !append <file.wav> to add context.")
            
            async def listen_for_input():
                print("> ", end="", flush=True)
                loop = asyncio.get_event_loop()
                while True:
                    try:
                        line = await loop.run_in_executor(None, sys.stdin.readline)
                        if not line:
                            break
                        line = line.strip()
                        if not line: continue
                        
                        # Handle speaker commands
                        if line.lower().startswith("/s1 "):
                            line = "[S1] " + line[4:]
                        elif line.lower().startswith("/s2 "):
                            line = "[S2] " + line[4:]
                        elif not line.startswith("[") and not line.startswith("!"):
                            # Default to S2 if no tag provided (optional, but helpful for user)
                            # Actually, let's just let the user type. But if they want S2, they should use /s2 or [S2]
                            pass
                        
                        if line.startswith("!append "):
                            parts = line.split(" ", 1)
                            if len(parts) > 1:
                                path = parts[1]
                                try:
                                    with open(path, "rb") as f:
                                        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                                    print(f"[client] Appending audio from {path}...")
                                    await ws.send(json.dumps({
                                        "type": "append_audio",
                                        "audio": audio_b64,
                                        "speaker": "speaker_2" # Assume user
                                    }))
                                except Exception as e:
                                    print(f"[client] Error reading file: {e}")
                            continue
                        
                        # Normal TTS
                        print("Sending...")
                        await ws.send(json.dumps({
                            "type": "tts",
                            "text": line,
                            "chunk_frames": 3,
                            "continue_session": True
                        }))
                    except Exception as e:
                        print(f"[client] Input error: {e}")
                        break
            
            asyncio.create_task(listen_for_input())

        # 4. Send initial TTS request if provided
        if text:
            print(f"[client] Sending TTS request: {text[:50]}...")
            await ws.send(json.dumps({
                "type": "tts",
                "text": text,
                "chunk_frames": 1,
                "continue_session": interactive # If interactive, we want to keep context
            }))

        sample_rate = None
        stream = None

        try:
            while True:
                try:
                    msg = await ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    print("[client] Connection closed.")
                    break

                # Text frame
                if isinstance(msg, str):
                    if "ping" in msg: continue
                        
                    data = json.loads(msg)
                    if data.get("event") == "config":
                        # Legacy/unused in this version but kept for compat
                        pass

                    elif data.get("event") == "ready":
                        if "sample_rate" in data:
                            sample_rate = int(data["sample_rate"])
                            if stream is None:
                                stream = sd.OutputStream(
                                    samplerate=sample_rate,
                                    channels=1,
                                    dtype="float32",
                                )
                                stream.start()
                        continue

                    elif data.get("event") == "done":
                        print("[client] Generation done.")
                        if not interactive:
                            break
                        continue
                        
                    elif data.get("event") == "appended":
                        print(f"[client] Audio appended. New step: {data.get('new_step')}")
                        continue
                        
                    elif "error" in data:
                        print("[client] Error from server:", data["error"])
                        if not interactive:
                            break
                    continue

                # Binary frame: [is_last:1 byte][pcm16...]
                if len(msg) < 1:
                    continue

                is_last = struct.unpack("!?", msg[:1])[0]
                pcm16_bytes = msg[1:]
                audio = pcm16_to_float(pcm16_bytes)

                if stream is None:
                    # Fallback if ready event missed or not sent yet (shouldn't happen with correct protocol)
                    # Default to 24k or wait?
                    # Let's assume 24000 if not set, or print warning
                    if sample_rate is None:
                         sample_rate = 24000
                         stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
                         stream.start()

                if stream is not None and audio.size > 0:
                    stream.write(audio)

                if is_last:
                    # In interactive mode, is_last just means this utterance is done.
                    # We don't break the loop.
                    pass
                    
        finally:
            if stream is not None:
                stream.stop()
                stream.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dia2 streaming TTS client")
    parser.add_argument("text", nargs="?", help="Script text (with [S1]/[S2])")
    parser.add_argument("--url", default="ws://localhost:8000/ws/stream_tts", help="WebSocket URL of the Dia2 server")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode (keep connection open)")
    parser.add_argument("--prefix-speaker-1", help="Prefix WAV for speaker 1")
    parser.add_argument("--prefix-speaker-2", help="Prefix WAV for speaker 2")
    parser.add_argument("--include-prefix", action="store_true", help="Include prefix audio in output")

    args = parser.parse_args()

    if not args.text and not args.interactive:
        print("Provide text as arg or via stdin.")
        if sys.stdin.isatty():
            parser.print_help()
            sys.exit(1)
        text = sys.stdin.read().strip() if not sys.stdin.isatty() else "[S1] Hello world."
        # If interactive, we don't want to send the default text immediately if it's just the example
    elif args.interactive and args.text:
        # User provided text arg with interactive mode, send it
        text = args.text
    else:
        text = args.text

    asyncio.run(
        stream_tts(
            text=text,
            ws_url=args.url,
            interactive=args.interactive,
            prefix_speaker_1=args.prefix_speaker_1,
            prefix_speaker_2=args.prefix_speaker_2,
            include_prefix=args.include_prefix,
        )
    )


if __name__ == "__main__":
    main()
