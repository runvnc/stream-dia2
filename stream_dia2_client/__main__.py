import asyncio
import json
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
    arr = np.frombuffer(pcm, dtype="<i2")  # little-endian int16
    return (arr.astype(np.float32) / 32767.0).clip(-1.0, 1.0)


async def stream_tts(
    text: str,
    ws_url: str = "ws://localhost:8000/ws/stream_tts",
    prefix_speaker_1: Optional[str] = None,
    prefix_speaker_2: Optional[str] = None,
    include_prefix: bool = False,
) -> None:
    """Connect to the Dia2 streaming server and play audio as it arrives."""
    payload: dict[str, object] = {
        "text": text,
        "include_prefix": include_prefix,
    }
    if prefix_speaker_1:
        payload["prefix_speaker_1"] = prefix_speaker_1
    if prefix_speaker_2:
        payload["prefix_speaker_2"] = prefix_speaker_2

    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps(payload))

        sample_rate: Optional[int] = None
        stream: Optional[sd.OutputStream] = None

        try:
            while True:
                msg = await ws.recv()

                # Text frame
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("event") == "config":
                        sample_rate = int(data["sample_rate"])
                        print(f"[client] Server sample_rate = {sample_rate}")

                        stream = sd.OutputStream(
                            samplerate=sample_rate,
                            channels=1,
                            dtype="float32",
                        )
                        stream.start()

                    elif data.get("event") == "done":
                        print("[client] Stream done.")
                        break
                    elif "error" in data:
                        print("[client] Error from server:", data["error"])
                        break
                    continue

                # Binary frame: [is_last:1 byte][pcm16...]
                if sample_rate is None:
                    print("[client] Received audio before config; ignoring frame.")
                    continue

                if len(msg) < 1:
                    continue

                is_last = struct.unpack("!?", msg[:1])[0]
                pcm16_bytes = msg[1:]
                audio = pcm16_to_float(pcm16_bytes)

                if stream is not None and audio.size > 0:
                    stream.write(audio)

                if is_last:
                    print("[client] Received last chunk.")
                    break
        finally:
            if stream is not None:
                stream.stop()
                stream.close()


async def interactive_mode(
    ws_url: str = "ws://localhost:8000/ws/stream_tts",
    prefix_speaker_1: Optional[str] = None,
    prefix_speaker_2: Optional[str] = None,
    include_prefix: bool = False,
) -> None:
    """Interactive mode: type text and hear it spoken as Speaker 1."""
    print("\n=== Dia2 Interactive Mode ===")
    print("Type text and press Enter to hear it spoken.")
    print("Commands:")
    print("  /quit or /q  - Exit")
    print("  /s2 <text>   - Speak as Speaker 2")
    print("  /both <s1> | <s2> - Speak dialogue (separate with |)")
    print("==============================\n")

    while True:
        try:
            user_input = input("[S1]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/quit", "/q", "quit", "exit"):
            print("Goodbye!")
            break

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

        print(f"  -> Sending: {text}")

        try:
            await stream_tts(
                text=text,
                ws_url=ws_url,
                prefix_speaker_1=prefix_speaker_1,
                prefix_speaker_2=prefix_speaker_2,
                include_prefix=include_prefix,
            )
        except Exception as e:
            print(f"  [error] {e}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dia2 streaming TTS client")
    parser.add_argument("text", nargs="?", help="Script text (with [S1]/[S2])")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--url", default="ws://localhost:8000/ws/stream_tts", help="WebSocket URL of the Dia2 server")
    parser.add_argument("--prefix-speaker-1", help="Prefix WAV for speaker 1")
    parser.add_argument("--prefix-speaker-2", help="Prefix WAV for speaker 2")
    parser.add_argument("--include-prefix", action="store_true", help="Include prefix audio in output")

    args = parser.parse_args()

    # Interactive mode
    if args.interactive:
        asyncio.run(
            interactive_mode(
                ws_url=args.url,
                prefix_speaker_1=args.prefix_speaker_1,
                prefix_speaker_2=args.prefix_speaker_2,
                include_prefix=args.include_prefix,
            )
        )
        return

    # Single-shot mode
    if not args.text:
        print("Provide text as arg, via stdin, or use -i for interactive mode.")
        if sys.stdin.isatty():
            parser.print_help()
            sys.exit(1)
        text = sys.stdin.read().strip()
    else:
        text = args.text

    asyncio.run(
        stream_tts(
            text=text,
            ws_url=args.url,
            prefix_speaker_1=args.prefix_speaker_1,
            prefix_speaker_2=args.prefix_speaker_2,
            include_prefix=args.include_prefix,
        )
    )


if __name__ == "__main__":
    main()
