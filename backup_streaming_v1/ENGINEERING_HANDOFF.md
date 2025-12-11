# Engineering Handoff: Dia2 Streaming TTS Optimization

## Current Status
**Version:** `1_second_clean` (git tag)
**File:** `realtime_dia2_server.py`
**Performance:**
- **First Frame Generation:** ~20-30ms (using cached CUDA graphs)
- **Time to First Audio:** ~950ms
- **Audio Quality:** Clean (no scrambling/artifacts)
- **Architecture:** Persistent Session with reused CUDA graphs, NO voice prefix required.

## The Challenge
We achieved clean audio and <1s latency, but the goal is **<500ms latency** (ideally ~300ms) suitable for conversational AI.

### The Trade-off
1. **Audio Alignment (Latency Source):** Dia2 uses a delay pattern for audio codebooks (max_delay=18 frames). To produce valid, aligned audio, we currently buffer `max_delay + 1` (19) frames before the first decode. 
   - 19 frames * ~50ms/frame = **~950ms latency**.
2. **CUDA Graphs (Speed Source):** We successfully implemented a `PersistentSession` that captures CUDA graphs once at startup and reuses them. This reduced per-frame generation time from ~125ms to ~25ms.

## Current Architecture (`realtime_dia2_server.py`)
- **PersistentSession:** Created at startup. Pre-allocates all tensors and captures CUDA graphs using a dummy warmup.
- **State Reuse:** Between requests, we reset the KV cache *lengths* to 0 and clear the audio buffer, but keep the tensor memory addresses constant. This allows graph reuse.
- **Decoding:** Uses `undelay_frames()` + batch `mimi.decode()` for correctness. Syncs CUDA before yielding.

## Backup Version
- `realtime_dia2_server_simple.py`: A simpler version without persistent sessions. Latency is ~2.5s due to graph capture overhead on every request. Keep this as a fallback.

## Next Steps for <500ms Latency

The next agent should focus on reducing the **buffering delay** without breaking audio quality.

### Strategy 1: Prefix Warmup without Output
*Concept:* Use a dummy prefix (silence or neutral sound) to pre-warm the model state *and* the audio buffer.
*Goal:* Have the buffer already filled with valid "past" tokens so the delay pattern is satisfied immediately for new tokens.
*Challenge:* Resetting the state for the new request without losing the "warmed" context or repeating the prefix audio.

### Strategy 2: Smarter Streaming Decode
*Concept:* Re-investigate `runtime.mimi.decode_streaming()`. 
*Current Issue:* We switched back to batch decode because streaming decode with delayed tokens produced scrambled audio.
*Idea:* Can we feed `decode_streaming` partially aligned frames? Or pre-pad the input with silence tokens so the first real frame is already aligned?

### Strategy 3: Reduce Buffer Size
*Concept:* We currently wait for 19 frames. 
*Idea:* Try reducing `frames_before_decode` to 10 or 5. 
*Risk:* The beginning of the audio will be scrambled/garbage. 
*Mitigation:* Maybe generate 10 frames of silence *first* (fast), then the real audio, and discard the silence? This effectively shifts the scrambling to the silent period.

### Strategy 4: Silence Padding
*Concept:* Pre-fill the `audio_buf` with silence tokens for the first `max_delay` steps.
*Mechanism:* When generation starts at `t=0`, the "past" context (negative steps) is effectively silence. This might allow `undelay_frames` to work immediately without waiting for 19 frames of *generated* content.

## Git Tags
- `1_second_clean`: Current best version (<1s latency, clean audio).

## Update: Low Latency Implementation (Silence Padding)
**Date:** 2025-12-11
**Status:** Implemented
**Changes:**
- Modified `realtime_dia2_server.py` to decode audio immediately (every 3 frames) instead of waiting for the full 19-frame buffer.
- Implemented dynamic silence padding: If the buffer has fewer than `max_delay + 1` frames, it is padded with `audio_pad` tokens.
- **Result:** Latency reduced from ~950ms to ~150ms (plus generation time). The first ~1 second of audio is reconstructed with partial codebooks (lower codebooks are real, higher codebooks are silence), which is a necessary trade-off for immediate start.

## Update: Quality & Consistency Fixes
**Date:** 2025-12-11
**Status:** Implemented
**Changes:**
- **Voice Consistency:** Added logic to reset the random seed (if provided via `--seed`) at the start of *every* request. This ensures the same "random" voice is generated for every connection.
- **Audio Quality:** Increased initial buffer from 3 frames to **15 frames** (~375ms). This eliminates the "unusable" start by providing sufficient context for the vocoder while keeping latency under 500ms.

## Update: Minimized Latency & Chunk Size
**Date:** 2025-12-11
**Status:** Implemented
**Changes:**
- **Decode Frequency:** Changed from decoding every 3 frames to **every frame** once the buffer is full.
- **Impact:** 
  1. Reduces first-chunk latency by eliminating the wait for the next multiple of 3 (saves ~50-100ms).
  2. Produces the smallest possible audio chunks (1 frame / ~20ms), which results in smoother streaming and prevents the "bursty" large chunks you observed.

## Update: Cached Prefix for Zero-Shot Voice Cloning
**Date:** 2025-12-11
**Status:** Implemented
**Changes:**
- **Architecture:** Implemented a `StateSnapshot` mechanism.
- **Startup:** The server now loads a prefix audio file (default: `/files/dia2stream/seed42a1.wav`), runs the heavy Whisper alignment and Dia2 pre-fill **once**, and saves the KV cache.
- **Runtime:** For every request, the session state is restored from this snapshot. This ensures the exact same "Assistant" voice is used every time with **zero latency penalty**.
- **Buffer:** Reverted to full buffering (`max_delay + 1` frames) to guarantee clean audio, as the prefix cache solves the voice consistency issue, and the generation speed is sufficient for <500ms latency with full buffering.

## Update: Final Low-Latency Architecture
**Date:** 2025-12-11
**Status:** Success (<150ms latency)
**Architecture:**
1.  **Cached Prefix (Zero-Shot):**
    -   Server loads `prefix.wav` (default) at startup.
    -   Pre-calculates KV cache and saves a snapshot.
    -   Restores this snapshot for every request, ensuring consistent voice and instant start.
2.  **State Machine Fix:**
    -   Prefix text is **excluded** from the runtime state machine.
    -   This prevents the model from re-generating the prefix text, solving the repetition issue.
3.  **Sliding Window Decoding:**
    -   Decodes only the last ~1 second of context + new frames.
    -   Prevents decoding time from growing linearly with session length.
4.  **Performance:**
    -   First audio: ~100ms.
    -   Generation speed: ~25ms/frame (faster than real-time).

## Update: Reverted to Stable Low-Latency Architecture
**Date:** 2025-12-11
**Status:** Stable (<150ms latency)
**Architecture:**
1.  **Cached Prefix:** Preserved. Provides instant start.
2.  **Prefix Exclusion:** Preserved. Prefix text is removed from state machine to prevent repetition.
3.  **Decoding:** Reverted to **Full Buffer Decoding**. The "Sliding Window" optimization was causing instability (Mimi kernel errors due to invalid window calculation) and has been removed. The generation speed is sufficient that full decoding is acceptable for standard sentence lengths.

## Update: Robust Audio Loading & Crash Fixes
**Date:** 2025-12-11
**Status:** Implemented
**Critical Fixes:**
- **Audio Loading:** Replaced `sphn` with `soundfile` to fix a critical bug where `prefix.wav` was being truncated to ~0.2s (20 frames) instead of 1.5s. This truncation caused the model to "hallucinate" the rest of the prefix text (re-rendering) and then crash due to buffer mismatches.
- **Safety Clamps:** Added `torch.clamp(aligned, 0, 2047)` before decoding to strictly enforce vocabulary limits and prevent CUDA device-side assertions.
- **Logic Cleanup:** Simplified the server to use full-buffer decoding and explicit prefix skipping, removing the complex and error-prone sliding window logic.
