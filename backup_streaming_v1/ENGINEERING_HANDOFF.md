# Engineering Handoff: Dia2 Streaming TTS Server

**Date:** 2025-12-11
**Project:** `stream-dia2`
**Current Branch:** `main` (Reset to commit `71dd75d`)

## 1. Project Overview
We are building a low-latency streaming TTS server using the Dia2 model. The server pre-calculates a "voice prefix" (warmup) at startup to clone a specific speaker (e.g., "Tom") and then waits for WebSocket requests to generate speech.

## 2. The Problem
When a TTS request comes in, the transition from the pre-calculated prefix state to the new generation is unstable. Symptoms include:
*   **Prefix Hallucination:** The model sometimes repeats the text from the prefix audio (e.g., "Target, this is Tom") before saying the requested text.
*   **"Machine Gun" Artifacts:** A stuttering or repeating noise at the very beginning of the generation.
*   **Speaker Loss:** The model often switches to a random voice instead of maintaining the cloned speaker identity.

## 3. Current Code State
The server now prioritizes **token correctness and boundary stability** over brittle teacher forcing.

*   **File:** `realtime_dia2_server.py`
*   **Key changes (Dec 11):**
    *   **No teacher forcing of raw word-piece tokens.** We do **not** stuff tokenizer IDs directly into `step_tokens` at request start.
    *   **First-frame ACTION kick:** on `t == start_step`, we force the *action* to `new_word` (not the word-piece tokens). This starts the request immediately while staying aligned with training.
    *   **Short cool-down (KV stabilization):** after prefix warmup, run **6 frames** (~80ms @ 75fps) of PAD (text) + `audio_bos` (audio-history) to reduce momentum from prefix.
    *   **Safe Mimi decode sanitization:** replace `clamp(0, 2047)` with mapping invalid/special audio IDs to 0 via `_sanitize_mimi_tokens()`.
      - Rationale: clamping turns 2048/2049 into 2047 (a real token) and can cause a “machine-gun” burst.
    *   **CUDA debug flag:** `--cuda-debug` adds synchronizations and token-range assertions (0..2047) to catch device-side asserts early.

## 4. Failed Approaches (Lessons Learned)
We attempted to implement a **"Cool-down Phase"** during server startup to let the model naturally settle into silence after the prefix. This involved running the model for ~30 frames with `PAD` text input.

*   **Why it failed:** It caused persistent **CUDA `device-side assert triggered` errors**.
*   **Root Cause:** The Depformer model has a smaller vocabulary (2048) than the Transformer (2050). When we fed `PAD` (2049) or `BOS` (2048) tokens into the Depformer (or even the Transformer's embedding layer in some cases), it triggered index-out-of-bounds errors on the GPU.
*   **Status:** We reverted this code to restore stability.

## 5. Next Steps / Recommendations
To fix the artifacts and speaker loss, the boundary strategy is likely still correct, but must be implemented with extreme care regarding token indices.

1.  **Re-implement Cool-down Safely:**
    *   Run the model for 20-30 frames after the prefix.
    *   **CRITICAL:** Clamp ALL audio tokens passed to the Depformer to `[0, 2047]`.
    *   **CRITICAL:** Ensure text tokens passed to the Transformer are within its valid text vocabulary range.
    *   Use `torch.cuda.synchronize()` to debug crashes immediately.

2.  **Alternative: Cross-fade:**
    *   Instead of a hard cut, consider blending the prefix tail with silence in the audio buffer before starting generation.

3.  **Verify State Machine:**
    *   Ensure the `StateMachine` is perfectly synced with the `step_tokens` we are forcing. If the machine expects to output `[S1]` but we force `Hi`, it will desync.

## 6. Key Commands
*   **Run Server:** `python realtime_dia2_server.py --prefix-audio prefix.wav`
    *   Debug: `python realtime_dia2_server.py --prefix-audio prefix.wav --cuda-debug`
*   **Run Client:** `python test_client.py "[S1] Hello world"`
 