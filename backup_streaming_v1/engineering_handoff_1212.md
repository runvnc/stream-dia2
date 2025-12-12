# Engineering Handoff (2025-12-12): Dia2 Streaming TTS — reconciling “README says streaming” with our observed regimes

**Date:** 2025-12-12
**Project:** stream-dia2
**Primary goal:** drop-in ElevenLabs replacement for phone-call TTS with VAD + LLM latency; target **sub-200ms perceived first audio**.

This handoff is written after:
- tracing Dia2 CLI (`/files/dia2`) vs our server implementation (`/files/stream-dia2/backup_streaming_v1/realtime_dia2_server.py`)
- adding boundary tracing (`--trace-boundary`) and running a real trace

---

## 0) What Dia2 README actually claims (and what it doesn’t)
From `/files/dia2/README.md`:
- “Dia2 is a **streaming dialogue TTS model**.”
- “The model does not need the entire text to produce the audio, and can start generating as the first few words are given as input.”
- “We provide inference code…” and “Upcoming: Dia2 TTS Server: Real streaming support.”

Key interpretation:
- The README’s “streaming” claim is about the **model’s incremental generation interface** (frame-by-frame, doesn’t require full text upfront), not a guarantee that our current decode strategy will produce **fully aligned** audio with <200ms latency.
- A “streaming server” can still stream **approximate** audio early (partial alignment / padded codebooks / crossfade), then converge to high quality later.

---

## 1) What we learned definitively from a real trace (current server)
The trace (Dec 12 runpod) shows:
- `runtime.frame_rate = 12.5`
- `mimi.samples_per_frame = 1920`
- `max_delay = 18`
- `prefix_frames = 224`, `start_step = 230`
- First emitted audio ~ **1121ms** after request, despite the model consuming request entries immediately.

### 1.1 Hard constraint for “fully aligned” audio
If we require **aligned frames past the crop point**, the earliest they can exist is roughly:
- `max_delay / frame_rate ≈ 18 / 12.5 ≈ 1.44s`

That matches the observed ~1.1–1.5s class delays. So:
- **Full-alignment correctness at the boundary cannot meet <200ms** under these runtime values.

This is not “Dia2 can’t stream” — it’s “fully aligned undelay correctness has latency.”

### 1.2 Why the “prefix transcript re-read” happens in the fast regime
Prefix warmup uses Whisper to create prefix transcript tokens, then feeds them into `step_tokens[0,0]/[0,1]` during warmup.
That bakes **linguistic content** (not just voice) into transformer KV.
When you decode early/approximately, the model can behave like continuing the prefix dialog and re-read the prefix transcript in the cloned voice.

Important: this is different from a deterministic bug where prefix entries are accidentally included in request entries.

---

## 2) Two distinct output regimes exist (and we must explicitly support both)
Across the repo history we repeatedly bounced between these, causing confusion:

### Regime A: “Early audio / phone-call feel”
- Time-to-first-audio can be ~100–200ms.
- Achieved by decoding before full delay alignment is satisfied.
- Techniques seen in previous commits:
  - decode after `min_frames` (10/15), not after `max_delay+1`.
  - pad missing future frames with `audio_pad` then sanitize (`>=2048 -> 0`)
  - fade-in to hide garbage.
- Failure mode: prefix-conditioned KV can dominate early → “prefix transcript re-read” and/or voice drift.

### Regime B: “Aligned correctness / stable”
- High quality, more stable semantics.
- Time-to-first-audio tends to be ~1.1–1.5s given frame_rate=12.5 and max_delay=18.
- This is what the current trace demonstrates.

**Engineering conclusion:** We should stop trying to make one code path do both. Implement a flag that selects the regime.

---

## 3) Why low-latency results happened “here and there”
They can happen when any of these were true:
1) The code was in Regime A (min_frames decode + padding + fade), regardless of alignment correctness.
2) The code had incorrect skip math (hardcoded `samples_per_frame=320`) causing prefix-neighborhood audio exposure.
3) The runtime differed (different Mimi/frame_rate/delay config), changing the effective floor.
4) CFG/seed conditions changed voice stability.

---

## 4) Current server state (as of Dec 12)
File: `backup_streaming_v1/realtime_dia2_server.py`

Key properties now:
- Corrects prefix skip: uses `runtime.mimi.samples_per_frame`.
- Adds `--trace-boundary` tracing.
- Adds safe Mimi token sanitization (`>=2048 -> 0`, not clamp-to-2047).
- Removes brittle teacher-forcing of raw word-piece tokens.
- Adds short cooldown after warmup.

Known remaining issues:
- **CFG is effectively disabled in server generation** (calls `apply_classifier_guidance(..., False, 1.0, ...)`). This can cause random / poor voice.
- Low-latency goal is not met under aligned regime.

---

## 5) Proposed plan: implement “two-mode streaming” explicitly

### 5.1 Add a server flag: `--decode-mode {aligned,early}`

**aligned** (current default):
- wait for aligned frames beyond crop; expect ~1.4s first audio (given current runtime)
- best quality, least artifacts

**early** (new):
- decode immediately (or after 1–3 frames) by padding missing future frames for undelay, sanitize to 0
- optional fade-in + optional crossfade at boundary
- target: <200ms perceived first audio
- accept that first ~0.5–1.5s may be lower fidelity

### 5.2 Fix voice stability regardless of mode
- Support CFG in the server:
  - take `cfg_scale`, `cfg_filter_k` from request JSON
  - pass `cfg_active=True` to `apply_classifier_guidance` when `cfg_scale != 1.0`
- Support seed per request (or fixed per server) if desired.

### 5.3 Reduce “prefix transcript re-read” risk
Two experiments:
1) **Audio-only prefix warmup** (skip Whisper transcript tokens):
   - keep Mimi encoding of prefix audio
   - do not feed prefix transcript tokens into the text streams during warmup
   - goal: voice conditioning without linguistic continuation prior

2) If Whisper transcript is needed for diarization/dialogue, then add explicit “end-of-prefix” stabilization:
   - longer cooldown that is safe token-wise
   - plus waveform crossfade

---

## 6) What to capture in future traces (so we don’t argue with anecdotes)
For each run, record:
- runtime: `frame_rate`, `mimi.samples_per_frame`, `max_delay`
- decode mode (aligned vs early)
- first audio latency
- whether prefix transcript re-read occurs
- whether request entries are consumed immediately

The existing `--trace-boundary` logs cover most of this.

---

## 7) Next actionable tasks
1) Implement CFG+cfg_scale wiring in server generation loop.
2) Implement `--decode-mode early`:
   - pad missing frames to `max_delay+1`
   - sanitize tokens to 0
   - decode and stream immediately
   - add 200ms fade-in
3) Add `--prefix-no-transcript` option to warmup.

---

## 8) Mental model reminder (why this is hard)
- Dia2 is “streaming” at the model level: it generates frame-by-frame and doesn’t need full text.
- Mimi uses multiple codebooks with delays; “aligned audio after a boundary” requires future frames.
- Therefore: sub-200ms requires approximations at the boundary.

