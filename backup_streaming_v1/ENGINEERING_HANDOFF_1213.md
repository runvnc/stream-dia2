# Engineering Handoff: Dia2 Streaming TTS (2025-12-13)

## Project Goal

Build a low-latency streaming TTS server using Dia2 for phone call applications.
- **Target:** <500ms to first audio
- **Requirements:** Consistent voice cloning, correct text generation
- **Use case:** Single speaker (S1) voice cloning for phone assistant

## Repository Structure

```
/files/stream-dia2/backup_streaming_v1/
├── realtime_dia2_server.py        # Complex server with KV snapshot/restore (NOT currently used)
├── realtime_dia2_server_simple.py # Simple server we've been debugging
├── test_client.py                 # Test client for complex server (endpoint: /)
├── test_client_simple.py          # Test client for simple server (endpoint: /ws/stream_tts)
├── dia2/                          # Vendored/modified dia2 runtime
│   └── runtime/
│       ├── streaming_generator.py # Key file - streaming generation loop
│       ├── generator.py           # Base generation, warmup_with_prefix()
│       ├── voice_clone.py         # Prefix plan building, S1/S2 handling
│       └── ...
└── prefix.wav, s2.mp3             # Test prefix audio files
```

## Key Concepts

### Mimi Codec
- Neural audio codec with 32 codebooks
- Delay pattern: [16, 18, 18, ...] - max_delay = 18 frames
- Frame rate: 12.5 fps, Sample rate: 24000 Hz
- samples_per_frame = 1920

### Voice Conditioning
- Prefix audio is transcribed by Whisper → creates entries with [S1]/[S2] tags
- Audio is encoded to tokens and stored in `audio_buf`
- `warmup_with_prefix()` runs prefix through transformer, filling KV cache
- Mimi's `decode_streaming()` can be warmed up with prefix audio tokens

### Two Decode Approaches

1. **Batch decode with undelay_frames** (slow, correct)
   - Wait for max_delay+1 frames before decoding
   - Use `undelay_frames()` to align codebooks
   - Call `runtime.mimi.decode(aligned_tokens)`
   - Latency: ~1.4-2.4 seconds to first audio

2. **Streaming decode** (fast, current approach)
   - Decode frame-by-frame with `runtime.mimi.decode_streaming(tokens, mimi_kv)`
   - Mimi maintains state in `mimi_kv`
   - Latency: ~500ms to first audio (server-side)

## What We Tried & Learned

### Problem 1: High Latency (2.4s first audio)
- **Cause:** `frames_before_decode = max_delay + 1` gate in streaming_generator.py
- **Fix:** Reverted to older code using `decode_streaming()` without the gate
- **Result:** Server shows ~500ms first audio ✓

### Problem 2: CUDA Graph Capture Overhead (2.4s first frame)
- **Cause:** Depformer CUDA graphs took 2.4s to capture
- **Fix:** Use eager execution for depformer instead of graphs
- **Result:** First frame now ~250ms ✓

### Problem 3: Client Latency Still High (~4s)
- **Cause:** `warmup_with_prefix()` runs on every request (~3.5s)
- **Note:** Complex server has snapshot/restore to avoid this, simple server doesn't
- **Status:** Not yet fixed in simple server

### Problem 4: Wrong Text Generated (prefix transcript)
- **Symptom:** Model generates prefix transcript verbatim, then correct text
- **Cause:** Transformer KV cache has prefix TEXT context from warmup
- **Tried:** Separate state machines for warmup vs generation - didn't fix it
- **Learning:** The transformer "continues" from prefix context regardless of state machine

### Problem 5: Wrong Voice (random or wrong speaker)
- **Symptom:** Voice on first part differs from second part
- **Cause:** Multiple factors:
  - With two speakers: Mimi warmup sees both, longer speaker dominates
  - Transformer generates audio tokens that "continue" prefix context
  - When context shifts to new text, voice changes

### Current Experiment: Mimi-Only Warmup (Approach #2)
- **Commit:** `6e53caf`
- **Idea:** Skip `warmup_with_prefix()` entirely to avoid transformer text pollution
- **Implementation:**
  - `build_initial_state(prefix=prefix_plan)` populates audio_buf with prefix tokens
  - DON'T call `warmup_with_prefix()` - transformer KV cache stays empty
  - Set `start_step = prefix_plan.aligned_frames`
  - Mimi warmup in streaming_generator.py still runs (decodes prefix tokens)
- **Result:** "Two layers of sound, one faint, no clear speech" - needs debugging

## Root Cause Understanding

### Why Prefix Transcript Gets Regenerated
1. Transformer KV cache filled during `warmup_with_prefix()` with prefix TEXT tokens
2. When generation starts, transformer "continues" from this context
3. Audio tokens generated are similar to prefix audio tokens
4. Mimi decodes them to speech that sounds like prefix transcript
5. Eventually context shifts to new text → audio diverges → random voice

### The Fundamental Challenge
Dia2 is designed for **conversation continuation**, not pure voice cloning:
- Prefix conditions both VOICE (audio tokens) and CONTEXT (text tokens)
- We want voice conditioning WITHOUT text continuation
- These are entangled in the transformer's KV cache

## Recommended Next Steps

### Option A: Fix Mimi-Only Warmup
The current approach (skip transformer warmup) produced garbled audio. Debug why:
1. Check if audio_buf has correct prefix tokens
2. Check if Mimi warmup is actually running in streaming_generator.py
3. Maybe transformer needs SOME warmup but with neutral/silence text tokens

### Option B: Use Complex Server with Fixes
The complex server (`realtime_dia2_server.py`) has:
- KV snapshot/restore (eliminates per-request warmup latency)
- CUDA graph caching

It showed ~925ms client latency but same voice/text issues. Could:
1. Apply Mimi-only warmup approach to complex server
2. Or apply snapshot/restore to simple server

### Option C: Audio-Only Warmup
Modify `warmup_with_prefix()` to:
1. Run transformer warmup with ONLY audio tokens (PAD text tokens)
2. This gives audio context without text continuation bias
3. More complex but might preserve best of both worlds

### Option D: Truncate/Skip Continuation Zone
Accept that first N frames will be prefix-like, skip them:
- Increase `samples_to_skip` to cover the continuation zone
- Trade-off: Adds latency equal to skip duration

## Key Files to Read

1. **streaming_generator.py** (lines 140-180) - Mimi warmup logic
2. **realtime_dia2_server_simple.py** (lines 200-250) - Current server approach
3. **generator.py** `warmup_with_prefix()` - How transformer warmup works
4. **voice_clone.py** `build_prefix_plan()` - How S1/S2 prefixes are built

## Git Tags of Interest

- `1_second_clean` - No-prefix version with ~1s latency
- `100ms_rerender_crash` - Aggressive early decode (crashed)
- `v0.2-low-latency` - Earlier simpler implementation

## Test Commands

```bash
# Run simple server with single speaker prefix
python realtime_dia2_server_simple.py --prefix-audio prefix.wav --seed 42

# Test from client
python test_client_simple.py "[S1] Hi, my name is Bob."
```

## Contact

User (runvnc) has extensive context on this project and can clarify requirements.
