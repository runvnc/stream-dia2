# Realtime streaming server flow — warmup + request (step-by-step trace table)

This document traces the actual realtime streaming attempt in:
- `/files/stream-dia2/backup_streaming_v1/realtime_dia2_server.py`
- `/files/stream-dia2/backup_streaming_v1/dia2/runtime/*` (generator/state_machine/voice_clone)

It is written as **one continuous list**:
1) everything at server startup (graph capture + prefix warmup + snapshot)
2) then request handling (restore snapshot + generate + decode + stream)

## Legend (state variables tracked)

- **VoiceSession** (server-local):
  - `session.gen_state`: `GenerationState` containing `step_tokens`, `audio_buf`, and caches
  - `session.buffers`: `NetworkBuffers` (text/cb0/dep logits)
  - `session.positions`: `(1,1)` long tensor reused for CUDA graph replay
  - `session.transformer_capture`: `(CUDAGraph, hidden_ref)`
  - `session.dep_captures`: list of CUDA graph captures for depformer stages
  - `session.prefix_plan`: `PrefixPlan` (whisper entries + aligned codec tokens)
  - `session.snapshot`: `StateSnapshot` (cloned transformer KV + audio_buf + step_tokens + start_step)

- **GenerationState** (`dia2/runtime/generator.py`):
  - `gen_state.decode.transformer`: transformer KVCache; each layer has `slot.keys/slot.values/slot.length`
  - `gen_state.decode.depformer`: depformer KVCache; server resets this each frame
  - `gen_state.step_tokens`: `(branches=2, channels=2+dep_q, 1)`
  - `gen_state.audio_buf`: `(branches=2, dep_q, total_steps)` delayed grid

- **StateMachine State** (`dia2/runtime/state_machine.py`):
  - created fresh per request (for request text only)
  - warmup uses a separate temporary state for prefix transcript

## Table: server warmup + request

| # | Phase | File → function | Trigger / Inputs | Key state before | Mutations (what changes) | Outputs / Notes |
|---:|---|---|---|---|---|---|
| 1 | Process start | `realtime_dia2_server.py` module import | Python starts file | none | imports + parses args early | `--prefix-audio`, `--seed`, plus debug flags (`--cuda-debug`, `--trace-boundary`). |
| 2 | Seed (optional) | `realtime_dia2_server.py` | `--seed` | none | sets `torch.manual_seed`, `torch.cuda.manual_seed`, `np.random.seed` | Makes sampling reproducible.
| 3 | Load model | `Dia2.from_repo()` | `MODEL_REPO`, `DEVICE`, `DTYPE` | none | loads Dia2 runtime lazily | Real runtime built later via `dia._ensure_runtime()`.
| 4 | Start session init | `_create_session()` via `_cuda_executor.submit(...).result()` | server startup | `_session=None` | enters session creation on single GPU thread | Ensures all CUDA ops serialized.
| 5 | Ensure runtime | `_create_session(): runtime = dia._ensure_runtime()` | Dia2 runtime | none | loads config/weights/tokenizer/mimi; sets constants/delays/machine | Same runtime concepts as CLI.
| 6 | Vocab sanity | `_create_session()` | `runtime.config.data.audio_vocab_size`, `runtime.constants.audio_pad` | unknown | clamps `runtime.constants.audio_pad` if it is ≥ vocab_size | Defensive fix.
| 7 | Build initial state (no prefix) | `_create_session(): build_initial_state(runtime, prefix=None)` | runtime | none | allocates `gen_state.step_tokens`, `gen_state.audio_buf`, transformer+depformer caches | `audio_buf` initialized to `token_ids.ungenerated` (-2) by default.
| 8 | Allocate network buffers | `_allocate_network_buffers(runtime, branches)` | branches=2 | none | allocates `buffers.text`, `buffers.cb0`, `buffers.dep[]` | Memory addresses stable for CUDA graphs.
| 9 | Allocate positions tensor | `_create_session()` | tensor on device | none | `positions = torch.empty(1,1,long,cuda)` | Reused for graph replay.
|10| Ensure cuBLAS ready | `_ensure_graph_cublas_ready()` | cuda device | none | runs trivial matmul + sync once | Avoids first-graph capture failures.
|11| CUDA graph capture loop (30 frames) | `_create_session()` for t=0..29 | dummy stepping | empty caches initially | Repeated per t:
- `gen_state.reset_dep_cache()`
- `positions.fill_(t)`
- `_fill_audio_channels(step_tokens, audio_buf, delay_tensor, t, audio_bos)`
- force unconditional branch tokens
- `_execute_transformer_graph(..., transformer_capture=None initially)` captures first time then replays
- sample dummy cb0 + dep tokens
- `_execute_depformer_graph(...)` captures per stage then replays
- write dummy tokens into `audio_buf[:, :, t+1]` | Goal: capture stable CUDA graphs with fixed tensor addresses. This is the latency-critical optimization.
|12| Prefix warmup prep | `_create_session()` | `--prefix-audio` exists | after graph capture | resets transformer KV lengths to 0; fills `gen_state.audio_buf` with `audio_bos` | Prepares to re-run warmup on real prefix.
|13| Prefix plan build | `dia2/runtime/voice_clone.py → build_prefix_plan()` | prefix speaker 1 + optional speaker 2 | none | Whisper transcribes prefix wav(s); Mimi encodes to aligned tokens; builds entries + new_word_steps | This is the key place prefix transcript enters the system.
|14| Prefix truncation guard | `_create_session()` | `max_allowed=1000` frames | prefix_plan built | may truncate `aligned_frames` and `aligned_tokens` | Avoids exceeding context.
|15| Warmup StateMachine state (prefix) | `_create_session(): state = runtime.machine.new_state(prefix_plan.entries)` | prefix transcript entries | new state | `state.entries` contains prefix transcript entries | This state is NOT reused for requests.
|16| Pre-fill delayed prefix in audio_buf | `_create_session(): delayed = delay_frames(prefix_plan.aligned_tokens, delays, audio_pad)` then copy to `gen_state.audio_buf` | prefix audio tokens | audio_buf currently audio_bos everywhere | writes delayed prefix tokens into `audio_buf[*, :, :length]` | Ensures transformer audio-history channels read prefix tokens during warmup.
|17| Warmup with prefix (fills transformer KV) | `dia2/runtime/generator.py → warmup_with_prefix(runtime, plan, state, gen_state)` | `plan.aligned_frames`, `plan.new_word_steps`, `plan.entries` | transformer KV lengths are 0 | For each prefix frame `t`:
- sets step_tokens audio channels directly from `plan.aligned_tokens` with delay
- transformer `forward_step` updates KV
- forces action = new_word at prefix word boundaries else pad
- state machine emits prefix transcript tokens into `step_tokens[0,0]`/`[0,1]` | Returns `start_step = aligned_frames - 1`.
Important: after this, transformer KV encodes BOTH prefix audio conditioning and prefix transcript conditioning.
|18| Cool-down (KV stabilization) | `_create_session()` | `cooldown_frames=6` | start_step at end of warmup | For each cooldown frame:
- sets text streams to PAD
- sets audio-history channels to `audio_bos`
- writes `audio_bos` into `audio_buf[:,:,t]` and `audio_buf[:,:,t+1]`
- replays transformer graph to advance KV one step | Goal: reduce “momentum” from prefix speech without adding much latency (~80ms). |
|19| Snapshot creation | `_create_session()` | clone KV + buffers | gen_state ready after cooldown | clones:
- each transformer cache slot’s `keys`, `values`, and `length`
- `audio_buf`
- `step_tokens`
- stores `snapshot.start_step = start_step + 1` | This snapshot is the reusable starting state for every request.
|20| Ready | websocket handler | client connects | session exists | sends `{event:'ready', sample_rate}` | Now waiting for requests.
|21| Receive request | websocket `type='tts'` | request JSON includes `text`, optional `temperature`, `top_k` | none | schedules `_run_tts(...)` on `_cuda_executor` thread | Streaming via queue.
|22| Restore snapshot | `_run_tts(): _reset_session(session)` | session snapshot | gen_state currently contains old tensors from previous run | copies snapshot KV keys/values/length into live cache tensors; copies snapshot `audio_buf` and `step_tokens` | This is where “prefix state reuse” happens.
|23| Parse request script → entries | `_run_tts(): parse_script([normalized], tokenizer, constants, runtime.frame_rate)` | request text | none | builds `new_entries` only for request | You already print these; they should contain ONLY the request words.
|24| Create request StateMachine state | `_run_tts(): state = runtime.machine.new_state(entries)` | request entries | new state | `state.entries` = request words; `pending_tokens` empty | Prefix state machine is not reused.
|25| Determine `start_step` | `_run_tts()` | `start_step = session.snapshot.start_step` | snapshot start step | none | This is the “time index” where generation resumes.
|26| Compute skip amount for streaming | `_run_tts()` | uses `runtime.mimi.samples_per_frame` if present else fallback | none | sets `total_samples_output = start_step * samples_per_frame` | This is *waveform-domain* skipping of already-warmed prefix frames.
|27| Per-frame generation loop (repeats) | `_run_tts(): for t in range(start_step, start_step+1500)` | restored KV/audio_buf/step_tokens + request state machine | each iteration mutates multiple tensors | Repeated steps per frame:
1) `gen.reset_dep_cache()` (depformer KV reset)
2) `positions.fill_(t)`
3) `_fill_audio_channels(step_tokens, audio_buf, delay_tensor, t, audio_bos)` (audio history pulled from delayed grid)
4) unconditional branch enforced (`zero/pad`)
5) transformer graph replay → updates logits buffers + transformer KV
6) sample action token from `buffers.text` (or force new_word on first step)
7) `runtime.machine.process(t, state, action)` → emits `main_token`/`aux_token`, advances request entries
8) sample cb0 token (or use ground truth if within prefix)
9) depformer stage graphs replay, sample remaining codebooks (or use ground truth if within prefix delays)
10) write generated tokens into `audio_buf[:, :, t+1]` | This loop is where the “correct voice but wrong text” can happen if the first post-prefix frames do not strongly steer away from the prefix-conditioned KV.
|28| Decode check | `_run_tts()` | `frames_before_decode = 1 if start_step>0 else max_delay+1` | frames_generated counter | decides when to decode | With prefix present, it tries to decode immediately (lowest latency path).
|29| Prepare tokens for Mimi decode | `_run_tts()` | `delayed_tokens = audio_buf[0,:,:end_pos].clone()` | delayed grid | `aligned = undelay_frames(delayed_tokens, delays, audio_pad)` then `_sanitize_mimi_tokens(aligned)` | Sanitizer maps any <0 or >=2048 to 0 (prevents “special token → 2047” artifact).
|30| Mimi decode | `_run_tts()` | `runtime.mimi.decode(aligned)` | aligned tokens | waveform produced | Full-buffer decode each time (simple & stable; cost grows with end_pos).
|31| Waveform skip + chunk output | `_run_tts()` | `waveform = full_waveform[total_samples_output:]` | total_samples_output | increments `total_samples_output`; puts PCM16 bytes into queue | This is the streaming interface.
|32| Websocket send loop | websocket handler | reads queue items | none | sends binary chunks then `{event:'done'}` | Client receives chunks.

