# Dia2 CLI flow (two-speaker conditioned) — step-by-step trace table

Command (example):

```bash
uv run -m dia2.cli \
  --hf nari-labs/Dia2-2B \
  --input input.txt \
  --prefix-speaker-1 example_prefix1.wav \
  --prefix-speaker-2 example_prefix2.wav \
  --cuda-graph --verbose \
  output_conditioned.wav
```

This document follows the code path in `/files/dia2/dia2/*`.

## Legend (important state objects)

- **RuntimeContext** (`dia2/runtime/context.py`): `runtime`
  - `runtime.model`: `Dia2Model(TransformerDecoder, Depformer)`
  - `runtime.constants`: `TokenIds` (text pad/new_word/zero + audio pad/bos)
  - `runtime.audio_delays` / `runtime.audio_delay_tensor`
  - `runtime.machine`: `StateMachine`
  - `runtime.frame_rate`: uses `getattr(mimi, "frame_rate", 75.0)`

- **StateMachine State** (`dia2/runtime/state_machine.py`): `state`
  - `entries`: deque of `Entry(tokens=[...], text='word', padding=N)`
  - `pending_tokens`: word-piece tokens currently being emitted
  - `padding_budget`, `forced_padding`, `end_step`, `transcript`

- **GenerationState** (`dia2/runtime/generator.py`): `gen_state`
  - `gen_state.step_tokens`: shape `(branches=2, channels=2+dep_q, 1)`
    - channel 0: main text stream token
    - channel 1: second stream token
    - channels 2..: audio history tokens (per codebook)
  - `gen_state.audio_buf`: shape `(branches=2, dep_q, total_steps)` (delayed codebook grid)
  - `gen_state.decode.transformer`: KV cache (persistent across frames)
  - `gen_state.decode.depformer`: KV cache (reset every frame)

- **CFG branches**:
  - branch 0 conditional
  - branch 1 unconditional (text forced to `zero/pad` each frame)

## Table: CLI end-to-end pipeline

| # | Phase | File → function | Trigger / Inputs | Key state before | Mutations (what changes) | Outputs / Notes |
|---:|---|---|---|---|---|---|
| 1 | CLI parse | `dia2/cli.py → main()` | Parse args (`--hf`, `--input`, `--prefix-speaker-1/2`, `--cuda-graph`, `--verbose`) | none | Creates `args` | Determines device (cuda if available) and dtype (default bf16). |
| 2 | Engine init | `dia2/cli.py → Dia2(repo=...)` | `repo='nari-labs/Dia2-2B'`, `device`, `dtype`, optional tokenizer/mimi overrides | `Dia2._runtime=None` | `Dia2` stores asset refs but does not load weights yet | Lazy init: actual model load happens in `_ensure_runtime()`. |
| 3 | Load script text | `dia2/generation.py → load_script_text()` | `args.input` (e.g. `input.txt`) | none | reads file | Returns full script string. |
| 4 | Validate sampling | `dia2/generation.py → validate_generation_params()` | `temperature`, `topk`, `cfg` | none | none | Ensures positive values. |
| 5 | Build GenerationConfig | `dia2/generation.py → build_generation_config()` | `temperature, top_k, cfg_scale` | none | Creates `GenerationConfig(text=SamplingConfig(...), audio=SamplingConfig(...), cfg_scale=...)` | CLI passes same sampling for text+audio via this helper. |
| 6 | Merge overrides | `dia2/cli.py → overrides{...}` then `Dia2.generate(..., **overrides)` | sets `use_cuda_graph=True` if `--cuda-graph`; sets prefix speaker paths; `include_prefix` optional | none | Prepares override dict | Note: CLI `--verbose` only affects runtime logging. |
| 7 | Ensure runtime | `dia2/engine.py → Dia2._ensure_runtime()` | called at start of `generate()` | `Dia2._runtime=None` | Builds runtime (loads config/weights/tokenizer/mimi) | Heavy step: downloads/loads assets if not cached. |
| 8 | Build runtime | `dia2/runtime/context.py → build_runtime()` | `config_path`, `weights_path`, `tokenizer_id`, `mimi_id`, `device`, `dtype_pref` | none | Loads `DiaConfig`; allocates model modules; loads safetensors into model; loads tokenizer; loads Mimi | Produces `RuntimeContext` + resolved tokenizer/mimi refs. |
| 9 | Token constants | `build_runtime()` | uses `data_cfg.*` + tokenizer vocab | none | `runtime.constants = TokenIds(...)` | Includes `text_pad`, `text_new_word`, `text_zero`, `audio_pad`, `audio_bos`, and speaker tokens `[S1]`, `[S2]`. |
|10| StateMachine init | `build_runtime()` | uses `constants` + `second_stream_ahead` | none | `runtime.machine = StateMachine(...)` | `max_padding=6`, `initial_padding=0` initially. |
|11| Merge generation config | `dia2/engine.py → merge_generation_config()` | base config + overrides | base `GenerationConfig` | Produces `merged` with prefix config populated | PrefixConfig: `speaker_1`, `speaker_2`, `include_audio` (from `--include-prefix`). |
|12| Normalize script | `dia2/generation.py → normalize_script()` | script text | raw script | strips / joins | Ensures consistent string input to parser. |
|13| Build prefix plan | `dia2/runtime/voice_clone.py → build_prefix_plan()` | `merged.prefix` with both speaker wav paths | none | Runs per-speaker pipeline (below) | Produces `PrefixPlan(entries, new_word_steps, aligned_tokens, aligned_frames)`.
|14| Prefix speaker 1: transcribe | `voice_clone.py → transcribe_words()` | `example_prefix1.wav` | none | Whisper model loads + transcribes | Returns list of words with timestamps. |
|15| Prefix speaker 1: words→entries | `voice_clone.py → words_to_entries()` | words + tokenizer + speaker_token(`[S1]`) + `runtime.frame_rate` | none | Creates `entries1` and `steps1` | `steps1` are `new_word_steps` (frame indices) BEFORE later offset. |
|16| Prefix speaker 1: encode audio | `voice_clone.py → encode_audio_tokens()` (via `audio_io.py`) | waveform samples at `runtime.mimi.sample_rate` | none | Mimi encoder produces `tokens1` | Shape typically `(num_codebooks, frames)`, values in `0..2047`. |
|17| Prefix speaker 2: transcribe/entries/audio | same as rows 14–16 | `example_prefix2.wav`, speaker_token(`[S2]`) | none | Builds `entries2`, `steps2`, `tokens2` | `steps2` are later offset by speaker-1 frames when concatenating. |
|18| Build combined PrefixPlan | `voice_clone.py → build_prefix_plan()` | `entries1/2`, `steps1/2`, `tokens1/2` | none | `audio_tokens = cat(tokens1, tokens2, dim=1)`; `entries = entries1+entries2`; `new_word_steps` offset by `offset=3` and `spk1_frames` | Important: `offset=3` is a legacy alignment constant. |
|19| Parse script (main text) | `dia2/runtime/script_parser.py → parse_script()` | `[text]`, tokenizer, constants, `runtime.frame_rate` | none | Produces list of `Entry` for the input text | Adds auto speaker token for first content if user didn’t specify; adds per-word padding. |
|20| Combine entries | `dia2/engine.py → Dia2.generate()` | prefix entries + text entries | none | `entries = prefix_plan.entries + parse_script(...)` | Prefix transcript is literally part of the state machine sequence in CLI mode. |
|21| Set initial padding | `Dia2.generate()` | `merged.initial_padding` (default 2) | `runtime.machine.initial_padding` maybe 0 | sets `runtime.machine.initial_padding = merged.initial_padding` | Affects when `new_word` can start and timestamp alignment. |
|22| Create StateMachine state | `runtime.machine.new_state(entries)` | full `entries` | none | `state.entries = deque(entries)` etc | `pending_tokens` empty initially; `padding_budget/forced_padding` set to initial padding. |
|23| Build GenerationState | `dia2/runtime/generator.py → build_initial_state(prefix=prefix_plan)` | runtime + prefix | none | Allocates `gen_state.step_tokens`, `gen_state.audio_buf`, and KV caches | `audio_buf` pre-filled with **delayed prefix tokens** via `delay_frames(prefix.aligned_tokens, delays, audio_pad)`.
|24| Warmup with prefix (KV fill) | `generator.py → warmup_with_prefix(runtime, plan, state, gen_state)` | `plan.aligned_frames`, `plan.new_word_steps`, `plan.aligned_tokens` | transformer KV empty | For each prefix frame `t`:
- fills audio-history channels of `step_tokens`
- runs transformer `forward_step` (KV grows)
- forces action `new_word` at word boundaries else `pad`
- runs state machine in forced mode to emit prefix transcript tokens into `step_tokens` | After loop:
- transformer KV represents prefix audio + prefix transcript
- `state.transcript` includes prefix words + their steps
- returns `start_step = aligned_frames - 1`. |
|25| Main generation loop | `generator.py → run_generation_loop(..., start_step)` | `merged` sampling + cfg + max_context | warmup-completed KV/audio_buf/step_tokens | For each frame `t`:
- reset depformer KV
- `_fill_audio_channels` from audio_buf (delayed history)
- transformer step (KV grows)
- sample action token (pad/new_word)
- state machine emits main/second text tokens
- sample cb0, then depformer stages, write `audio_buf[:, :, t+1]` | Stops on EOS (`state.end_step`) + flush tail or on max steps.
Returns `(first_word_frame, trimmed_audio_buf)`.
|
|26| Undelay tokens | `dia2/audio/grid.py → undelay_frames()` | `audio_buf[0]` (conditional branch) | delayed grid | Produces `aligned` grid (codebooks aligned) | Shape: `(codebooks, frames_aligned)`.
|27| Crop prefix audio | `Dia2.generate()` | `crop = 0 if include_prefix else first_word_frame` | aligned tokens | `aligned = aligned[:, :, crop:]` if crop valid | This is token-domain cropping of the prefix segment.
|28| Mimi decode | `generator.py → decode_audio()` | `aligned` | none | `runtime.mimi.decode(aligned)` | Produces waveform tensor.
|29| Write WAV | `audio/grid.py → write_wav()` | waveform + sample_rate | none | filesystem write | Saves output wav. |
|30| Build timestamps | `Dia2.generate()` | `state.transcript`, `prefix_entry_count`, `crop`, `frame_rate` | transcript contains prefix+text | removes prefix transcript entries if prefix not included | Produces list of (word, seconds). |

