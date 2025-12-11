![Banner](banner.gif)

<div align="center">
  <a href="https://huggingface.co/nari-labs/Dia2-2B"><img src="https://img.shields.io/badge/HF%20Repo-Dia2--2B-orange?style=for-the-badge"></a>
  <a href="https://discord.gg/bJq6vjRRKv"><img src="https://img.shields.io/badge/Discord-Join%20Chat-7289DA?logo=discord&style=for-the-badge"></a>
  <a href="https://github.com/nari-labs/dia2/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge"></a>
</div>


**Dia2** is a **streaming dialogue TTS model** created by Nari Labs.

The model does not need the entire text to produce the audio, and can start generating as the first few words are given as input. You can condition the output on audio, enabling natural conversations in realtime.

We provide model checkpoints (1B, 2B) and inference code to accelerate research. The model only supports up to 2 minutes of generation in English.

⚠️ Quality and voices vary per generation, as the model is not fine-tuned on a specific voice. Use with prefix or fine-tune in order to obtain stable output.

Try it now on Hugging Face [Spaces](https://huggingface.co/spaces/nari-labs/Dia2-2B)

## Upcoming

- Bonsai (JAX) implementation
- Dia2 TTS Server: Real streaming support
- Sori: Dia2-powered speech-to-speech engine written in Rust

## Quickstart

> **Requirement** — install [uv](https://docs.astral.sh/uv/) and use CUDA 12.8+
> drivers. All commands below run through `uv run …` as a rule.

1. **Install dependencies (one-time):**
   ```bash
   uv sync
   ```
2. **Prepare a script:** edit `input.txt` using `[S1]` / `[S2]` speaker tags.
3. **Generate audio:**
   ```bash
   uv run -m dia2.cli \
     --hf nari-labs/Dia2-2B \
     --input input.txt \
     --cfg 6.0 --temperature 0.8 \
     --cuda-graph --verbose \
     output.wav
   ```
   The first run downloads weights/tokenizer/Mimi. The CLI auto-selects CUDA when available (otherwise CPU) and defaults to bfloat16 precision—override with `--device` / `--dtype` if needed.
4. **Conditional Generation (recommended for stable use):**
   ```bash
   uv run -m dia2.cli \
     --hf nari-labs/Dia2-2B \
     --input input.txt \
     --prefix-speaker-1 example_prefix1.wav \
     --prefix-speaker-2 example_prefix2.wav \
     --cuda-graph --verbose \
     output_conditioned.wav
   ```
   Condition the generation on previous conversational context in order to generate natural output for your speech-to-speech system. For example, place the voice of your assistant as prefix speaker 1, place user's audio input as prefix speaker 2, and generate the response to user's input.

   Whisper is used to transcribe each prefix file, which takes additional time. We include example prefix files as `example_prefix1.wav` and `example_prefix2.wav` (both files are output created by the model).
6. **Gradio for Easy Usage**
   ```bash
   uv run gradio_app.py
   ```

### Programmatic Usage
```python
from dia2 import Dia2, GenerationConfig, SamplingConfig

dia = Dia2.from_repo("nari-labs/Dia2-2B", device="cuda", dtype="bfloat16")
config = GenerationConfig(
    cfg_scale=2.0,
    audio=SamplingConfig(temperature=0.8, top_k=50),
    use_cuda_graph=True,
)
result = dia.generate("[S1] Hello Dia2!", config=config, output_wav="hello.wav", verbose=True)
```
Generation runs until the runtime config's `max_context_steps` (1500, 2 minutes)
or until EOS is detected. `GenerationResult` includes audio tokens, waveform tensor,
and word timestamps relative to Mimi’s ~12.5 Hz frame rate.

## Hugging Face

| Variant | Repo |
| --- | --- |
| Dia2-1B | [`nari-labs/Dia2-1B`](https://huggingface.co/nari-labs/Dia2-1B)
| Dia2-2B | [`nari-labs/Dia2-2B`](https://huggingface.co/nari-labs/Dia2-2B)

## License & Attribution

Licensed under [Apache 2.0](LICENSE). All third-party assets (Kyutai Mimi codec, etc.) retain their original licenses.

## Disclaimer

This project offers a high-fidelity speech generation model intended for research and educational use. The following uses are **strictly forbidden**:

- **Identity Misuse**: Do not produce audio resembling real individuals without permission.
- **Deceptive Content**: Do not use this model to generate misleading content (e.g. fake news)
- **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

By using this model, you agree to uphold relevant legal standards and ethical responsibilities. We **are not responsible** for any misuse and firmly oppose any unethical usage of this technology.

## Acknowledgements
- We thank the [TPU Research Cloud](https://sites.research.google/trc/about/) program for providing compute for training.
- Our work was heavily inspired by [KyutaiTTS](https://kyutai.org/next/tts) and [Sesame](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)

---
Questions? Join our [Discord](https://discord.gg/bJq6vjRRKv) or open an issue.
