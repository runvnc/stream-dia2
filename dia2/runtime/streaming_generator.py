"""Streaming generator for Dia2 - yields audio chunks during generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from ..audio.grid import delay_frames, mask_audio_logits, undelay_frames
from ..generation import GenerationConfig
from .context import RuntimeContext
from .generator import (
    GenerationState,
    NetworkBuffers,
    _allocate_network_buffers,
    _ensure_graph_cublas_ready,
    _execute_depformer_graph,
    _execute_depformer_stage,
    _execute_transformer_graph,
    _execute_transformer_step,
    _fill_audio_channels,
)
from .guidance import apply_classifier_guidance, sample_audio_logits
from .sampler import sample_token
from .state_machine import State
from .logger import RuntimeLogger


@dataclass
class StreamingChunk:
    """A chunk of generated audio."""
    waveform: torch.Tensor  # 1D float tensor
    sample_rate: int
    frame_start: int  # Starting frame index
    frame_end: int    # Ending frame index  
    is_final: bool    # True if this is the last chunk


def decode_audio_chunk(
    runtime: RuntimeContext,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Decode audio tokens to waveform."""
    if tokens.shape[-1] == 0:
        return torch.zeros(0, device=runtime.device)
    with torch.inference_mode():
        pcm = runtime.mimi.decode(tokens.to(runtime.device))
        return pcm[0, 0]


def run_streaming_generation(
    runtime: RuntimeContext,
    *,
    state: State,
    generation: GenerationState,
    config: GenerationConfig,
    start_step: int = 0,
    chunk_frames: int = 25,  # ~2 seconds at 12.5 Hz
    logger: RuntimeLogger | None = None,
) -> Iterator[StreamingChunk]:
    """
    Streaming generation loop that yields audio chunks as they're generated.
    
    To handle the audio delay alignment, we decode from the start each time
    and only yield the NEW samples that weren't sent before.
    """
    step_tokens = generation.step_tokens
    audio_buf = generation.audio_buf
    branches = step_tokens.shape[0]
    max_context = runtime.config.runtime.max_context_steps
    
    print(f"[streaming] max_context={max_context}, audio_buf.shape={audio_buf.shape}, start_step={start_step}")
    
    if max_context <= 0:
        raise ValueError("Runtime configuration must specify a positive max_context_steps")
    
    positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
    main_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    aux_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    
    cfg_active = config.cfg_scale != 1.0
    token_ids = runtime.constants
    delay_tensor = runtime.audio_delay_tensor
    max_delay = int(delay_tensor.max().item()) if delay_tensor.numel() else 0
    flush_tail = max_delay + getattr(runtime.machine, "max_padding", 0)
    
    print(f"[streaming] max_delay={max_delay}, flush_tail={flush_tail}")
    
    first_word_frame: Optional[int] = None
    eos_cutoff: Optional[int] = None
    last_step = start_step - 1
    
    use_graph = config.use_cuda_graph and runtime.device.type == "cuda"
    
    sample_token_fn = sample_token
    sample_audio_logits_fn = sample_audio_logits
    
    transformer_step = runtime.transformer_step
    depformer_step = runtime.depformer_step
    buffers = _allocate_network_buffers(runtime, branches)
    positions_view = positions.expand(branches, -1)
    
    transformer_capture = None
    dep_captures: list[dict] | None = None
    
    if use_graph:
        _ensure_graph_cublas_ready(runtime.device)
    
    # Track streaming state
    sample_rate = runtime.mimi.sample_rate
    last_yielded_sample = 0  # Track how many PCM samples we've already sent
    last_chunk_frame = start_step
    
    # Mimi's frame rate (samples per frame)
    # At 24kHz sample rate and ~12.5 Hz frame rate, each frame is ~1920 samples
    samples_per_frame = sample_rate // 12  # Approximate
    
    print(f"[streaming] Starting generation loop, samples_per_frame~={samples_per_frame}")
    steps_completed = 0
    
    with torch.inference_mode():
        for offset in range(max_context):
            t = start_step + offset
            
            if eos_cutoff is not None and t >= eos_cutoff:
                print(f"[streaming] Breaking: t={t} >= eos_cutoff={eos_cutoff}")
                break
            if t + 1 >= audio_buf.shape[-1]:
                print(f"[streaming] Breaking: buffer full")
                break
            
            generation.reset_dep_cache()
            positions.fill_(t)
            _fill_audio_channels(step_tokens, audio_buf, delay_tensor, t, token_ids.audio_bos)
            
            if branches > 1:
                step_tokens[1:, 0, 0] = token_ids.zero
                step_tokens[1:, 1, 0] = token_ids.pad
            
            if not use_graph:
                hidden_t = _execute_transformer_step(
                    step_tokens,
                    positions_view,
                    generation,
                    transformer_step,
                    buffers,
                )
            else:
                transformer_capture, dep_captures = _execute_transformer_graph(
                    runtime=runtime,
                    step_tokens=step_tokens,
                    positions_view=positions_view,
                    branches=branches,
                    generation=generation,
                    transformer_step=transformer_step,
                    buffers=buffers,
                    transformer_capture=transformer_capture,
                    dep_captures=dep_captures,
                )
                hidden_t = transformer_capture[1]
            
            guided_text = apply_classifier_guidance(
                buffers.text, cfg_active, config.cfg_scale, config.cfg_filter_k
            )
            if guided_text.shape[0] > 1:
                guided_text = guided_text[:1]
            
            text_token = sample_token_fn(
                guided_text,
                temp=config.text.temperature,
                top_k=config.text.top_k,
            ).item()
            
            main_token, aux_token, _ = runtime.machine.process(t, state, text_token)
            second_token = aux_token if aux_token != -1 else token_ids.pad
            
            if first_word_frame is None and main_token == token_ids.new_word:
                first_word_frame = t - config.initial_padding
            
            step_tokens[:, 0, 0] = main_token
            step_tokens[:, 1, 0] = second_token
            
            guided_cb0 = apply_classifier_guidance(
                buffers.cb0, cfg_active, config.cfg_scale, config.cfg_filter_k
            )
            if guided_cb0.shape[0] > 1:
                guided_cb0 = guided_cb0[:1]
            masked_cb0 = mask_audio_logits(guided_cb0, token_ids.audio_pad, token_ids.audio_bos)
            codebook_token = sample_audio_logits_fn(
                masked_cb0, config.audio.temperature, config.audio.top_k
            )
            audio_buf[:, 0, t + 1] = codebook_token
            
            prev_audio = codebook_token.expand(branches)
            main_tokens.fill_(main_token)
            aux_tokens.fill_(second_token)
            
            for stage in range(runtime.model.depformer.num_depth):
                if use_graph and dep_captures is not None:
                    dep_captures[stage] = _execute_depformer_graph(
                        stage=stage,
                        prev_audio=prev_audio,
                        hidden_t=hidden_t,
                        generation=generation,
                        depformer_step=depformer_step,
                        main_tokens=main_tokens,
                        aux_tokens=aux_tokens,
                        buffers=buffers,
                        capture=dep_captures[stage],
                    )
                else:
                    _execute_depformer_stage(
                        stage_index=stage,
                        prev_audio=prev_audio,
                        hidden_t=hidden_t,
                        generation=generation,
                        depformer_step=depformer_step,
                        main_tokens=main_tokens,
                        second_tokens=aux_tokens,
                        buffers=buffers,
                    )
                
                dep_logits = apply_classifier_guidance(
                    buffers.dep[stage], cfg_active, config.cfg_scale, config.cfg_filter_k
                )
                if dep_logits.shape[0] > 1:
                    dep_logits = dep_logits[:1]
                stage_token = sample_audio_logits_fn(
                    dep_logits,
                    config.audio.temperature,
                    config.audio.top_k,
                )
                audio_buf[:, stage + 1, t + 1] = stage_token
                prev_audio = stage_token.expand(branches)
            
            last_step = t
            steps_completed += 1
            
            if eos_cutoff is None and state.end_step is not None:
                eos_cutoff = state.end_step + flush_tail
                print(f"[streaming] EOS detected at step {state.end_step}, eos_cutoff={eos_cutoff}")
            
            # Check if we should yield a chunk
            frames_generated = t + 2 - start_step  # How many frames we have now
            frames_since_last = t + 1 - last_chunk_frame
            is_final = (eos_cutoff is not None and t + 1 >= eos_cutoff) or (t + 2 >= audio_buf.shape[-1])
            
            # Only yield if we have enough frames (accounting for delay) or it's final
            min_frames_for_decode = max_delay + 5  # Need at least this many to get output
            
            if (frames_since_last >= chunk_frames and frames_generated > min_frames_for_decode) or is_final:
                # Decode ALL frames from start to current position
                current_end = t + 2
                all_tokens = audio_buf[0:1, :, :current_end].clone()
                
                # Undelay the full buffer
                aligned = undelay_frames(
                    all_tokens[0],
                    runtime.audio_delays,
                    token_ids.audio_pad
                ).unsqueeze(0)
                
                if aligned.shape[-1] > 0:
                    # Decode full audio
                    full_waveform = decode_audio_chunk(runtime, aligned)
                    
                    # Only yield the NEW samples
                    if full_waveform.shape[0] > last_yielded_sample:
                        new_audio = full_waveform[last_yielded_sample:]
                        print(f"[streaming] Yielding {new_audio.shape[0]} new samples (total {full_waveform.shape[0]}), is_final={is_final}")
                        
                        yield StreamingChunk(
                            waveform=new_audio,
                            sample_rate=sample_rate,
                            frame_start=last_chunk_frame,
                            frame_end=current_end,
                            is_final=is_final,
                        )
                        
                        last_yielded_sample = full_waveform.shape[0]
                        last_chunk_frame = current_end
                    else:
                        print(f"[streaming] No new samples to yield")
                else:
                    print(f"[streaming] aligned.shape[-1]=0, frames_generated={frames_generated}")
                
                if is_final:
                    break
    
    print(f"[streaming] Loop finished, steps_completed={steps_completed}, total_samples_yielded={last_yielded_sample}")


__all__ = [
    "StreamingChunk",
    "run_streaming_generation",
]
