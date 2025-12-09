"""Streaming generator for Dia2 - yields audio chunks during generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Any

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
    
    Due to the delay alignment in audio codebooks, we decode from the start
    each time but only yield the NEW samples not yet sent.
    """
    step_tokens = generation.step_tokens
    audio_buf = generation.audio_buf
    branches = step_tokens.shape[0]
    max_context = runtime.config.runtime.max_context_steps
    
    print(f"[streaming] max_context={max_context}, start_step={start_step}")
    
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
    
    first_word_frame: Optional[int] = None
    eos_cutoff: Optional[int] = None
    
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
    
    # Streaming state
    sample_rate = runtime.mimi.sample_rate
    samples_sent = 0  # Track how many PCM samples we've already sent
    last_decode_frame = start_step
    
    # Need enough frames before first decode (to handle delay alignment)
    min_frames_for_decode = max_delay + 5
    
    print(f"[streaming] max_delay={max_delay}, min_frames_for_decode={min_frames_for_decode}")
    steps_completed = 0
    
    with torch.inference_mode():
        for offset in range(max_context):
            t = start_step + offset
            
            if eos_cutoff is not None and t >= eos_cutoff:
                break
            if t + 1 >= audio_buf.shape[-1]:
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
            
            steps_completed += 1
            
            if eos_cutoff is None and state.end_step is not None:
                eos_cutoff = state.end_step + flush_tail
            
            # Check if we should yield a chunk
            current_frame = t + 2
            frames_available = current_frame - start_step
            frames_since_decode = current_frame - last_decode_frame
            is_final = (eos_cutoff is not None and t + 1 >= eos_cutoff) or (t + 2 >= audio_buf.shape[-1])
            
            should_decode = (
                (frames_since_decode >= chunk_frames and frames_available >= min_frames_for_decode)
                or (is_final and frames_available >= min_frames_for_decode)
            )
            
            if should_decode:
                # Decode all frames from start to current
                all_tokens = audio_buf[0:1, :, :current_frame].clone()
                
                aligned = undelay_frames(
                    all_tokens[0],
                    runtime.audio_delays,
                    token_ids.audio_pad
                ).unsqueeze(0)
                
                if aligned.shape[-1] > 0:
                    # Decode to waveform
                    with torch.inference_mode():
                        pcm = runtime.mimi.decode(aligned)
                        full_waveform = torch.clamp(pcm[0, 0], -1.0, 1.0)
                    
                    # Only yield NEW samples
                    total_samples = full_waveform.shape[0]
                    if total_samples > samples_sent:
                        new_samples = full_waveform[samples_sent:]
                        print(f"[streaming] Yielding {new_samples.shape[0]} new samples (total: {total_samples}), is_final={is_final}")
                        
                        yield StreamingChunk(
                            waveform=new_samples,
                            sample_rate=sample_rate,
                            frame_start=last_decode_frame,
                            frame_end=current_frame,
                            is_final=is_final,
                        )
                        
                        samples_sent = total_samples
                
                last_decode_frame = current_frame
                
                if is_final:
                    break
    
    print(f"[streaming] Done: {steps_completed} steps, {samples_sent} samples")


__all__ = [
    "StreamingChunk",
    "run_streaming_generation",
]
