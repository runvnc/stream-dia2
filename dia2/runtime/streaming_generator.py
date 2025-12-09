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
    
    Uses incremental Mimi decoding with KV cache for efficient streaming.
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
    
    # Streaming decode state
    sample_rate = runtime.mimi.sample_rate
    decoder_cache: Optional[Any] = None
    last_decoded_frame = start_step
    total_samples_yielded = 0
    
    print(f"[streaming] Starting generation loop")
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
            frames_available = t + 2 - start_step
            frames_since_decode = t + 2 - last_decoded_frame
            is_final = (eos_cutoff is not None and t + 1 >= eos_cutoff) or (t + 2 >= audio_buf.shape[-1])
            
            # Need enough frames to account for delay before first decode
            min_frames_needed = max_delay + 2
            
            should_decode = (
                (frames_since_decode >= chunk_frames and frames_available >= min_frames_needed)
                or (is_final and frames_available >= min_frames_needed)
            )
            
            if should_decode:
                # Get frames to decode
                decode_end = t + 2
                
                # Get the audio tokens and undelay them
                chunk_tokens = audio_buf[0, :, last_decoded_frame:decode_end].clone()
                
                # For incremental decode, we need to handle the delay alignment
                # The undelay operation shifts channels, so we need context
                # For simplicity, decode the full range from start for now,
                # but use the KV cache for efficiency
                
                if decoder_cache is None:
                    # First decode: process all frames from start
                    all_tokens = audio_buf[0:1, :, :decode_end].clone()
                    aligned = undelay_frames(
                        all_tokens[0],
                        runtime.audio_delays,
                        token_ids.audio_pad
                    ).unsqueeze(0)
                    
                    if aligned.shape[-1] > 0:
                        audio, decoder_cache = runtime.mimi.decode_streaming(aligned, None)
                        waveform = audio[0, 0]
                        
                        if waveform.shape[0] > 0:
                            print(f"[streaming] First chunk: {waveform.shape[0]} samples, is_final={is_final}")
                            total_samples_yielded += waveform.shape[0]
                            
                            yield StreamingChunk(
                                waveform=waveform,
                                sample_rate=sample_rate,
                                frame_start=start_step,
                                frame_end=decode_end,
                                is_final=is_final,
                            )
                else:
                    # Incremental decode: only new frames
                    # Get new tokens since last decode
                    new_tokens = audio_buf[0:1, :, last_decoded_frame:decode_end].clone()
                    
                    # Undelay just the new portion
                    # Note: This is approximate - proper streaming would need overlap handling
                    aligned_new = undelay_frames(
                        new_tokens[0],
                        runtime.audio_delays,
                        token_ids.audio_pad
                    ).unsqueeze(0)
                    
                    if aligned_new.shape[-1] > 0:
                        audio, decoder_cache = runtime.mimi.decode_streaming(aligned_new, decoder_cache)
                        waveform = audio[0, 0]
                        
                        if waveform.shape[0] > 0:
                            print(f"[streaming] Chunk: {waveform.shape[0]} samples, is_final={is_final}")
                            total_samples_yielded += waveform.shape[0]
                            
                            yield StreamingChunk(
                                waveform=waveform,
                                sample_rate=sample_rate,
                                frame_start=last_decoded_frame,
                                frame_end=decode_end,
                                is_final=is_final,
                            )
                
                last_decoded_frame = decode_end
                
                if is_final:
                    break
    
    print(f"[streaming] Loop finished, steps={steps_completed}, samples_yielded={total_samples_yielded}")


__all__ = [
    "StreamingChunk",
    "run_streaming_generation",
]
