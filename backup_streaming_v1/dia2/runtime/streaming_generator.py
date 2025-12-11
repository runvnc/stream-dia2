"""Streaming generator for Dia2 - yields audio chunks during generation."""
from __future__ import annotations
import time

from dataclasses import dataclass
from typing import Iterator, Optional, Any, Tuple, List, Dict, Union

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


@dataclass
class CachedGraphs:
    """Cached CUDA graphs for reuse across TTS calls."""
    transformer_capture: Optional[Tuple[Any, torch.Tensor]] = None
    dep_captures: Optional[List[Dict]] = None
    buffers: Optional[NetworkBuffers] = None
    positions: Optional[torch.Tensor] = None
    mimi_kv: Optional[Any] = None


def run_streaming_generation(
    runtime: RuntimeContext,
    *,
    state: State,
    generation: GenerationState,
    config: GenerationConfig,
    start_step: int = 0,
    chunk_frames: int = 1,
    include_prefix_audio: bool = False,
    logger: RuntimeLogger | None = None,
    cached_graphs: Optional[CachedGraphs] = None,
    prefix_samples_to_skip: int = 0,
) -> Iterator[StreamingChunk]:
    """
    Streaming generation loop that yields audio chunks as they're generated.
    
    Key insight: Due to the delay pattern, when we decode delayed tokens directly,
    the first max_delay frames of output are "shifted" prefix audio. After that,
    we get correct new audio. So we decode immediately and skip the prefix portion.
    
    Args:
        runtime: The Dia2 runtime context
        state: The state machine state
        generation: The generation state with audio buffer
        config: Generation configuration
        start_step: Frame to start generation from (after prefix warmup)
        chunk_frames: How many frames to accumulate before yielding a chunk
        include_prefix_audio: If True, include prefix audio in output
        logger: Optional logger for progress
        cached_graphs: Optional pre-captured CUDA graphs for faster generation
        prefix_samples_to_skip: Number of audio samples to skip (prefix duration)
        initial_mimi_kv: Optional pre-warmed Mimi KV cache (not used in new approach)
        
    Yields:
        StreamingChunk objects containing waveform data
    """
    step_tokens = generation.step_tokens
    audio_buf = generation.audio_buf
    branches = step_tokens.shape[0]
    max_context = runtime.config.runtime.max_context_steps
    
    if max_context <= 0:
        raise ValueError("Runtime configuration must specify a positive max_context_steps")
    
    # Use cached positions if available to ensure graph replay uses correct memory
    if cached_graphs is not None and cached_graphs.positions is not None:
        positions = cached_graphs.positions
    else:
        positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
        if cached_graphs is not None:
            cached_graphs.positions = positions

    main_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    aux_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    
    cfg_active = config.cfg_scale != 1.0
    token_ids = runtime.constants
    delay_tensor = runtime.audio_delay_tensor
    delays = runtime.audio_delays
    max_delay = int(delay_tensor.max().item()) if delay_tensor.numel() else 0
    flush_tail = max_delay + getattr(runtime.machine, "max_padding", 0)
    
    first_word_frame: Optional[int] = None
    eos_cutoff: Optional[int] = None
    
    use_graph = config.use_cuda_graph and runtime.device.type == "cuda"
    
    sample_token_fn = sample_token
    sample_audio_logits_fn = sample_audio_logits
    
    transformer_step = runtime.transformer_step
    depformer_step = runtime.depformer_step
    positions_view = positions.expand(branches, -1)
    
    # Use cached graphs if available, otherwise create new
    if cached_graphs is not None and cached_graphs.buffers is not None:
        buffers = cached_graphs.buffers
        transformer_capture = cached_graphs.transformer_capture
        dep_captures = cached_graphs.dep_captures
    else:
        buffers = _allocate_network_buffers(runtime, branches)
        transformer_capture = None
        dep_captures = None
    
    if use_graph:
        _ensure_graph_cublas_ready(runtime.device)
    
    sample_rate = runtime.mimi.sample_rate
    samples_per_frame = runtime.mimi.samples_per_frame
    
    # We need to buffer max_delay frames before we can produce aligned output.
    # The delay pattern means codebook N is offset by delays[N] frames.
    # undelay_frames() shifts them back into alignment, but requires max_delay
    # extra frames of context.
    frames_before_decode = max_delay + 1  # Need this many frames before first decode
    
    mimi_kv = None
    samples_to_skip = 0  # No longer needed - undelay_frames handles alignment
    
    print(f"[streaming] max_delay={max_delay}, frames_before_decode={frames_before_decode}")

    
    first_frame_time = None
    first_audio_time = None
    
    t_loop_start = time.perf_counter()
    graphs_were_cached = cached_graphs is not None and cached_graphs.transformer_capture is not None
    
    frames_generated = 0
    chunks_sent = 0
    total_samples_output = 0
    samples_skipped = 0
    
    # Track how many frames we've generated (for buffering before first decode)
    frames_generated = 0
    
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
                
                # Update cache after first capture
                if cached_graphs is not None and cached_graphs.transformer_capture is None:
                    cached_graphs.transformer_capture = transformer_capture
                    cached_graphs.dep_captures = dep_captures
                    cached_graphs.buffers = buffers
            
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
            
            frames_generated = offset + 1
            
            if first_frame_time is None:
                first_frame_time = time.perf_counter()
                print(f"[streaming] First frame complete: {(first_frame_time - t_loop_start)*1000:.0f}ms (graphs_cached={graphs_were_cached})")
            
            if eos_cutoff is None and state.end_step is not None:
                eos_cutoff = state.end_step + flush_tail
            
            is_final = (eos_cutoff is not None and t + 1 >= eos_cutoff) or (t + 2 >= audio_buf.shape[-1])
            
            # We need max_delay+1 frames before we can produce any aligned output
            # After that, decode every chunk_frames
            can_decode = frames_generated >= frames_before_decode
            should_decode = can_decode and (frames_generated % chunk_frames == 0 or is_final)
            
            if should_decode:
                # Get all generated frames and undelay them to align codebooks
                end_pos = start_step + frames_generated + 1
                delayed_tokens = audio_buf[0, :, start_step:end_pos].clone()
                
                # Undelay to align codebooks - this produces (frames_generated - max_delay) aligned frames
                aligned_tokens = undelay_frames(
                    delayed_tokens,
                    runtime.audio_delays,
                    token_ids.audio_pad
                ).unsqueeze(0)  # Add batch dim
                
                # Sanitize tokens for Mimi
                aligned_tokens[aligned_tokens >= 2048] = 0
                aligned_tokens[aligned_tokens < 0] = 0
                
                # Decode with Mimi (batch decode for now - simpler and correct)
                with torch.inference_mode():
                    pcm = runtime.mimi.decode(aligned_tokens)
                
                full_waveform = torch.clamp(pcm[0, 0], -1.0, 1.0)
                
                # Force GPU to complete this frame before yielding for true streaming
                torch.cuda.synchronize()
                
                # Only output new samples (not ones we've already sent)
                if full_waveform.shape[0] > total_samples_output:
                    waveform = full_waveform[total_samples_output:]
                else:
                    waveform = torch.tensor([], device=runtime.device)
                
                if waveform.shape[0] > 0:
                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()
                        print(f"[streaming] First audio: {(first_audio_time - t_loop_start)*1000:.0f}ms")
                    
                    chunks_sent += 1
                    total_samples_output += waveform.shape[0]
                    
                    yield StreamingChunk(
                        waveform=waveform,
                        sample_rate=sample_rate,
                        frame_start=start_step,
                        frame_end=start_step + frames_generated,
                        is_final=is_final,
                    )
                
                if is_final:
                    break
    
    duration = total_samples_output / sample_rate if total_samples_output > 0 else 0
    print(f"[streaming] Done: {chunks_sent} chunks, {duration:.2f}s audio, skipped {samples_skipped} samples")


__all__ = [
    "StreamingChunk",
    "CachedGraphs",
    "run_streaming_generation",
]
