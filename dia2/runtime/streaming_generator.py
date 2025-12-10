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


def _undelay_single_frame(
    audio_buf: torch.Tensor,
    frame_idx: int,
    delays: List[int],
    pad_id: int,
) -> torch.Tensor:
    """
    Extract a single undelayed frame from the delayed audio buffer.
    
    For undelayed frame M, we need delayed[cb_i, M + delay[cb_i]] for each codebook.
    
    Args:
        audio_buf: Delayed audio buffer [batch, codebooks, frames]
        frame_idx: The undelayed frame index to extract
        delays: List of delays per codebook
        pad_id: Padding token ID
        
    Returns:
        Undelayed frame tensor [batch, codebooks, 1]
    """
    batch, num_codebooks, total_frames = audio_buf.shape
    out = audio_buf.new_full((batch, num_codebooks, 1), pad_id)
    
    for cb_idx, delay in enumerate(delays):
        delayed_pos = frame_idx + delay
        if 0 <= delayed_pos < total_frames:
            out[:, cb_idx, 0] = audio_buf[:, cb_idx, delayed_pos]
    
    return out


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
    initial_mimi_kv: Optional[Any] = None,
) -> Iterator[StreamingChunk]:
    """
    Streaming generation loop that yields audio chunks as they're generated.
    
    Uses incremental undelaying and Mimi streaming decode for efficiency.
    Due to the delay pattern, first audio output is delayed by max_delay frames.
    
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
        initial_mimi_kv: Optional pre-warmed Mimi KV cache
        
    Yields:
        StreamingChunk objects containing waveform data
    """
    step_tokens = generation.step_tokens
    audio_buf = generation.audio_buf
    branches = step_tokens.shape[0]
    max_context = runtime.config.runtime.max_context_steps
    
    if max_context <= 0:
        raise ValueError("Runtime configuration must specify a positive max_context_steps")
    
    positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
    main_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    aux_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    
    cfg_active = config.cfg_scale != 1.0
    token_ids = runtime.constants
    delay_tensor = runtime.audio_delay_tensor
    delays = runtime.audio_delays
    max_delay = int(delay_tensor.max().item()) if delay_tensor.numel() else 0
    flush_tail = max_delay + getattr(runtime.machine, "max_padding", 0)
    print(f"[streaming] max_delay={max_delay}, start_step={start_step}")
    
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
    
    # Streaming state
    sample_rate = runtime.mimi.sample_rate
    samples_per_frame = runtime.mimi.samples_per_frame
    
    # For streaming with delay pattern:
    # - We start generating at delayed frame start_step
    # - To output undelayed frame N, we need delayed frame N + max_delay
    # - So first new audio (undelayed frame start_step) needs delayed frame start_step + max_delay
    # - This means we wait max_delay frames before first output
    
    # Track the next undelayed frame to output
    # First new audio is at undelayed frame start_step (if not including prefix)
    next_undelayed_frame = 0 if include_prefix_audio else start_step
    
    # Mimi streaming decode state
    mimi_kv = initial_mimi_kv
    
    # Warm up Mimi with prefix frames if needed
    if mimi_kv is None and start_step > 0:
        t_warmup_start = time.perf_counter()
        # Decode prefix frames to build Mimi KV cache
        prefix_end = start_step  # Delayed frames 0 to start_step-1
        if prefix_end > max_delay:
            # Undelay all prefix frames at once and decode in one batch
            prefix_tokens_delayed = audio_buf[0:1, :, :prefix_end].clone()
            prefix_tokens_undelayed = undelay_frames(
                prefix_tokens_delayed[0], delays, token_ids.audio_pad
            ).unsqueeze(0)
            
            undelayed_prefix_len = prefix_tokens_undelayed.shape[-1]
            if undelayed_prefix_len > 0:
                # Decode all prefix frames in one call to build KV cache
                _, mimi_kv = runtime.mimi.decode_streaming(
                    prefix_tokens_undelayed, mimi_kv
                )
                t_warmup = time.perf_counter() - t_warmup_start
                print(f"[streaming] Mimi warmup: {t_warmup*1000:.0f}ms ({undelayed_prefix_len} frames, batched)")
    
    first_frame_time = None
    first_audio_time = None
    
    t_loop_start = time.perf_counter()
    graphs_were_cached = cached_graphs is not None and cached_graphs.transformer_capture is not None
    
    frames_generated = 0
    chunks_sent = 0
    total_samples = 0
    
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
            
            frames_generated += 1
            
            if first_frame_time is None:
                first_frame_time = time.perf_counter()
                print(f"[streaming] First frame complete: {(first_frame_time - t_loop_start)*1000:.0f}ms (graphs_cached={graphs_were_cached})")
            
            if eos_cutoff is None and state.end_step is not None:
                eos_cutoff = state.end_step + flush_tail
            
            # Current delayed frame is t+1 (we just generated it)
            # We can output undelayed frame (t+1) - max_delay = t + 1 - max_delay
            current_delayed_frame = t + 1
            max_undelayed_frame = current_delayed_frame - max_delay
            
            is_final = (eos_cutoff is not None and t + 1 >= eos_cutoff) or (t + 2 >= audio_buf.shape[-1])
            
            # Check if we can output new frames
            frames_ready = max_undelayed_frame - next_undelayed_frame
            should_output = frames_ready >= chunk_frames or (is_final and frames_ready > 0)
            
            if should_output and max_undelayed_frame > next_undelayed_frame:
                # Output frames from next_undelayed_frame to max_undelayed_frame
                waveforms = []
                
                for frame_idx in range(next_undelayed_frame, max_undelayed_frame):
                    # Extract single undelayed frame
                    undelayed_frame = _undelay_single_frame(
                        audio_buf[0:1], frame_idx, delays, token_ids.audio_pad
                    )
                    
                    # Decode with Mimi streaming
                    pcm, mimi_kv = runtime.mimi.decode_streaming(undelayed_frame, mimi_kv)
                    waveform = torch.clamp(pcm[0, 0], -1.0, 1.0)
                    
                    # Take only the last frame's samples (Mimi may output more)
                    if waveform.shape[0] > samples_per_frame:
                        waveform = waveform[-samples_per_frame:]
                    
                    waveforms.append(waveform)
                
                if waveforms:
                    combined_waveform = torch.cat(waveforms, dim=0)
                    
                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()
                        print(f"[streaming] First audio: {(first_audio_time - t_loop_start)*1000:.0f}ms")
                    
                    chunks_sent += 1
                    total_samples += combined_waveform.shape[0]
                    
                    yield StreamingChunk(
                        waveform=combined_waveform,
                        sample_rate=sample_rate,
                        frame_start=next_undelayed_frame,
                        frame_end=max_undelayed_frame,
                        is_final=is_final,
                    )
                    
                    next_undelayed_frame = max_undelayed_frame
                
                if is_final:
                    break
    
    duration = total_samples / sample_rate if total_samples > 0 else 0
    print(f"[streaming] Done: {chunks_sent} chunks, {duration:.2f}s audio")


__all__ = [
    "StreamingChunk",
    "CachedGraphs",
    "run_streaming_generation",
]
