"""Dia2 TTS Server - Continuous Session Mode"""
import asyncio
import base64
import json
import copy
import struct
import tempfile
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from dia2 import Dia2, GenerationConfig, SamplingConfig
from dia2.runtime import voice_clone
from dia2.runtime.voice_clone import WhisperWord, build_prefix_plan, PrefixPlan
from dia2.runtime.audio_io import load_mono_audio, encode_audio_tokens
from dia2.generation import PrefixConfig, normalize_script
from dia2.runtime.generator import build_initial_state, warmup_with_prefix, GenerationState, _execute_transformer_step, _execute_depformer_stage, _execute_transformer_graph, _execute_depformer_graph, _allocate_network_buffers, _ensure_graph_cublas_ready, _fill_audio_channels
from dia2.runtime.streaming_generator import CachedGraphs
from dia2.runtime.state_machine import State
from dia2.runtime.script_parser import parse_script
from dia2.runtime.guidance import apply_classifier_guidance, sample_audio_logits
from dia2.runtime.sampler import sample_token
from dia2.audio.grid import mask_audio_logits

app = FastAPI()

MODEL_REPO = "nari-labs/Dia2-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"

# ============================================================================
# Caching & Helpers
# ============================================================================

_whisper_model = None
_transcription_cache = {}
_audio_data_cache = {}
_audio_token_cache = {}
_current_encoding_path = None

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper_timestamped as wts
        print("[Dia2] Loading Whisper model...")
        _whisper_model = wts.load_model("openai/whisper-large-v3", device=DEVICE)
    return _whisper_model

def _hash_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _cached_transcribe_words(audio_path, device, language=None):
    file_hash = _hash_file(audio_path)
    if file_hash in _transcription_cache:
        return _transcription_cache[file_hash]
    
    import whisper_timestamped as wts
    model = _get_whisper_model()
    result = wts.transcribe(model, audio_path, language=language)
    words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            text = (word.get("text") or word.get("word") or "").strip()
            if not text: continue
            words.append(WhisperWord(text=text, start=float(word.get("start", 0.0)), end=float(word.get("end", 0.0))))
    _transcription_cache[file_hash] = words
    return words

voice_clone.transcribe_words = _cached_transcribe_words

def _cached_load_audio(audio_path, sample_rate):
    global _current_encoding_path
    _current_encoding_path = audio_path
    file_hash = _hash_file(audio_path)
    if file_hash in _audio_data_cache: return _audio_data_cache[file_hash]
    audio = load_mono_audio(audio_path, sample_rate)
    _audio_data_cache[file_hash] = audio
    return audio

def _cached_encode_audio(mimi, audio):
    if _current_encoding_path:
        file_hash = _hash_file(_current_encoding_path)
        if file_hash in _audio_token_cache: return _audio_token_cache[file_hash]
    tokens = encode_audio_tokens(mimi, audio)
    if _current_encoding_path:
        _audio_token_cache[_hash_file(_current_encoding_path)] = tokens
    return tokens

_original_build_prefix_plan = build_prefix_plan
def _cached_build_prefix_plan(runtime, prefix, **kwargs):
    if prefix is None: return None
    return _original_build_prefix_plan(runtime, prefix, transcribe_fn=_cached_transcribe_words, load_audio_fn=_cached_load_audio, encode_fn=lambda a: _cached_encode_audio(runtime.mimi, a), **kwargs)

# ============================================================================
# Model Init
# ============================================================================

print(f"[Dia2] Initializing Dia2 from {MODEL_REPO}...")
dia = Dia2.from_repo(MODEL_REPO, device=DEVICE, dtype=DTYPE)
_get_whisper_model()

# ============================================================================
# Continuous Session
# ============================================================================

class ContinuousSession:
    def __init__(self, runtime, speaker_1_path, speaker_2_path):
        self.runtime = runtime
        self.input_queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        
        # Setup state
        self._setup(speaker_1_path, speaker_2_path)
        
    def _setup(self, s1_path, s2_path):
        print("[Dia2] Setting up continuous session...")
        prefix_config = PrefixConfig(speaker_1=s1_path, speaker_2=s2_path)
        self.prefix_plan = _cached_build_prefix_plan(self.runtime, prefix_config)
        
        self.gen_state = build_initial_state(self.runtime, prefix=self.prefix_plan)
        self.runtime.machine.initial_padding = 0
        self.state = self.runtime.machine.new_state(self.prefix_plan.entries)
        
        # Warmup
        self.start_step = warmup_with_prefix(self.runtime, self.prefix_plan, self.state, self.gen_state)
        
        # Clear prefix text from state so it doesn't repeat
        self.state.pending_tokens.clear()
        self.state.lookahead_tokens.clear()
        self.state.entries.clear()
        self.state.forced_padding = 0
        
        # Warmup Mimi
        prefix_len = self.prefix_plan.aligned_frames
        prefix_tokens = self.gen_state.audio_buf[0:1, :, :prefix_len].clone()
        prefix_tokens[prefix_tokens >= 2048] = 0
        prefix_tokens[prefix_tokens < 0] = 0
        _, self.mimi_kv = self.runtime.mimi.decode_streaming(prefix_tokens, None)
        
        # Start generation after prefix
        self.current_step = self.start_step + 1
        
        # Buffers
        self.cached_graphs = CachedGraphs()
        self.cached_graphs.mimi_kv = self.mimi_kv
        
        print("[Dia2] Session ready.")

    async def add_text(self, text: str):
        normalized = normalize_script(text)
        entries = parse_script([normalized], self.runtime.tokenizer, self.runtime.constants, self.runtime.frame_rate)
        await self.input_queue.put(('text', entries))
        
    async def add_audio(self, audio_path: str, speaker: str):
        # Build a prefix plan for just this audio segment
        p_cfg = PrefixConfig(speaker_2=audio_path) if speaker == "speaker_2" else PrefixConfig(speaker_1=audio_path)
        plan = _cached_build_prefix_plan(self.runtime, p_cfg)
        await self.input_queue.put(('audio', plan))

    async def generate_loop(self, websocket: WebSocket):
        runtime = self.runtime
        state = self.state
        generation = self.gen_state
        
        # Config
        config = GenerationConfig(
            cfg_scale=1.0,
            text=SamplingConfig(0.6, 50),
            audio=SamplingConfig(0.8, 50),
            use_cuda_graph=True
        )
        
        # Tensors
        step_tokens = generation.step_tokens
        audio_buf = generation.audio_buf
        branches = step_tokens.shape[0]
        positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
        self.cached_graphs.positions = positions
        
        buffers = _allocate_network_buffers(runtime, branches)
        self.cached_graphs.buffers = buffers
        
        transformer_capture = None
        dep_captures = None
        
        _ensure_graph_cublas_ready(runtime.device)
        
        token_ids = runtime.constants
        delay_tensor = runtime.audio_delay_tensor
        
        # Streaming state
        chunk_frames = 3
        last_decode_pos = self.current_step
        sent_done = False
        
        print("[Dia2] Starting continuous generation loop...")
        
        while not self.stop_event.is_set():
            # 1. Check for new input if we are idle
            # We check for input in two cases:
            # A) We are idle (no entries, no pending tokens) -> Block and wait for input
            # B) We are busy -> Check if there is new input to append (non-blocking)
            
            new_item = None
            is_idle = not state.entries and not state.pending_tokens
            
            if is_idle:
                # Flush remaining audio frames that didn't meet the chunk size
                current_pos = self.current_step + 1
                if current_pos > last_decode_pos:
                    new_tokens = audio_buf[0:1, :, last_decode_pos:current_pos].clone()
                    new_tokens[new_tokens >= 2048] = 0
                    new_tokens[new_tokens < 0] = 0
                    
                    pcm, self.mimi_kv = runtime.mimi.decode_streaming(new_tokens, self.mimi_kv)
                    self.cached_graphs.mimi_kv = self.mimi_kv
                    waveform = torch.clamp(pcm[0, 0], -1.0, 1.0)
                    
                    if waveform.shape[0] > 0:
                        pcm16 = (waveform.detach().cpu().numpy() * 32767.0).astype(np.int16).tobytes()
                        header = struct.pack("!?", True) # Mark as last for this utterance
                        await websocket.send_bytes(header + pcm16)
                    
                    last_decode_pos = current_pos

                if not sent_done:
                    await websocket.send_text(json.dumps({"event": "done"}))
                    sent_done = True
                    print("[Dia2] Idle. Waiting for input...")
                
                try:
                    new_item = await self.input_queue.get()
                    sent_done = False
                except asyncio.CancelledError:
                    break
            else:
                try:
                    new_item = self.input_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            
            if new_item:
                item_type, item_data = new_item
                if item_type == 'text':
                    state.entries.extend(item_data)
                    print(f"[Dia2] Processing text input ({len(item_data)} entries)")
                elif item_type == 'audio':
                    # Process audio insertion immediately
                    plan = item_data
                    print(f"[Dia2] Processing audio input ({plan.aligned_frames} frames)")
                    
                    # We need to run the transformer for these frames to update state/history
                    # but we force the audio tokens from the plan
                    tokens = plan.aligned_tokens.to(runtime.device)
                    new_word_steps = set(plan.new_word_steps)
                    
                    for i in range(plan.aligned_frames):
                        t = self.current_step
                        positions.fill_(t)
                        
                        # Fill step_tokens from plan (delayed)
                        channels = tokens.shape[0]
                        for cb in range(channels):
                            delay = runtime.audio_delays[cb] if cb < len(runtime.audio_delays) else 0
                            idx = i - delay
                            value = tokens[cb, idx] if idx >= 0 else runtime.constants.audio_bos
                            step_tokens[:, 2 + cb, 0] = value
                        
                        # Forward pass to update KV cache
                        # We can't use the graph here easily because it's a different mode (forcing)
                        # So we use the eager step function
                        hidden_t = _execute_transformer_step(step_tokens, positions.expand(branches, -1), generation, runtime.transformer_step, buffers)
                        
                        # Update state machine (force words)
                        forced = runtime.constants.new_word if i in new_word_steps else runtime.constants.pad
                        main_token, aux_token, _ = runtime.machine.process(t, state, forced, is_forced=True)
                        
                        # Update audio buffer
                        generation.audio_buf[:, :, t] = step_tokens[:, :, 0]
                        
                        self.current_step += 1
                    
                    # Clear any pending tokens from this audio so they aren't generated
                    state.pending_tokens.clear()
                    state.lookahead_tokens.clear()
                    state.entries.clear()
                    state.forced_padding = 0
                    
                    # Update decode pos so we don't try to decode this inserted audio
                    last_decode_pos = self.current_step
                    
                    # Update Mimi KV by decoding the inserted audio (silently)
                    # This keeps the decoder in sync
                    new_tokens = generation.audio_buf[0:1, :, self.current_step-plan.aligned_frames:self.current_step].clone()
                    new_tokens[new_tokens >= 2048] = 0
                    new_tokens[new_tokens < 0] = 0
                    _, self.mimi_kv = runtime.mimi.decode_streaming(new_tokens, self.mimi_kv)
                    self.cached_graphs.mimi_kv = self.mimi_kv
                    
                    continue
            
            # 2. Run one generation step
            t = self.current_step
            if t + 1 >= audio_buf.shape[-1]:
                print("[Dia2] Context limit reached. Resetting session (TODO).")
                break
            
            generation.reset_dep_cache()
            positions.fill_(t)
            _fill_audio_channels(step_tokens, audio_buf, delay_tensor, t, token_ids.audio_bos)
            
            if branches > 1:
                step_tokens[1:, 0, 0] = token_ids.zero
                step_tokens[1:, 1, 0] = token_ids.pad
            
            # Transformer Step
            if config.use_cuda_graph:
                transformer_capture, dep_captures = _execute_transformer_graph(
                    runtime, step_tokens, positions.expand(branches, -1), branches, generation,
                    runtime.transformer_step, buffers, transformer_capture, dep_captures
                )
                hidden_t = transformer_capture[1]
            else:
                hidden_t = _execute_transformer_step(step_tokens, positions.expand(branches, -1), generation, runtime.transformer_step, buffers)

            # Text Sampling
            guided_text = apply_classifier_guidance(buffers.text, False, 1.0, 50)
            text_token = sample_token(guided_text[:1], temp=config.text.temperature, top_k=config.text.top_k).item()
            
            # State Machine
            main_token, aux_token, _ = runtime.machine.process(t, state, text_token)
            step_tokens[:, 0, 0] = main_token
            step_tokens[:, 1, 0] = aux_token if aux_token != -1 else token_ids.pad
            
            # Audio Sampling (CB0)
            guided_cb0 = apply_classifier_guidance(buffers.cb0, False, 1.0, 50)
            masked_cb0 = mask_audio_logits(guided_cb0[:1], token_ids.audio_pad, token_ids.audio_bos)
            codebook_token = sample_audio_logits(masked_cb0, config.audio.temperature, config.audio.top_k)
            audio_buf[:, 0, t + 1] = codebook_token
            
            # Depformer
            prev_audio = codebook_token.expand(branches)
            main_tokens = torch.full((branches,), main_token, dtype=torch.long, device=runtime.device)
            aux_tokens = torch.full((branches,), step_tokens[0, 1, 0], dtype=torch.long, device=runtime.device)
            
            for stage in range(runtime.model.depformer.num_depth):
                if config.use_cuda_graph and dep_captures:
                    dep_captures[stage] = _execute_depformer_graph(
                        stage, prev_audio, hidden_t, generation, runtime.depformer_step,
                        main_tokens, aux_tokens, buffers, dep_captures[stage]
                    )
                else:
                    _execute_depformer_stage(
                        stage, prev_audio, hidden_t, generation, runtime.depformer_step,
                        main_tokens, aux_tokens, buffers
                    )
                
                dep_logits = apply_classifier_guidance(buffers.dep[stage], False, 1.0, 50)
                stage_token = sample_audio_logits(dep_logits[:1], config.audio.temperature, config.audio.top_k)
                audio_buf[:, stage + 1, t + 1] = stage_token
                prev_audio = stage_token.expand(branches)
            
            self.current_step += 1
            
            # 3. Decode Audio
            current_pos = self.current_step + 1
            frames_to_decode = current_pos - last_decode_pos
            
            if frames_to_decode >= chunk_frames:
                new_tokens = audio_buf[0:1, :, last_decode_pos:current_pos].clone()
                new_tokens[new_tokens >= 2048] = 0
                new_tokens[new_tokens < 0] = 0
                
                pcm, self.mimi_kv = runtime.mimi.decode_streaming(new_tokens, self.mimi_kv)
                waveform = torch.clamp(pcm[0, 0], -1.0, 1.0)
                
                if waveform.shape[0] > 0:
                    pcm16 = (waveform.detach().cpu().numpy() * 32767.0).astype(np.int16).tobytes()
                    header = struct.pack("!?", False)
                    await websocket.send_bytes(header + pcm16)
                
                last_decode_pos = current_pos
                
                # Yield to event loop to allow input_queue to be populated
                await asyncio.sleep(0)

# ============================================================================
# WebSocket
# ============================================================================

@app.websocket("/ws/stream_tts")
async def stream_tts(ws: WebSocket):
    await ws.accept()
    session = None
    gen_task = None
    
    try:
        await ws.send_text(json.dumps({"event": "ready", "sample_rate": dia.sample_rate}))
        
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            msg_type = data.get("type")
            
            if msg_type == "set_voice":
                s1 = save_audio_from_base64(data["speaker_1"])
                s2 = save_audio_from_base64(data["speaker_2"])
                await ws.send_text(json.dumps({"event": "warming", "message": "Initializing session..."}))
                
                # Run setup in thread to avoid blocking
                loop = asyncio.get_running_loop()
                session = await loop.run_in_executor(None, lambda: ContinuousSession(dia._ensure_runtime(), s1, s2))
                
                # Start generation loop
                gen_task = asyncio.create_task(session.generate_loop(ws))
                await ws.send_text(json.dumps({"event": "voice_ready"}))
                
            elif msg_type == "tts":
                if session:
                    await session.add_text(data["text"])
                else:
                    await ws.send_text(json.dumps({"error": "Session not initialized"}))
                    
            elif msg_type == "append_audio":
                 if session:
                     audio_path = save_audio_from_base64(data["audio"])
                     await session.add_audio(audio_path, data.get("speaker", "speaker_2"))
                     await ws.send_text(json.dumps({"event": "appended"}))
                 else:
                     await ws.send_text(json.dumps({"error": "Session not initialized"}))
                 
    except WebSocketDisconnect:
        print("Disconnected")
    finally:
        if session: session.stop_event.set()
        if gen_task: await gen_task

def save_audio_from_base64(b64, suffix=".wav"):
    data = base64.b64decode(b64)
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, data)
    os.close(fd)
    return path
