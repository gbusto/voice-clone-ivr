#!/usr/bin/env python3
"""
LFM2-Audio-1.5B Modal deployment
End-to-end audio foundation model for voice-to-voice conversation
"""
import modal
import os
import io
import json
import base64
import time
from typing import Optional, Dict, Any
import traceback

app = modal.App("lfm2-audio-modal")

# Volumes for model storage
model_volume = modal.Volume.from_name("lfm2-models", create_if_missing=True)

# Build the image with all dependencies
lfm2_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-runtime-ubuntu22.04", add_python="3.12")
    .apt_install(
        "git",
        "ffmpeg",
        "libsndfile1",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .pip_install(
        [
            "torch==2.4.0",
            "torchaudio==2.4.0",
        ],
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        [
            "transformers",
            "accelerate",
            "huggingface_hub",
            "numpy",
            "safetensors",
            "liquid-audio",
        ]
    )
    .pip_install(
        [
            "fastapi",
            "hf_transfer",
        ]
    )
    .apt_install(
        "gcc",
        "g++",
        "build-essential",
    )
    # Note: Flash Attention 2 is optional - PyTorch 2.4 SDPA is fast enough for T4
    # Skipping flash-attn to avoid build complexity (requires CUDA build env)
    .run_commands([
        "mkdir -p /models",
        "python -c 'import torch, transformers, huggingface_hub'",
    ])
)


@app.function(
    image=lfm2_image,
    timeout=60 * 60,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_models(
    repo_id: str = "LiquidAI/LFM2-Audio-1.5B",
    local_dir: str = "/models/lfm2-audio-1.5b",
) -> Dict[str, Any]:
    """Download LFM2-Audio model and processor to persistent volume"""
    from huggingface_hub import snapshot_download
    
    os.makedirs(local_dir, exist_ok=True)
    
    # Check if already downloaded
    if os.path.exists(f"{local_dir}/config.json"):
        print(f"Model already downloaded at {local_dir}")
        return {"status": "already_downloaded", "path": local_dir}
    
    print(f"Downloading {repo_id} to {local_dir}...")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    
    model_volume.commit()
    print(f"Model downloaded and committed to volume: {path}")
    return {"status": "ok", "path": path}


# Global model cache
MODEL = None
PROCESSOR = None


# Lightweight tracing/profiling controls
TRACE_ENABLED = True# os.environ.get("LFM2_TRACE", "0") not in ("0", "false", "False", "")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class _Timer:
    def __init__(self, name: str, metrics: Dict[str, Any]):
        self.name = name
        self.metrics = metrics
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = _now_ms()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = _now_ms() - self.t0
        self.metrics[f"{self.name}_ms"] = self.metrics.get(f"{self.name}_ms", 0.0) + dt


def _trace_print(*args, **kwargs):
    if TRACE_ENABLED:
        print("[TRACE]", *args, **kwargs)


def _load_lfm2_model(model_path: str = "/models/lfm2-audio-1.5b"):
    """Load the LFM2-Audio model and processor (cached globally)"""
    global MODEL, PROCESSOR
    
    if MODEL is not None and PROCESSOR is not None:
        return MODEL, PROCESSOR
    
    # Check if model exists locally
    if not os.path.exists(model_path) or not os.listdir(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please download the model first using: modal run lfm2_audio_modal.py --action download"
        )
    
    import torch
    from pathlib import Path
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
    
    print(f"Loading model from {model_path}...")
    
    # Pass Path object (not string) - liquid-audio handles Path differently than repo_id strings
    model_path_obj = Path(model_path)
    PROCESSOR = LFM2AudioProcessor.from_pretrained(model_path_obj).eval()
    MODEL = LFM2AudioModel.from_pretrained(model_path_obj).eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        MODEL = MODEL.cuda()
        print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    
    print("Model and processor loaded successfully!")
    return MODEL, PROCESSOR


@app.function(
    image=lfm2_image,
    gpu="H100",  # upgraded GPU
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=2,  # Keep alive for 2 minutes after last request
    timeout=300,  # 5 minute timeout per request
)
@modal.fastapi_endpoint(method="POST")
def tts_generate(request: Dict[str, Any]):
    """
    Simple inference endpoint for text-to-speech or audio-to-audio
    
    Request body:
    - text: str (optional) - Text input for TTS
    - audio_b64: str (optional) - Base64 encoded audio input (WAV/MP3)
    - system_prompt: str (optional) - System prompt for the model
    - max_tokens: int (optional) - Max tokens to generate (default: 512)
    - audio_temperature: float (optional) - Temperature for audio generation (default: 1.0)
    - audio_top_k: int (optional) - Top-k sampling for audio (default: 4)
    - format: str (optional) - Response format: "json" (base64) or "wav" (raw bytes)
    """
    import torch
    import torchaudio
    from liquid_audio import ChatState, LFMModality
    
    body = request if isinstance(request, dict) else json.loads(request)
    metrics: Dict[str, Any] = {"path": "tts_generate"}
    t_start = _now_ms()
    
    text_input = body.get("text")
    audio_b64 = body.get("audio_b64")
    system_prompt = body.get("system_prompt", "Respond with interleaved text and audio.")
    max_tokens = int(body.get("max_tokens", 512))
    audio_temperature = float(body.get("audio_temperature", 1.0))
    audio_top_k = int(body.get("audio_top_k", 4))
    response_format = body.get("format", "json")
    
    if not text_input and not audio_b64:
        return {"error": "Either 'text' or 'audio_b64' is required"}, 400
    
    # Load model
    with _Timer("load_model", metrics):
        model, processor = _load_lfm2_model()
        try:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    
    # Initialize chat state
    with _Timer("init_chat", metrics):
        chat = ChatState(processor)
    
    # Add system prompt
    chat.new_turn("system")
    chat.add_text(system_prompt)
    chat.end_turn()
    
    # Add user input
    chat.new_turn("user")
    
    if audio_b64:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_b64)
        audio_buffer = io.BytesIO(audio_bytes)
        wav, sampling_rate = torchaudio.load(audio_buffer)
        chat.add_audio(wav, sampling_rate)
    elif text_input:
        chat.add_text(text_input)
    
    chat.end_turn()
    
    # Generate response
    chat.new_turn("assistant")
    
    text_out = []
    audio_out = []
    modality_out = []
    
    _trace_print(f"Generating response (max_tokens={max_tokens})...")
    
    with _Timer("generate_loop", metrics):
        with torch.no_grad():
            for t in model.generate_interleaved(
                **chat, 
                max_new_tokens=max_tokens, 
                audio_temperature=audio_temperature, 
                audio_top_k=audio_top_k
            ):
                if t.numel() == 1:
                    # Text token
                    decoded = processor.text.decode(t)
                    text_out.append(t)
                    modality_out.append(LFMModality.TEXT)
                else:
                    # Audio token
                    audio_out.append(t)
                    modality_out.append(LFMModality.AUDIO_OUT)
    metrics["num_text_tokens"] = len(text_out)
    metrics["num_audio_tokens"] = len(audio_out)
    
    # Decode audio tokens to waveform
    response_text = ""
    if text_out:
        response_text = processor.text.decode(torch.cat(text_out))
    
    wav_bytes = b""
    if audio_out and len(audio_out) > 1:
        # Stack audio tokens (removing last "end-of-audio" token)
        with _Timer("mimi_decode", metrics):
            mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
            with torch.no_grad():
                waveform = processor.mimi.decode(mimi_codes)[0]
        
        # Convert to WAV bytes (force 16-bit PCM for client compatibility)
        with _Timer("wav_encode", metrics):
            wav_buffer = io.BytesIO()
            torchaudio.save(
                wav_buffer,
                waveform.cpu(),
                24000,
                format="wav",
                encoding="PCM_S",
                bits_per_sample=16,
            )
            wav_bytes = wav_buffer.getvalue()
        metrics["wav_bytes"] = len(wav_bytes)
    
    _trace_print(f"Generated text: {response_text[:100]}...")
    _trace_print(f"Generated audio: {len(wav_bytes)} bytes")
    
    # Return response based on format
    if response_format.lower() == "wav" and wav_bytes:
        from fastapi import Response as FastAPIResponse
        return FastAPIResponse(content=wav_bytes, media_type="audio/wav")
    
    # Default: JSON with base64 encoded audio
    metrics["total_ms"] = _now_ms() - t_start
    result = {
        "text": response_text,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii") if wav_bytes else None,
        "content_type": "audio/wav",
        "sample_rate": 24000,
    }
    if TRACE_ENABLED:
        result["metrics"] = metrics
    
    return result


# Global session storage for WebSocket connections
CHAT_SESSIONS = {}

# Global model cache for WebSocket endpoint
WS_MODEL = None
WS_PROCESSOR = None

def _ensure_ws_model_loaded():
    """Ensure model is loaded once for WebSocket endpoint"""
    global WS_MODEL, WS_PROCESSOR
    if WS_MODEL is None or WS_PROCESSOR is None:
        print("Loading model for WebSocket endpoint...")
        WS_MODEL, WS_PROCESSOR = _load_lfm2_model()
        print("Model loaded and ready!")
    return WS_MODEL, WS_PROCESSOR

@app.function(
    image=lfm2_image,
    gpu="H100",
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=600,  # Keep alive 10 minutes for conversations
    timeout=1800,  # 30 minute max conversation
    allow_concurrent_inputs=4,  # Support multiple simultaneous users
)
@modal.asgi_app(label="voice-chat-ws")
def voice_chat_ws():
    """WebSocket endpoint for real-time voice conversation"""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import torch
    import torchaudio
    from liquid_audio import ChatState, LFMModality
    import uuid
    import asyncio
    
    # Load model once at startup (before any connections)
    model, processor = _ensure_ws_model_loaded()
    # Enable fast math on supported GPUs
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    
    # Warm up MIMI decoder (first decode is slow)
    print("Warming up MIMI decoder...")
    try:
        dummy_codes = torch.randint(0, 2048, (1, 8, 10), dtype=torch.long)
        if torch.cuda.is_available():
            dummy_codes = dummy_codes.cuda()
        with torch.no_grad():
            _ = processor.mimi.decode(dummy_codes)
        print("MIMI decoder warmed up!")
    except Exception as e:
        print(f"MIMI warmup failed (non-critical): {e}")
    
    web_app = FastAPI()
    
    @web_app.get("/")
    async def root():
        return {"status": "LFM2-Audio Voice Chat WebSocket", "endpoint": "/ws"}
    
    @web_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        
        # Create unique session
        session_id = str(uuid.uuid4())[:8]
        print(f"[{session_id}] New connection established")
        
        try:
            
            # Initialize chat state for this session
            chat = ChatState(processor)
            
            # Add system prompt
            chat.new_turn("system")
            # NOTE: you *must* ask it to respond with interleaved text and audio. If not, it won't output audio properly (or we just get gibberish random data).
            chat.add_text("You are a helpful voice assistant. Keep responses concise and natural. Please respond with interleaved text and audio.")
            chat.end_turn()
            
            CHAT_SESSIONS[session_id] = {
                "chat": chat,
                "turn_count": 0
            }
            
            # Utility: sanitize Mimi codes (remove EOS=2048 frames, clamp to [0,2047])
            def _sanitize_codes(codes: "torch.Tensor") -> "torch.Tensor":
                # codes shape: (codebooks, time) then unsqueezed to (1, codebooks, time)
                if codes.ndim == 3:
                    codes2 = codes.squeeze(0)
                else:
                    codes2 = codes
                # mask out any time steps that contain EOS (2048) in any codebook
                valid_mask = (codes2 < 2048).all(dim=0)
                codes2 = codes2[:, valid_mask]
                if codes2.numel() == 0:
                    return codes2.new_empty((1, codes.shape[1] if codes.ndim==3 else codes.shape[0], 0))
                # clamp just in case
                codes2 = codes2.clamp_(min=0, max=2047).to(torch.long).contiguous()
                return codes2.unsqueeze(0)

            # Send ready signal
            await websocket.send_json({"type": "ready", "session_id": session_id})
            
            # Conversation loop
            while True:
                # Receive audio from client (WAV bytes)
                data = await websocket.receive()
                
                if "bytes" in data:
                    audio_bytes = data["bytes"]
                    
                    # Decode audio
                    t_decode_in_start = _now_ms()
                    audio_buffer = io.BytesIO(audio_bytes)
                    wav, sampling_rate = torchaudio.load(audio_buffer)
                    recv_decode_ms = _now_ms() - t_decode_in_start
                    
                    # Add to chat state
                    chat.new_turn("user")
                    chat.add_audio(wav, sampling_rate)
                    chat.end_turn()
                    
                    # Generate and stream response
                    chat.new_turn("assistant")
                    await websocket.send_json({
                        "type": "response_start",
                        "audio": {"sample_rate": 24000, "format": "pcm_s16"}
                    })

                    text_tokens = []
                    audio_tokens = []
                    last_sent_idx = 0
                    chunk_stride = 16  # decode and send audio every N steps
                    pending_text = ""
                    last_enqueued_sig: Optional[str] = None

                    # Separate queues for text and audio - UNBOUNDED so generation never blocks
                    text_queue: asyncio.Queue = asyncio.Queue()
                    audio_queue: asyncio.Queue = asyncio.Queue()
                    
                    decode_metrics = {"chunks": 0, "bytes": 0, "decode_ms": 0.0, "mimi_decode_ms": 0.0, "network_send_ms": 0.0}
                    text_metrics = {"text_deltas_sent": 0, "text_send_ms": 0.0}
                    gen_metrics = {
                        "tokens_generated": 0,
                        "queue_put_ms": 0.0,
                        "sanitize_ms": 0.0,
                        "actual_generation_ms": 0.0,
                    }

                    async def text_consumer():
                        """Consumes text deltas and sends over WebSocket"""
                        try:
                            while True:
                                delta = await text_queue.get()
                                if delta is None:
                                    break
                                try:
                                    t_send_start = _now_ms()
                                    await websocket.send_json({"type": "text_delta", "delta": delta})
                                    text_metrics["text_send_ms"] += _now_ms() - t_send_start
                                    text_metrics["text_deltas_sent"] += 1
                                except Exception as e:
                                    print(f"[{session_id}] Text consumer error: {e}")
                                finally:
                                    text_queue.task_done()
                        except Exception as e:
                            print(f"[{session_id}] Text consumer task error: {e}")

                    async def audio_consumer():
                        """Consumes audio chunks, decodes MIMI, and sends over WebSocket"""
                        # Dedicated CUDA stream (if available) to reduce contention
                        stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                        try:
                            while True:
                                item = await audio_queue.get()
                                if item is None:
                                    break
                                slice_codes = item  # tensor on device or CPU
                                if slice_codes.shape[2] <= 0:
                                    continue
                                try:
                                    t_total_start = _now_ms()
                                    
                                    # MIMI decode on separate CUDA stream
                                    t_mimi_start = _now_ms()
                                    target_device = processor.device if hasattr(processor, "device") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                                    codes_dev = slice_codes.to(device=target_device, dtype=torch.long, non_blocking=True).contiguous()
                                    if stream is not None:
                                        with torch.cuda.stream(stream):
                                            wave = processor.mimi.decode(codes_dev)[0]
                                            pcm = (wave.clamp(-1, 1) * 32767.0).to(torch.int16)
                                        # ensure decode finished before CPU transfer
                                        stream.synchronize()
                                        chunk_bytes = pcm.cpu().numpy().tobytes()
                                    else:
                                        with torch.no_grad():
                                            wave = processor.mimi.decode(codes_dev)[0]
                                        chunk_bytes = (wave.clamp(-1, 1) * 32767.0).to(torch.int16).cpu().numpy().tobytes()
                                    decode_metrics["mimi_decode_ms"] += _now_ms() - t_mimi_start
                                    
                                    # Network send
                                    t_send_start = _now_ms()
                                    await websocket.send_json({"type": "audio_chunk", "size": len(chunk_bytes)})
                                    await websocket.send_bytes(chunk_bytes)
                                    decode_metrics["network_send_ms"] += _now_ms() - t_send_start
                                    
                                    decode_metrics["chunks"] += 1
                                    decode_metrics["bytes"] += len(chunk_bytes)
                                    decode_metrics["decode_ms"] += _now_ms() - t_total_start
                                except Exception as dec_err:
                                    print(f"[{session_id}] Audio consumer decode error: {dec_err}")
                                finally:
                                    audio_queue.task_done()
                        except Exception as e:
                            print(f"[{session_id}] Audio consumer task error: {e}")

                    # Start both consumer tasks
                    text_consumer_task = asyncio.create_task(text_consumer())
                    audio_consumer_task = asyncio.create_task(audio_consumer())

                    print(f"[{session_id}] Generating response (streaming)...")

                    t_gen_start = _now_ms()
                    t_last_token = t_gen_start
                    with torch.no_grad():
                        for t in model.generate_interleaved(
                            **chat,
                            max_new_tokens=512,
                            audio_temperature=1.0,
                            audio_top_k=4,
                        ):
                            # Measure time since last token (actual generation time)
                            t_now = _now_ms()
                            gen_metrics["actual_generation_ms"] += t_now - t_last_token
                            gen_metrics["tokens_generated"] += 1
                            
                            if t.numel() == 1:
                                # Text token: decode and enqueue for text consumer
                                text_tokens.append(t)
                                delta = processor.text.decode(t)
                                if delta:
                                    # Enqueue text delta (non-blocking)
                                    t_put_start = _now_ms()
                                    text_queue.put_nowait(delta)
                                    gen_metrics["queue_put_ms"] += _now_ms() - t_put_start
                                    
                                    pending_text += delta
                                    # Flush audio at sentence boundaries via queue
                                    if any(p in pending_text for p in ".!?") and len(audio_tokens) > last_sent_idx:
                                        t_san_start = _now_ms()
                                        raw_codes = torch.stack(audio_tokens[last_sent_idx:], 1).unsqueeze(0)
                                        slice_codes = _sanitize_codes(raw_codes)
                                        gen_metrics["sanitize_ms"] += _now_ms() - t_san_start
                                        
                                        if slice_codes.shape[2] > 0:
                                            sig = f"{last_sent_idx}:{len(audio_tokens)}:{slice_codes.shape[2]}"
                                            if sig != last_enqueued_sig:
                                                t_put_start = _now_ms()
                                                audio_queue.put_nowait(slice_codes)
                                                gen_metrics["queue_put_ms"] += _now_ms() - t_put_start
                                                last_enqueued_sig = sig
                                        last_sent_idx = len(audio_tokens)
                                        pending_text = ""
                            else:
                                # Audio token: accumulate and enqueue in chunks
                                audio_tokens.append(t)
                                if len(audio_tokens) - last_sent_idx >= chunk_stride:
                                    t_san_start = _now_ms()
                                    raw_codes = torch.stack(audio_tokens[last_sent_idx:], 1).unsqueeze(0)
                                    slice_codes = _sanitize_codes(raw_codes)
                                    gen_metrics["sanitize_ms"] += _now_ms() - t_san_start
                                    
                                    if slice_codes.shape[2] > 0:
                                        sig = f"{last_sent_idx}:{len(audio_tokens)}:{slice_codes.shape[2]}"
                                        if sig != last_enqueued_sig:
                                            t_put_start = _now_ms()
                                            audio_queue.put_nowait(slice_codes)
                                            gen_metrics["queue_put_ms"] += _now_ms() - t_put_start
                                            last_enqueued_sig = sig
                                    last_sent_idx = len(audio_tokens)
                            
                            t_last_token = _now_ms()
                    
                    # Generation complete - measure pure generation time
                    gen_loop_ms = _now_ms() - t_gen_start

                    # Flush remaining audio (excluding trailing end-of-audio code)
                    if len(audio_tokens) > last_sent_idx + 1:
                        raw_final = torch.stack(audio_tokens[last_sent_idx:-1], 1).unsqueeze(0)
                        final_codes = _sanitize_codes(raw_final)
                        if final_codes.shape[2] > 0:
                            sig = f"{last_sent_idx}:{len(audio_tokens)-1}:{final_codes.shape[2]}"
                            if sig != last_enqueued_sig:
                                audio_queue.put_nowait(final_codes)
                                last_enqueued_sig = sig

                    # Signal consumers to finish and wait for them
                    text_queue.put_nowait(None)
                    audio_queue.put_nowait(None)
                    
                    t_consumer_wait_start = _now_ms()
                    try:
                        await asyncio.wait_for(asyncio.gather(text_consumer_task, audio_consumer_task), timeout=10)
                    except asyncio.TimeoutError:
                        text_consumer_task.cancel()
                        audio_consumer_task.cancel()
                    consumer_wait_ms = _now_ms() - t_consumer_wait_start
                    _trace_print(f"[{session_id}] gen_loop_ms={gen_loop_ms:.1f} recv_decode_ms={recv_decode_ms:.1f} chunks={decode_metrics['chunks']} bytes={decode_metrics['bytes']} decode_ms={decode_metrics['decode_ms']:.1f}")

                    # Append generated tokens to chat history
                    if text_tokens or audio_tokens:
                        modality = ([LFMModality.TEXT] * len(text_tokens)) + ([LFMModality.AUDIO_OUT] * len(audio_tokens))
                        chat.append(
                            text=torch.stack(text_tokens, 1) if text_tokens else chat.text.new_empty((1, 0)),
                            audio_out=torch.stack(audio_tokens, 1) if audio_tokens else chat.audio_out.new_empty((chat.audio_out.shape[0], 0)),
                            modality_flag=torch.tensor(modality).unsqueeze(0),
                        )
                    chat.end_turn()

                    full_text = processor.text.decode(torch.cat(text_tokens)) if text_tokens else ""
                    await websocket.send_json({"type": "response_end", "text": full_text})
                    if TRACE_ENABLED:
                        await websocket.send_json({
                            "type": "metrics",
                            "gen_loop_ms": gen_loop_ms,
                            "gen_actual_generation_ms": gen_metrics["actual_generation_ms"],
                            "gen_queue_put_ms": gen_metrics["queue_put_ms"],
                            "gen_sanitize_ms": gen_metrics["sanitize_ms"],
                            "gen_tokens_generated": gen_metrics["tokens_generated"],
                            "consumer_wait_ms": consumer_wait_ms,
                            "recv_decode_ms": recv_decode_ms,
                            "text_consumer_send_ms": text_metrics["text_send_ms"],
                            "text_consumer_deltas_sent": text_metrics["text_deltas_sent"],
                            "audio_consumer_total_ms": decode_metrics["decode_ms"],
                            "audio_consumer_mimi_decode_ms": decode_metrics["mimi_decode_ms"],
                            "audio_consumer_network_send_ms": decode_metrics["network_send_ms"],
                            "audio_consumer_chunks": decode_metrics["chunks"],
                            "audio_consumer_bytes": decode_metrics["bytes"],
                            "num_text_tokens": len(text_tokens),
                            "num_audio_tokens": len(audio_tokens),
                        })
                    print(f"[{session_id}] Response done: {full_text[:50]}...")
                                       
                    CHAT_SESSIONS[session_id]["turn_count"] += 1
                    
                elif "text" in data:
                    # Handle text messages (control commands)
                    msg = json.loads(data["text"])
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
        
        except WebSocketDisconnect:
            print(f"[{session_id}] Client disconnected")
        except Exception as e:
            # print stack trace
            print(traceback.format_exc())
            print(f"[{session_id}] Error: {e}")
            await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            # Cleanup
            if session_id in CHAT_SESSIONS:
                turns = CHAT_SESSIONS[session_id]["turn_count"]
                del CHAT_SESSIONS[session_id]
                print(f"[{session_id}] Session ended ({turns} turns)")
    
    return web_app


@app.local_entrypoint()
def main(
    action: str = "info",
    text: Optional[str] = None,
    audio_file: Optional[str] = None,
):
    """
    Local entrypoint for testing
    
    Examples:
        modal run lfm2_audio_modal.py --action download
        modal run lfm2_audio_modal.py --action test --text "Hello, how are you today?"
        modal run lfm2_audio_modal.py --action test --audio-file "input.wav"
    """
    if action == "download":
        print("Downloading LFM2-Audio model...")
        res = download_models.remote()
        print(json.dumps(res, indent=2))
    
    elif action == "test":
        if not text and not audio_file:
            raise SystemExit("Either --text or --audio-file is required for test")
        
        payload = {}
        
        if text:
            payload["text"] = text
        
        if audio_file:
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
            payload["audio_b64"] = base64.b64encode(audio_bytes).decode("ascii")
        
        print(f"Sending request with payload keys: {list(payload.keys())}")
        result = tts_generate.remote(payload)
        print(f"\nResponse text: {result.get('text', 'N/A')}")
        
        if result.get("audio_b64"):
            output_file = "lfm2_output.wav"
            audio_data = base64.b64decode(result["audio_b64"])
            with open(output_file, "wb") as f:
                f.write(audio_data)
            print(f"Audio saved to: {output_file}")
    
    elif action == "info":
        print("LFM2-Audio Modal Deployment")
        print("=" * 50)
        print("Actions:")
        print("  download - Download model to persistent volume (REQUIRED before first use)")
        print("  test     - Test inference (requires --text or --audio-file)")
        print("  info     - Show this help")
        print("\nSetup steps:")
        print("  1. modal deploy lfm2_audio_modal.py")
        print("  2. modal run lfm2_audio_modal.py --action download")
        print("  3. Use client script or test action to generate audio")
        print("\nEndpoint will be available at:")
        print(f"  https://modal.com/apps/{app.name}")
    
    else:
        raise SystemExit(f"Unknown action: {action}")

