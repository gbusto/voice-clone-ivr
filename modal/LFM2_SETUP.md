# LFM2-Audio on Modal Setup Guide

This guide walks you through deploying and using [LFM2-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B) on Modal for real-time voice-to-voice conversation.

## 🚀 Quick Start

### 1. Deploy to Modal

First, deploy the Modal app:

```bash
cd modal
modal deploy lfm2_audio_modal.py
```

This will:
- Build the Docker image with CUDA, PyTorch, and liquid-audio
- Create persistent volumes for model storage
- Deploy the inference endpoint
- Give you an endpoint URL like: `https://yourname--lfm2-audio-modal-tts-generate.modal.run`

### 2. Download the Model

**Required before first use:**

```bash
modal run lfm2_audio_modal.py --action download
```

This downloads ~1.5B parameters to the persistent volume (takes a few minutes).

### 3. Test the Endpoint

```bash
python modal/lfm2_client.py \
  --endpoint "https://yourname--lfm2-audio-modal-tts-generate.modal.run" \
  --text "Hello! I'm LFM2 Audio. Nice to meet you!" \
  --output test.wav

# Play the audio (macOS)
afplay test.wav
```

### 4. Audio-to-Audio Test

```bash
python modal/lfm2_client.py \
  --endpoint "https://yourname--lfm2-audio-modal-tts-generate.modal.run" \
  --audio question.wav \
  --output response.wav
```

## 📝 API Reference

### Endpoint: POST `/tts_generate`

**Request Body (JSON):**

```json
{
  "text": "Optional text input for TTS",
  "audio_b64": "Optional base64 encoded audio (WAV/MP3)",
  "system_prompt": "Respond with interleaved text and audio.",
  "max_tokens": 512,
  "audio_temperature": 1.0,
  "audio_top_k": 4,
  "format": "json"
}
```

**Response (format="json"):**

```json
{
  "text": "Generated text response",
  "audio_b64": "base64 encoded WAV audio",
  "content_type": "audio/wav",
  "sample_rate": 24000
}
```

**Response (format="wav"):**

Raw WAV bytes with `Content-Type: audio/wav`

## 🎯 Current Features

- ✅ Text-to-Speech (TTS)
- ✅ Audio-to-Audio (voice conversation)
- ✅ Interleaved text+audio generation
- ✅ T4 GPU support (efficient for 1.5B params)
- ✅ Auto-scaling with 2-minute window
- ✅ Base64 JSON or raw WAV responses

## 🎙️ Real-Time Voice Chat (NEW!)

### Setup

The WebSocket endpoint is now available! Use the Python client for voice conversations.

**1. Install client dependencies:**

```bash
pip install websockets sounddevice numpy pynput
```

**2. Deploy the WebSocket endpoint:**

```bash
modal deploy lfm2_audio_modal.py
```

**3. Get your WebSocket URL:**

Go to Modal dashboard and find the `voice_chat_ws` endpoint URL, or check:

```bash
modal app list
```

It will be something like:
```
wss://yourname--lfm2-audio-modal-voice-chat-ws.modal.run/ws
```

**4. Start voice chat:**

```bash
python modal/voice_chat_client.py --url "wss://yourname--lfm2-audio-modal-voice-chat-ws.modal.run/ws"
```

### How to Use

- **Press and hold SPACE** to record your message
- **Release SPACE** to send and get a response
- **Press ESC** to exit

The conversation history is maintained during your session!

### Architecture

- **WebSocket connection** - persistent, low-latency
- **In-memory sessions** - each connection gets its own `ChatState`
- **GPU efficiency** - 10-minute scaledown window keeps conversations fast
- **Multi-user support** - handles up to 4 simultaneous conversations

## 💡 Performance Notes

- **Model size**: 1.5B params @ bfloat16 = ~3GB VRAM
- **T4 GPU**: 16GB VRAM → plenty of headroom
- **Latency**: Designed for low-latency real-time conversation
- **Sample rate**: 24kHz audio output (via Mimi tokenizer)
- **Context**: 32,768 tokens

## 🔧 Troubleshooting

**Model download fails:**
- Ensure `huggingface-secret` is set in Modal
- Check HF token has access to LiquidAI models

**CUDA out of memory:**
- T4 should be sufficient; try A10G if issues persist
- Check `max_tokens` parameter (lower = less memory)

**Audio quality issues:**
- Adjust `audio_temperature` (lower = more deterministic)
- Try different `audio_top_k` values (4 is recommended)

## 🌐 Resources

- [LFM2-Audio HuggingFace](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B)
- [liquid-audio Package](https://github.com/LiquidAI/liquid-audio)
- [Modal Documentation](https://modal.com/docs)

