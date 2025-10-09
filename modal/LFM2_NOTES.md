# LFM2-Audio on Modal — Setup & Refactor Notes

This doc doubles as a quick setup guide and a scratchpad for the ongoing low-latency streaming refactor.

## 🚀 Quick Start

### Deploy
```bash
cd modal
modal deploy lfm2_audio_modal.py
```

### Download model (first time only)
```bash
modal run lfm2_audio_modal.py --action download
```

### Start voice chat
```bash
python modal/voice_chat_client.py --url "wss://<you>--voice-chat-ws.modal.run/ws"
```

## ✅ Refactor Progress (batch 1)

- WebSocket send locking
  - Added a per-connection `asyncio.Lock` and wrapped all `send_json`/`send_bytes` calls.
  - Prevents interleaved writes from text/audio tasks and improves reliability.

- Normalized WebSocket API
  - Introduced helpers: `send_response_start`, `send_text_delta`, `send_response_end`, `ws_send_audio_chunk`, `send_metrics`.
  - Ensures consistent schema and ordering.

- Centralized MIMI sanitize
  - New `sanitize_mimi_codes()` with constants `MIMI_EOS=2048`, `MIMI_MAX_CODE=2047`.
  - Replaced ad‑hoc inline sanitize functions.

- Graceful shutdown
  - New `drain_consumers()` to signal and await text/audio consumers with timeout and cancellation.

- Metrics gating and cleanup
  - Restored `TRACE_ENABLED` env flag to avoid overhead when disabled.
  - Consolidated metrics (lean by default; detailed when tracing).

- TF32/precision setup
  - Centralized in `_setup_torch_optimizations()` and used `torch.inference_mode()`.

- Event‑loop yields to reduce time‑to‑first‑token (TTFX)
  - Inserted `await asyncio.sleep(0)` after enqueuing first text/audio slices so consumers can run immediately.

## 🔜 Next Up (batch 2+)

- First‑token fast path
  - On the first audio token, immediately enqueue a tiny slice (e.g., 4–6 frames) and yield.

- Adaptive stride (latency vs throughput)
  - Start small stride (2–4) for the first flush, ramp to 16+ as the queue grows.

- Audio consumer smoothing
  - Rebuffer decoded PCM into consistent 20–50 ms frames before sending, to reduce jitter.

- Session wrapper (structure/readability)
  - `ChatSession` class to encapsulate per-connection state and helpers.

- Queue policy (prod hardening)
  - Large but bounded queues with coalescing under backpressure (never block generation).

## 📊 Enabling Metrics (server → client)

- Per-connection toggle (recommended): append `?trace=1` to the WS URL; metrics will be emitted for that session.
```bash
python modal/voice_chat_client.py --url "wss://<you>--voice-chat-ws.modal.run/ws?trace=1" --show-metrics
```
- Global (optional): you can also hard-enable tracing in code by setting `TRACE_ENABLED = True` (dev only), or pass an env var in the `@app.function` decorator (`env={"LFM2_TRACE":"1"}`) and redeploy.

The client flag `--show-metrics` only controls printing locally; it does not turn server metrics on by itself.

## ⏱️ Per‑message timestamps & latency

The server includes a `ts_ms` (server epoch ms) on control frames (`text_delta`, `audio_chunk`, `response_*`, and `metrics`).
The client, when run with `--show-metrics`, computes `latency_ms = now_ms() - ts_ms` upon receipt and prints it. Clock skew can affect values slightly; for relative tuning this is sufficient.

## 📈 Observed latency findings (H100/H200 tests)

- Network and handshake are fast: `response_start` in ~50–400 ms.
- First audio header generally arrives in ~1.6–2.4 s after start; this is model-side generation ordering (text-first) rather than decode/network.
- `tokens_before_first_audio` ≈ 6 consistently → the model emits ~6 text tokens before audio begins.
- Client playback gap after “play … ms” was due to device buffer; setting `blocksize=240, latency='low'` removed that extra delay.

## 🛠️ Current implementation highlights

- 3-task architecture: generation loop + text consumer + audio consumer (non-blocking, queues)
- WebSocket send locking (single `asyncio.Lock`) to serialize concurrent sends
- First-audio fast path + adaptive stride (start small, ramp to steady 16) for earlier/continuous audio
- Cooperative yields (`await asyncio.sleep(0)`) to avoid event-loop starvation during fast generation
- MIMI sanitize centralized; decode on separate CUDA stream; startup MIMI warmup
- Generation warmup (tiny dummy generate) to reduce first-turn stalls
- TF32 and inference-mode enabled
- Server emits `ts_ms` on control frames; client prints per-message latency and first-play latency
- Client playback tuned: `sd.OutputStream(..., blocksize=240, latency='low')` and 50 ms prime buffer

## 🔬 What to try next (optional)

- Prompt nudge for earlier audio: ask the model to begin speaking immediately, keep text to 0–3 words.
- Reduce first-audio flush threshold from 4 → 2 frames for a slightly earlier, still-continuous start.
- Audio consumer smoothing: reframe PCM to fixed 20–50 ms chunks for steadier playback (less jitter).
- Bounded queues with coalescing (prod hardening) while keeping generation non-blocking.
- Structural cleanup: `ChatSession` to encapsulate session state; migrate away from deprecated `allow_concurrent_inputs` to `@modal.concurrent`.

## 🧪 Tips for diagnosing TTFX

- Watch for `t_first_text_delta_ms` and `t_first_audio_bytes_ms` (if enabled) to separate generation vs network delay.
- Warmup: first MIMI decode is slower; we run a dummy decode at startup.
- Client buffer: we reduced priming from 200ms → 50ms; tune as needed.

## 🧷 Appendix: original API reference (HTTP)

POST `/tts_generate` body:
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

Response (json):
```json
{
  "text": "Generated text response",
  "audio_b64": "base64 encoded WAV audio",
  "content_type": "audio/wav",
  "sample_rate": 24000
}
```

