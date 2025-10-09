#!/usr/bin/env python3
"""
Voice chat client for LFM2-Audio with microphone input and speaker output
Press SPACE to record, release to send
"""
import asyncio
import sys
import wave
import io
import json
from pathlib import Path

try:
    import websockets
    import sounddevice as sd
    import numpy as np
    from pynput import keyboard
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install websockets sounddevice numpy pynput")
    sys.exit(1)


def now_epoch_ms() -> int:
    import time
    return int(time.time() * 1000)


def normalize_latency_ms(lat_ms: int) -> int:
    """If clocks are skewed (e.g., >1h), drop whole-hour multiples and keep remainder."""
    HOUR = 3600 * 1000
    if lat_ms >= HOUR:
        return lat_ms % HOUR
    if lat_ms <= -HOUR:
        return -((-lat_ms) % HOUR)
    return lat_ms


class VoiceChatClient:
    def __init__(self, ws_url: str, sample_rate: int = 24000, show_metrics: bool = False):
        self.ws_url = ws_url
        self.sample_rate = sample_rate
        self.recording = False
        self.recorded_frames = []
        self.websocket = None
        self.session_id = None
        self.show_metrics = show_metrics
        
    def on_press(self, key):
        """Called when a key is pressed"""
        if key == keyboard.Key.space and not self.recording:
            self.recording = True
            self.recorded_frames = []
            print("\n🎤 Recording... (release SPACE to send)")
            
    def on_release(self, key):
        """Called when a key is released"""
        if key == keyboard.Key.space and self.recording:
            self.recording = False
            print("⏹️  Recording stopped, processing...")
            return False  # Stop listener to trigger send
        elif key == keyboard.Key.esc:
            print("\n👋 Exiting...")
            return False
    
    def audio_callback(self, indata, frames, time, status):
        """Called for each audio block from the microphone"""
        if status:
            print(f"Audio callback status: {status}")
        if self.recording:
            self.recorded_frames.append(indata.copy())
    
    def play_audio(self, audio_bytes: bytes):
        """Play audio through speakers"""
        # Read WAV bytes and normalize to float32 for playback
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            # Some libs might send float32 PCM
            audio_array = np.frombuffer(frames, dtype=np.float32)
        else:
            # Fallback: interpret as 16-bit
            audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        if n_channels == 2:
            audio_array = audio_array.reshape(-1, 2)

        # Play as float32
        sd.play(audio_array, sample_rate)
        sd.wait()
    
    def encode_audio_to_wav(self, audio_frames: list) -> bytes:
        """Encode recorded audio frames to WAV bytes"""
        if not audio_frames:
            return b""
        
        # Concatenate all frames
        audio_data = np.concatenate(audio_frames, axis=0)
        
        # Convert float32 to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        return wav_buffer.getvalue()
    
    async def conversation_loop(self):
        """Main conversation loop"""
        async with websockets.connect(
            self.ws_url,
            open_timeout=120,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_size=None,  # allow large audio frames from server
        ) as websocket:
            self.websocket = websocket
            
            # Wait for ready signal
            ready_msg = await websocket.recv()
            ready_data = json.loads(ready_msg)
            self.session_id = ready_data.get("session_id")
            print(f"✅ Connected! Session ID: {self.session_id}")
            print("\n" + "="*60)
            print("🎙️  Press and hold SPACE to record your message")
            print("🛑 Press ESC to exit")
            print("="*60 + "\n")
            
            # Start audio input stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                dtype='float32'
            ):
                while True:
                    # Wait for space key press/release
                    with keyboard.Listener(
                        on_press=self.on_press,
                        on_release=self.on_release
                    ) as listener:
                        listener.join()
                    
                    # Check if we should exit
                    if not self.recording and not self.recorded_frames:
                        break
                    
                    # Encode and send audio
                    if self.recorded_frames:
                        audio_wav = self.encode_audio_to_wav(self.recorded_frames)
                        
                        if len(audio_wav) > 0:
                            print(f"📤 Sending {len(audio_wav)} bytes...")
                            await websocket.send(audio_wav)
                            
                            # Stream response
                            print("⏳ Waiting for response (streaming)...")

                            # Open output stream; add a small jitter buffer before playback
                            stream = sd.OutputStream(
                                samplerate=self.sample_rate,
                                channels=1,
                                dtype='float32',
                                blocksize=240,  # ~10 ms at 24 kHz
                                latency='low'
                            )
                            pending_audio = []
                            primed = False
                            first_audio_hdr_server_ms = None
                            first_audio_hdr_client_ms = None
                            first_play_printed = False

                            accumulated_text = []
                            while True:
                                msg = await websocket.recv()
                                try:
                                    data = json.loads(msg)
                                    recv_ms = now_epoch_ms()
                                    ts_ms = data.get("ts_ms")
                                    lat_ms = None
                                    if ts_ms is not None:
                                        try:
                                            lat_ms = normalize_latency_ms(recv_ms - int(ts_ms))
                                        except Exception:
                                            lat_ms = None
                                except Exception:
                                    # Binary frame (raw PCM S16LE)
                                    pcm = msg
                                    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                                    if not primed:
                                        pending_audio.append(audio)
                                        # Prime when we have ~50ms buffered (reduced for faster start)
                                        total_len = sum(a.shape[0] for a in pending_audio)
                                        if total_len >= int(0.05 * self.sample_rate):
                                            stream.start()
                                            primed = True
                                            if self.show_metrics and not first_play_printed:
                                                play_ms_cli = None
                                                play_ms_srv = None
                                                now_ms = now_epoch_ms()
                                                if first_audio_hdr_client_ms is not None:
                                                    play_ms_cli = normalize_latency_ms(now_ms - first_audio_hdr_client_ms)
                                                if first_audio_hdr_server_ms is not None:
                                                    play_ms_srv = normalize_latency_ms(now_ms - first_audio_hdr_server_ms)
                                                if play_ms_srv is not None:
                                                    print(f"[play {play_ms_srv} ms from server_hdr]")
                                                elif play_ms_cli is not None:
                                                    print(f"[play {play_ms_cli} ms from client_hdr]")
                                                first_play_printed = True
                                            # Flush buffer
                                            for a in pending_audio:
                                                stream.write(a.reshape(-1, 1))
                                            pending_audio = []
                                    else:
                                        stream.write(audio.reshape(-1, 1))
                                    continue

                                mtype = data.get("type")
                                if mtype == "response_start":
                                    if self.show_metrics and lat_ms is not None:
                                        print(f"[start {lat_ms} ms]")
                                
                                if mtype == "text_delta":
                                    delta = data.get("delta", "")
                                    if self.show_metrics and lat_ms is not None:
                                        print(f"[text {lat_ms} ms]")
                                    accumulated_text.append(delta)
                                    sys.stdout.write(delta)
                                    sys.stdout.flush()
                                elif mtype == "audio_chunk":
                                    # size metadata; actual bytes will arrive next frame (binary)
                                    if self.show_metrics and lat_ms is not None:
                                        print(f"[audio_hdr {lat_ms} ms] size={data.get('size')}")
                                    if first_audio_hdr_client_ms is None:
                                        first_audio_hdr_client_ms = recv_ms
                                        try:
                                            first_audio_hdr_server_ms = int(ts_ms) if ts_ms is not None else None
                                        except Exception:
                                            first_audio_hdr_server_ms = None
                                    pass
                                elif mtype == "response_end":
                                    if primed:
                                        stream.stop()
                                    stream.close()
                                    if self.show_metrics and lat_ms is not None:
                                        print(f"[end {lat_ms} ms]")
                                    print("\n✨ Done!\n")
                                    break
                                elif mtype == "metrics":
                                    if self.show_metrics:
                                        if lat_ms is not None:
                                            print(f"\n📊 WS latency (server→client): {lat_ms} ms")
                                        metrics = {k: data.get(k) for k in sorted(data.keys()) if k != "type"}
                                        print("\n📊 Metrics:")
                                        for k, v in metrics.items():
                                            print(f"  - {k}: {v}")
                                elif mtype == "error":
                                    if primed:
                                        stream.stop()
                                    stream.close()
                                    print(f"❌ Error: {data.get('message')}")
                                    break
                        
                        self.recorded_frames = []
            
            print("\n👋 Disconnected")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice chat with LFM2-Audio")
    parser.add_argument(
        "--url",
        required=True,
        help="WebSocket URL (e.g., wss://yourname--lfm2-audio-modal-voice-chat-ws.modal.run/ws)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Audio sample rate (default: 24000)"
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Print server-sent metrics (requires server tracing enabled)"
    )
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url.startswith("ws://") and not args.url.startswith("wss://"):
        print("❌ Error: URL must start with ws:// or wss://")
        sys.exit(1)
    
    # Create and run client
    client = VoiceChatClient(args.url, args.sample_rate, show_metrics=args.show_metrics)
    
    try:
        await client.conversation_loop()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

