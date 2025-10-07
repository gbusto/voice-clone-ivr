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


class VoiceChatClient:
    def __init__(self, ws_url: str, sample_rate: int = 24000):
        self.ws_url = ws_url
        self.sample_rate = sample_rate
        self.recording = False
        self.recorded_frames = []
        self.websocket = None
        self.session_id = None
        
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
                            
                            # Wait for response
                            print("⏳ Waiting for response...")
                            
                            # Receive JSON metadata
                            response_meta = await websocket.recv()
                            meta_data = json.loads(response_meta)
                            
                            if meta_data.get("type") == "response":
                                response_text = meta_data.get("text", "")
                                print(f"\n💬 Assistant: {response_text}")
                                
                                # Receive audio bytes
                                response_audio = await websocket.recv()
                                print(f"🔊 Playing response ({len(response_audio)} bytes)...")
                                self.play_audio(response_audio)
                                print("✨ Done!\n")
                            elif meta_data.get("type") == "error":
                                print(f"❌ Error: {meta_data.get('message')}")
                        
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
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url.startswith("ws://") and not args.url.startswith("wss://"):
        print("❌ Error: URL must start with ws:// or wss://")
        sys.exit(1)
    
    # Create and run client
    client = VoiceChatClient(args.url, args.sample_rate)
    
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

