#!/usr/bin/env python3
"""
Client script for testing LFM2-Audio Modal endpoint
"""
import requests
import base64
import json
import argparse
import sys
from pathlib import Path


def call_lfm2_endpoint(
    endpoint_url: str,
    text: str = None,
    audio_file: str = None,
    output_file: str = "output.wav",
    system_prompt: str = "Respond with interleaved text and audio.",
    max_tokens: int = 512,
    audio_temperature: float = 1.0,
    audio_top_k: int = 4,
    response_format: str = "json",
):
    """
    Call the LFM2-Audio endpoint with text or audio input
    
    Args:
        endpoint_url: Modal endpoint URL
        text: Text input (for TTS)
        audio_file: Path to audio file (for audio-to-audio)
        output_file: Where to save the generated audio
        system_prompt: System prompt for the model
        max_tokens: Maximum tokens to generate
        audio_temperature: Temperature for audio generation
        audio_top_k: Top-k sampling for audio
        response_format: "json" or "wav"
    """
    if not text and not audio_file:
        raise ValueError("Either text or audio_file must be provided")
    
    payload = {
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "audio_temperature": audio_temperature,
        "audio_top_k": audio_top_k,
        "format": response_format,
    }
    
    if text:
        payload["text"] = text
        print(f"Sending text: {text}")
    
    if audio_file:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        payload["audio_b64"] = base64.b64encode(audio_bytes).decode("ascii")
        print(f"Sending audio file: {audio_file} ({len(audio_bytes)} bytes)")
    
    print(f"\nCalling endpoint: {endpoint_url}")
    print(f"Parameters: max_tokens={max_tokens}, temp={audio_temperature}, top_k={audio_top_k}")
    print("Generating response...")
    
    response = requests.post(endpoint_url, json=payload, timeout=300)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)
    
    if response_format == "wav":
        # Raw WAV response
        audio_data = response.content
        with open(output_file, "wb") as f:
            f.write(audio_data)
        print(f"\n✅ Audio saved to: {output_file}")
        print(f"Audio size: {len(audio_data)} bytes")
    else:
        # JSON response with base64 audio
        result = response.json()
        
        if "text" in result:
            print(f"\n📝 Generated text:\n{result['text']}")
        
        if result.get("audio_b64"):
            audio_data = base64.b64decode(result["audio_b64"])
            with open(output_file, "wb") as f:
                f.write(audio_data)
            print(f"\n✅ Audio saved to: {output_file}")
            print(f"Audio size: {len(audio_data)} bytes")
            print(f"Sample rate: {result.get('sample_rate', 24000)} Hz")
        else:
            print("\n⚠️ No audio generated")


def main():
    parser = argparse.ArgumentParser(description="Test LFM2-Audio Modal endpoint")
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Modal endpoint URL (e.g., https://modal.com/apps/...)",
    )
    parser.add_argument(
        "--text",
        help="Text input for TTS",
    )
    parser.add_argument(
        "--audio",
        help="Path to audio file for audio-to-audio",
    )
    parser.add_argument(
        "--output",
        default="lfm2_output.wav",
        help="Output audio file path (default: lfm2_output.wav)",
    )
    parser.add_argument(
        "--system-prompt",
        default="Respond with interleaved text and audio.",
        help="System prompt for the model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Audio generation temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Top-k sampling for audio (default: 4)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "wav"],
        default="json",
        help="Response format (default: json)",
    )
    
    args = parser.parse_args()
    
    if not args.text and not args.audio:
        parser.error("Either --text or --audio must be provided")
    
    try:
        call_lfm2_endpoint(
            endpoint_url=args.endpoint,
            text=args.text,
            audio_file=args.audio,
            output_file=args.output,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            audio_temperature=args.temperature,
            audio_top_k=args.top_k,
            response_format=args.format,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

