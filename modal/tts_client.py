#!/usr/bin/env python3
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import requests


def synthesize(endpoint: str, payload: dict, timeout: int) -> bytes:
    headers = {
        "accept": "application/json,audio/wav",
        "content-type": "application/json",
    }
    resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=timeout)
    if resp.status_code >= 400:
        # Try to surface JSON error details if present
        try:
            err = resp.json()
        except Exception:
            err = {"detail": resp.text}
        raise SystemExit(f"Request failed {resp.status_code}: {err}")
    # If JSON, decode base64; otherwise assume raw WAV
    ctype = resp.headers.get("content-type", "")
    if "application/json" in ctype:
        data = resp.json()
        b64 = data.get("audio_b64")
        if not b64:
            raise SystemExit("Response JSON missing 'audio_b64'")
        import base64
        return base64.b64decode(b64)
    return resp.content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Client for Orpheus Modal TTS web endpoint (returns WAV)",
    )
    parser.add_argument(
        "endpoint",
        help="Modal web endpoint URL for tts_generate (e.g. https://<region>.modal.run/<path>)",
    )
    parser.add_argument(
        "--prompt",
        default="Hello from Orpheus!",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--voice",
        default="tara",
        help="Voice name (default: tara)",
    )
    parser.add_argument(
        "--model-dir",
        default="/models/orpheus-3b-0.1-ft",
        dest="model_dir",
        help="Model directory on Modal volume (default: /models/orpheus-3b-0.1-ft)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        dest="top_p",
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        dest="max_tokens",
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.3,
        dest="repetition_penalty",
        help="Repetition penalty",
    )
    parser.add_argument(
        "-o",
        "--out",
        dest="outfile",
        default=None,
        help="Output WAV file path (default auto-generated)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP timeout in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.prompt:
        raise SystemExit("--prompt is required")

    payload = {
        "prompt": args.prompt,
        "voice": args.voice,
        "model_dir": args.model_dir,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
    }

    audio_bytes = synthesize(args.endpoint, payload, timeout=args.timeout)

    # Determine output path
    if args.outfile:
        out_path = Path(args.outfile)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_voice = "".join(c for c in args.voice if c.isalnum() or c in ("-", "_")) or "voice"
        out_path = Path(f"orpheus_{safe_voice}_{ts}.wav")

    out_path.write_bytes(audio_bytes)
    print(f"Saved {len(audio_bytes)} bytes to {out_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)


