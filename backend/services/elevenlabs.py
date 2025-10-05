import requests
from typing import Optional
from io import BytesIO
try:
    from elevenlabs.client import ElevenLabs  # type: ignore
except Exception:  # pragma: no cover
    ElevenLabs = None  # SDK optional until installed
from ..config import settings


ELEVEN_BASE = "https://api.elevenlabs.io"


def clone_voice_from_bytes(
    audio_bytes: bytes,
    name: str,
    filename: str = "recording.wav",
    content_type: str = "audio/wav",
) -> str:
    # Prefer official SDK when available
    if ElevenLabs is not None:
        client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        voice = client.voices.ivc.create(name=name, files=[BytesIO(audio_bytes)])
        voice_id = getattr(voice, "voice_id", None) or (voice.get("voice_id") if isinstance(voice, dict) else None)
        if not voice_id:
            raise RuntimeError("ElevenLabs SDK did not return voice_id")
        return voice_id

    # Fallback to raw HTTP if SDK isn't installed
    url = f"{ELEVEN_BASE}/v1/voices/add"
    headers = {"xi-api-key": settings.ELEVENLABS_API_KEY}
    files = {"files": (filename, audio_bytes, content_type)}
    data = {"name": name}
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    resp.raise_for_status()
    j = resp.json()
    voice_id = j.get("voice_id") or j.get("voice", {}).get("voice_id")
    if not voice_id:
        raise RuntimeError("Failed to obtain voice_id from ElevenLabs response")
    return voice_id


def tts_to_mp3(voice_id: str, text: str) -> bytes:
    url = f"{ELEVEN_BASE}/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": settings.ELEVENLABS_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    payload = {"text": text}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.content


