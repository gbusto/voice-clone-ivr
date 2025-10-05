import base64
import requests
from ..config import settings


def tts_orpheus_to_wav(prompt: str, voice: str = "tara") -> bytes:
    if not settings.ORPHEUS_MODAL_ENDPOINT:
        raise RuntimeError("ORPHEUS_MODAL_ENDPOINT not configured")
    url = settings.ORPHEUS_MODAL_ENDPOINT
    headers = {"content-type": "application/json", "accept": "application/json"}
    payload = {
        "prompt": prompt,
        "voice": voice,
        # Return JSON base64 by default; server also supports format="wav"
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "application/json" in ctype:
        j = resp.json()
        b64 = j.get("audio_b64")
        if not b64:
            raise RuntimeError("Modal Orpheus response missing audio_b64")
        return base64.b64decode(b64)
    return resp.content


