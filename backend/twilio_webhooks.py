from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session
import requests
import logging

from .db import get_db
from .models import VoiceSession, Voice, User
from .config import settings
from .services.elevenlabs import clone_voice_from_bytes


router = APIRouter(tags=["twilio"])


def twiml(content: str) -> Response:
    return Response(content=content, media_type="text/xml")


@router.post("/voice/answer")
def voice_answer(request: Request):
    session_id = request.query_params.get("session_id", "")
    xml = f"""
    <Response>
      <Gather input="dtmf" numDigits="1" action="/voice/start?session_id={session_id}" method="POST">
        <Say voice="alice">Welcome to the voice cloning demo. Press 1 to start recording.</Say>
      </Gather>
      <Say voice="alice">We didn't receive any input. Goodbye.</Say>
    </Response>
    """
    return twiml(xml)


@router.post("/voice/start")
def voice_start(request: Request):
    session_id = request.query_params.get("session_id", "")
    xml = f"""
    <Response>
      <Say voice="alice">Please speak after the beep. Press 1 to stop recording.</Say>
      <Record playBeep="true" finishOnKey="1" maxLength="180" action="/voice/after-record?session_id={session_id}" recordingStatusCallback="{settings.PUBLIC_BASE_URL}/voice/recording-status?session_id={session_id}" method="POST" />
    </Response>
    """
    return twiml(xml)


@router.post("/voice/after-record")
def voice_after_record(request: Request):
    xml = """
    <Response>
      <Say voice=\"alice\">Thank you. We will process your voice now. Goodbye.</Say>
      <Hangup />
    </Response>
    """
    return twiml(xml)


@router.post("/voice/recording-status")
async def recording_status(request: Request, db: Session = Depends(get_db)):
    session_id = request.query_params.get("session_id", "")
    form = await request.form()
    recording_url = form.get("RecordingUrl")
    recording_sid = form.get("RecordingSid")
    call_sid = form.get("CallSid")

    if not session_id or not recording_url:
        return Response(status_code=400)

    vs = db.query(VoiceSession).filter(VoiceSession.session_id == session_id).first()
    if not vs:
        return Response(status_code=404)

    vs.status = "processing"
    vs.recording_sid = recording_sid
    vs.call_sid = call_sid
    vs.recording_url = recording_url
    db.add(vs)
    db.commit()

    # Download audio from Twilio (append .wav)
    try:
        # Try MP3 first per ElevenLabs docs; fallback to WAV
        for ext, content_type in [("mp3", "audio/mpeg"), ("wav", "audio/wav")]:
            url = f"{recording_url}.{ext}"
            logging.info(f"[recording-status] Fetching recording: {url}")
            audio_resp = requests.get(
                url,
                auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
                timeout=120,
            )
            if audio_resp.ok and audio_resp.content:
                audio_bytes = audio_resp.content
                logging.info(
                    f"[recording-status] Downloaded {len(audio_bytes)} bytes, content-type={audio_resp.headers.get('Content-Type')}"
                )
                chosen_ext = ext
                chosen_ct = content_type
                break
        else:
            logging.error("[recording-status] Failed to download recording in both MP3 and WAV")
            raise RuntimeError("Failed to download recording")

        # Create voice in ElevenLabs
        voice_name = f"demo-{session_id[:8]}"
        filename = f"recording.{chosen_ext}"
        voice_id = clone_voice_from_bytes(audio_bytes, voice_name, filename=filename, content_type=chosen_ct)

        # Persist voice
        voice = Voice(user_id=vs.user_id, name=voice_name, voice_id=voice_id)
        db.add(voice)
        vs.status = "ready"
        db.add(vs)
        db.commit()
        logging.info(f"[recording-status] ElevenLabs voice created: {voice_id}")
    except Exception as e:
        logging.exception(f"[recording-status] Error processing recording for session {session_id}: {e}")
        vs.status = "failed"
        db.add(vs)
        db.commit()
        # Still return 200 so Twilio doesn't retry excessively; logs should capture details
        return Response(status_code=200)

    return Response(status_code=200)


