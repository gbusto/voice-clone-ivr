from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import uuid

from .auth import get_current_user
from .db import get_db, engine
from .models import User, Voice, VoiceSession, TTSJob
from .schemas import MeResponse, UpdatePhoneRequest, CreateVoiceSessionResponse, VoiceOut, TTSRequest, TTSJobResponse, TTSJobStatusResponse
from .services.twilio_client import place_outbound_call
from .services.elevenlabs import tts_to_mp3
from .services.orpheus import tts_orpheus_to_wav
from .services.tts_worker import start_tts_job_async

router = APIRouter(tags=["api"])


@router.on_event("startup")
def on_startup():
    # Create tables on startup
    from .db import Base
    Base.metadata.create_all(bind=engine)


@router.get("/me", response_model=MeResponse)
def get_me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    voices = db.query(Voice).filter(Voice.user_id == user.id).order_by(Voice.id.desc()).all()
    return MeResponse(
        email=user.email,
        phone=user.phone,
        voices=[VoiceOut(id=v.id, name=v.name, voice_id=v.voice_id) for v in voices],
    )


@router.post("/me/phone")
def update_phone(payload: UpdatePhoneRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user.phone = payload.phone
    db.add(user)
    db.commit()
    return {"ok": True}


@router.get("/voices", response_model=List[VoiceOut])
def list_voices(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    voices = db.query(Voice).filter(Voice.user_id == user.id).order_by(Voice.id.desc()).all()
    return [VoiceOut(id=v.id, name=v.name, voice_id=v.voice_id) for v in voices]


@router.post("/voice/sessions", response_model=CreateVoiceSessionResponse)
def create_voice_session(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not user.phone:
        raise HTTPException(status_code=400, detail="Phone not set")
    import uuid
    session_id = uuid.uuid4().hex
    vs = VoiceSession(session_id=session_id, user_id=user.id, phone=user.phone, status="calling")
    db.add(vs)
    db.commit()
    db.refresh(vs)

    try:
        call_sid = place_outbound_call(user.phone, session_id)
        vs.call_sid = call_sid
        db.add(vs)
        db.commit()
    except Exception:
        vs.status = "failed"
        db.add(vs)
        db.commit()
        raise

    return CreateVoiceSessionResponse(session_id=session_id)


@router.post("/tts", response_model=TTSJobResponse)
def tts(payload: TTSRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Validate voice access for custom voices
    if not payload.voice_id.startswith("orpheus:"):
        voice = db.query(Voice).filter(Voice.voice_id == payload.voice_id, Voice.user_id == user.id).first()
        if not voice:
            raise HTTPException(status_code=404, detail="Voice not found")
    
    # Create TTS job
    job_id = uuid.uuid4().hex
    job = TTSJob(
        job_id=job_id,
        user_id=user.id,
        voice_id=payload.voice_id,
        text=payload.text,
        status="pending"
    )
    db.add(job)
    db.commit()
    
    # Start async processing
    start_tts_job_async(job_id)
    
    return TTSJobResponse(job_id=job_id, status="pending")


@router.get("/tts/jobs/{job_id}", response_model=TTSJobStatusResponse)
def get_tts_job_status(job_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.query(TTSJob).filter(TTSJob.job_id == job_id, TTSJob.user_id == user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return TTSJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        audio_url=job.audio_url,
        error_message=job.error_message
    )


