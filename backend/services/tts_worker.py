import threading
import time
import base64
from datetime import datetime
from sqlalchemy.orm import Session
from ..db import SessionLocal
from ..models import TTSJob
from .orpheus import tts_orpheus_to_wav
from .elevenlabs import tts_to_mp3


def process_tts_job(job_id: str):
    """Background worker to process a single TTS job"""
    db = SessionLocal()
    try:
        job = db.query(TTSJob).filter(TTSJob.job_id == job_id).first()
        if not job:
            return
        
        # Mark as processing
        job.status = "processing"
        job.updated_at = datetime.utcnow()
        db.commit()
        
        try:
            # Check if Orpheus or ElevenLabs
            if job.voice_id.startswith("orpheus:"):
                voice_name = job.voice_id.split(":", 1)[1] or "tara"
                audio_bytes = tts_orpheus_to_wav(prompt=job.text, voice=voice_name)
                # Store as base64 data URL for simplicity
                b64 = base64.b64encode(audio_bytes).decode('ascii')
                job.audio_url = f"data:audio/wav;base64,{b64}"
            else:
                audio_bytes = tts_to_mp3(job.voice_id, job.text)
                b64 = base64.b64encode(audio_bytes).decode('ascii')
                job.audio_url = f"data:audio/mpeg;base64,{b64}"
            
            job.status = "completed"
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
        
        job.updated_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        print(f"Error processing TTS job {job_id}: {e}")
    finally:
        db.close()


def start_tts_job_async(job_id: str):
    """Start TTS job processing in a background thread"""
    thread = threading.Thread(target=process_tts_job, args=(job_id,), daemon=True)
    thread.start()
