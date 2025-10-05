import os
from pydantic import BaseModel


class Settings(BaseModel):
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_CALLER_ID: str = os.getenv("TWILIO_CALLER_ID", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-this-secret")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:////Users/gabrielbusto/projects/fun-apps/voice-clone-ivr/backend/app.db",
    )
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")


settings = Settings()


