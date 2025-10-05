from pydantic import BaseModel, EmailStr
from typing import Optional, List


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    phone: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserProfile(BaseModel):
    email: EmailStr
    phone: Optional[str] = None


class VoiceOut(BaseModel):
    id: int
    name: str
    voice_id: str


class MeResponse(BaseModel):
    email: EmailStr
    phone: Optional[str]
    voices: List[VoiceOut]


class UpdatePhoneRequest(BaseModel):
    phone: str


class CreateVoiceSessionResponse(BaseModel):
    session_id: str


class TTSRequest(BaseModel):
    voice_id: str
    text: str


