from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .db import get_db
from .models import User
from .schemas import SignupRequest, LoginRequest, UserProfile
from .security import hash_password, verify_password


router = APIRouter(prefix="/auth", tags=["auth"])


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@router.post("/signup", response_model=UserProfile)
def signup(payload: SignupRequest, request: Request, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=payload.email, password_hash=hash_password(payload.password), phone=payload.phone)
    db.add(user)
    db.commit()
    db.refresh(user)

    request.session["user_id"] = user.id
    return UserProfile(email=user.email, phone=user.phone)


@router.post("/login", response_model=UserProfile)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    request.session["user_id"] = user.id
    return UserProfile(email=user.email, phone=user.phone)


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


