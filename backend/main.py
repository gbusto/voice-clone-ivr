import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from .config import settings
from .twilio_webhooks import router as twilio_router
from .routes import router as api_router
from .auth import router as auth_router


def create_app() -> FastAPI:
    app = FastAPI(title="Voice Clone IVR API", version="0.1.0")

    allowed_origins = settings.ALLOWED_ORIGINS.split(",") if settings.ALLOWED_ORIGINS else ["http://localhost:3000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY, same_site="lax")

    app.include_router(auth_router)
    app.include_router(api_router, prefix="/api")
    app.include_router(twilio_router)

    return app


app = create_app()


