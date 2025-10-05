from twilio.rest import Client
from ..config import settings


def get_twilio_client() -> Client:
    return Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)


def place_outbound_call(to_phone: str, session_id: str) -> str:
    client = get_twilio_client()
    call = client.calls.create(
        to=to_phone,
        from_=settings.TWILIO_CALLER_ID,
        url=f"{settings.PUBLIC_BASE_URL}/voice/answer?session_id={session_id}",
        method="POST",
    )
    return call.sid


