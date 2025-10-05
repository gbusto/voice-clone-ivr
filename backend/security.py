import hashlib
from passlib.context import CryptContext


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    # Pre-hash with SHA256 to handle any length password and then bcrypt the result
    # This is a common pattern to work around bcrypt's 72-byte limit
    prehashed = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return pwd_context.hash(prehashed)


def verify_password(password: str, hashed: str) -> bool:
    # Pre-hash with SHA256 to match the hash_password behavior
    prehashed = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return pwd_context.verify(prehashed, hashed)


