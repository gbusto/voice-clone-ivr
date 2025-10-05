from .db import engine, Base, SessionLocal
from .models import User
from .security import hash_password


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        email = "demo@example.com"
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(email=email, password_hash=hash_password("password"), phone="+1XXXXXXXXXX")
            db.add(user)
            db.commit()
            print("Created demo user:", email, "password")
        else:
            print("Demo user already exists:", email)
    finally:
        db.close()


if __name__ == "__main__":
    main()


