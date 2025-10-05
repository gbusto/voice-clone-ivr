## Voice Clone IVR (FastAPI + Next.js + Twilio + ElevenLabs)

A minimal demo where a logged-in user can:

- Create a cloned voice by phone (one Twilio call)
- See cloned voices in a dropdown
- Type text and synthesize speech (ephemeral MP3)

SQLite for persistence. Twilio webhooks exposed via ngrok for local dev.

### Motivation

Right now, if you want to clone your voice, you need to either upload recorded samples of yourself, or go through some web interface walkthrough. The web interface (or app interface) will require giving the browser/app microphone access which you may not want to do. And unless you have on headphones, the mic is likely a little too far away.

I thought it might be cool to instead use your phone to help clone your voice, and it turns out it's super easy to do!

---

### Prerequisites
- Python 3.11+
- Node.js 18+
- ngrok account + authtoken
- Twilio account with a voice-enabled phone number
- ElevenLabs account + API key

---

### Project Layout
- `backend/` FastAPI app, SQLite DB, Twilio webhooks, ElevenLabs integration
- `frontend/` Next.js App Router (TypeScript, Tailwind) with rewrites to backend

---

### Backend Environment
Export these in the terminal where you run the backend:

```
export SECRET_KEY='change-this'
export TWILIO_ACCOUNT_SID='ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
export TWILIO_AUTH_TOKEN='your_twilio_auth_token'
export TWILIO_CALLER_ID='+1XXXXXXXXXX'   # your Twilio number
export ELEVENLABS_API_KEY='your_elevenlabs_api_key'
export ALLOWED_ORIGINS='http://localhost:3000'
export PUBLIC_BASE_URL='https://<your-ngrok>.ngrok.io'  # set after starting ngrok
# optional (defaults to sqlite in backend/):
# export DATABASE_URL='sqlite:////absolute/path/to/backend/app.db'
```

---

### Install & Run
1) Backend deps (repo root):
```
python3 -m venv backend/.venv
source backend/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

2) Frontend deps:
```
cd frontend
npm install
cd ..
```

3) Start ngrok (new terminal):
```
brew install ngrok
ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
ngrok http 8000
```
Copy the HTTPS URL (e.g. `https://abcd1234.ngrok.io`) and export it as `PUBLIC_BASE_URL` in the backend terminal. Restart backend if already running.

4) Configure Twilio webhook:
- Twilio Console → Phone Numbers → Your Number → Voice & Fax → A CALL COMES IN
  - HTTP POST to `https://<ngrok>/voice/answer`

5) Run backend (terminal with env vars):
```
source backend/.venv/bin/activate
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

6) Run frontend (another terminal):
```
cd frontend
npm run dev
```

Visit `http://localhost:3000/signup` to create an account, or `http://localhost:3000/login` if you already have one.

---

### Demo Flow
1) Log in → `/profile`
2) Save your phone number
3) Click “Create New Voice”
   - Backend creates a session and places a Twilio call
   - Answer → “Press 1 to start recording” → speak → press 1 to stop → hang up
4) Twilio posts `/voice/recording-status`
   - Backend downloads the recording (MP3 preferred, WAV fallback)
   - Clones with ElevenLabs SDK → stores `voice_id` in SQLite
5) Profile polls `GET /api/voices` → new voice appears in dropdown
6) Enter text → “Speak” → `/api/tts` returns MP3 → audio plays in the page

---

### Notes
- Auth: email + password with session cookie
- Tables: `users`, `voices`, `voice_sessions`
- ElevenLabs: SDK for cloning; REST for TTS
- Twilio: simple IVR prompts; signature validation can be added later

---

### Troubleshooting
- Backend missing deps (`itsdangerous`, etc.): reinstall requirements in venv
- Bcrypt 72-byte limit: handled by SHA256 → bcrypt pre-hash
- ElevenLabs 401: ensure `ELEVENLABS_API_KEY` is exported; restart backend
- No call: set webhook to `POST https://<ngrok>/voice/answer` and verify `TWILIO_CALLER_ID`
- Voice not appearing: check backend logs for `[recording-status]` and verify DB rows

---

### Makefile
```
make backend   # run FastAPI backend
make frontend  # run Next.js frontend
make seed      # create demo user (demo@example.com / password)
make tunnel    # reminder to start ngrok
```

---

### Why this demo
- Clear telephony + AI story
- Easy local setup via ngrok
- Obvious extensions (rate limits, verification, streaming TTS, UI polish)


