PY=python3

.PHONY: dev backend frontend seed tunnel

dev:
	@echo "Starting backend and frontend..."
	$(PY) -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 &
	cd frontend && npm run dev

backend:
	$(PY) -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

seed:
	$(PY) -m backend.seed

tunnel:
	@echo "Run: ngrok http 8000"


