# Robot Agent Backend

FastAPI backend for a robot-training AI agent that decides whether to collect more data or start training.

If you already have collection, training, and status scripts, set `DATA_COLLECTION_SCRIPT`, `TRAINING_SCRIPT`, and `STATUS_SCRIPT` in `.env`. Each script should accept a single JSON string argument and print JSON to stdout.

## Run

```bash
cd robot-agent-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
uvicorn robot_agent_backend.main:app --reload --app-dir src --host 127.0.0.1 --port 8001
```

## Endpoints

- `POST /agent/start`
- `POST /agent/stop`
- `POST /agent/step`
- `GET /agent/status`
- `WS /agent/ws`
