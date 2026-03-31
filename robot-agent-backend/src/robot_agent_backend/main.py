from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .models import AgentStatusResponse, PipelineStatus, StartAgentRequest, StepAgentRequest
from .service import AgentOrchestrator


settings = get_settings()
orchestrator = AgentOrchestrator(settings)

app = FastAPI(title="Robot Training Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/start", response_model=AgentStatusResponse)
async def start_agent(payload: StartAgentRequest) -> AgentStatusResponse:
    status = await orchestrator.start(payload)
    return AgentStatusResponse(status=status)


@app.post("/agent/stop", response_model=AgentStatusResponse)
async def stop_agent() -> AgentStatusResponse:
    status = await orchestrator.stop()
    return AgentStatusResponse(status=status)


@app.post("/agent/step", response_model=AgentStatusResponse)
async def step_agent(payload: StepAgentRequest) -> AgentStatusResponse:
    status = await orchestrator.manual_step(payload)
    return AgentStatusResponse(status=status)


@app.get("/agent/status", response_model=AgentStatusResponse)
async def agent_status() -> AgentStatusResponse:
    status = await orchestrator.status_snapshot()
    return AgentStatusResponse(status=status)


@app.websocket("/agent/ws")
async def agent_status_ws(websocket: WebSocket) -> None:
    await orchestrator.broadcaster.connect(websocket)
    try:
        snapshot = await orchestrator.status_snapshot()
        await websocket.send_json({"type": "status", "status": snapshot.model_dump(mode="json")})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        orchestrator.broadcaster.disconnect(websocket)
    except Exception:
        orchestrator.broadcaster.disconnect(websocket)
