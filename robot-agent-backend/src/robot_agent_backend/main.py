from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .gpu_service import GpuTrainingService
from .models import (
    AgentStatusResponse,
    GpuJobListResponse,
    GpuJobResponse,
    StartAgentRequest,
    StepAgentRequest,
    SubmitGpuTrainingRequest,
)
from .service import AgentOrchestrator


settings = get_settings()
orchestrator = AgentOrchestrator(settings)
gpu_service = GpuTrainingService(settings)

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


@app.post("/gpu/train", response_model=GpuJobResponse)
async def submit_gpu_training(payload: SubmitGpuTrainingRequest) -> GpuJobResponse:
    job = await gpu_service.submit_job(payload)
    return GpuJobResponse(job=job)


@app.get("/gpu/jobs", response_model=GpuJobListResponse)
async def list_gpu_jobs() -> GpuJobListResponse:
    return GpuJobListResponse(jobs=gpu_service.list_jobs())


@app.get("/gpu/jobs/{job_id}", response_model=GpuJobResponse)
async def get_gpu_job(job_id: str) -> GpuJobResponse:
    job = gpu_service.get_job(job_id)
    if job is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="GPU 작업을 찾을 수 없습니다.")
    return GpuJobResponse(job=job)
