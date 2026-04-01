from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentMode(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    FINISHED = "finished"
    ERROR = "error"


class ActionType(str, Enum):
    COLLECT_DATA = "collect_data"
    START_TRAINING = "start_training"
    GET_TRAINING_STATUS = "get_training_status"
    FINISH = "finish"


class AgentState(BaseModel):
    data_count: int = 0
    target_data: int = 100
    loss: float = 1.0
    target_loss: float = 0.1
    iteration: int = 0
    max_iteration: int = 10


class AgentAction(BaseModel):
    type: ActionType
    reasoning: str
    tool_input: dict[str, Any] = Field(default_factory=dict)


class LogEntry(BaseModel):
    step: int
    actor: str
    message: str
    action: ActionType | None = None


class PipelineStatus(BaseModel):
    state: AgentState
    mode: AgentMode = AgentMode.IDLE
    done: bool = False
    current_action: ActionType | None = None
    logs: list[LogEntry] = Field(default_factory=list)
    last_tool_result: dict[str, Any] = Field(default_factory=dict)
    session_name: str = "default-session"
    error: str | None = None


class StartAgentRequest(BaseModel):
    session_name: str = "robot-training-session"
    state: AgentState = Field(default_factory=AgentState)
    training_config: dict[str, Any] = Field(default_factory=lambda: {"epochs": 3, "batch_size": 16, "learning_rate": 0.0005})


class StepAgentRequest(BaseModel):
    session_name: str | None = None
    state: AgentState | None = None
    training_config: dict[str, Any] | None = None


class AgentStatusResponse(BaseModel):
    status: PipelineStatus


class GpuJobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class DatasetSummary(BaseModel):
    mode: str
    project_name: str
    task_names: list[str] = Field(default_factory=list)
    sample_counts: list[int] = Field(default_factory=list)
    clip_count: int = 0
    has_raw_episodes: bool = False
    notes: str = ""


class SubmitGpuTrainingRequest(BaseModel):
    session_name: str
    dataset: DatasetSummary
    training_config: dict[str, Any] = Field(default_factory=dict)


class GpuJob(BaseModel):
    job_id: str
    session_name: str
    state: GpuJobState
    message: str
    progress: int = 0
    dataset: DatasetSummary
    training_config: dict[str, Any] = Field(default_factory=dict)
    remote_job_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class GpuJobResponse(BaseModel):
    job: GpuJob


class GpuJobListResponse(BaseModel):
    jobs: list[GpuJob]


class VesslConfigRequest(BaseModel):
    access_token: str
    organization_name: str = ""
    project_name: str = ""


class VesslConfigStatus(BaseModel):
    configured: bool
    organization_name: str = ""
    project_name: str = ""


class VesslConfigResponse(BaseModel):
    status: VesslConfigStatus
