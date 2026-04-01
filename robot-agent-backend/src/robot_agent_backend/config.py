from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    agent_poll_interval_seconds: float
    backend_host: str
    backend_port: int
    data_collection_script: str
    training_script: str
    status_script: str
    gpu_training_endpoint: str
    gpu_status_endpoint: str
    vessl_access_token: str
    vessl_default_organization: str
    vessl_default_project: str


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip() or "gpt-5.4-mini",
        agent_poll_interval_seconds=float(os.getenv("AGENT_POLL_INTERVAL_SECONDS", "1.5")),
        backend_host=os.getenv("BACKEND_HOST", "127.0.0.1").strip() or "127.0.0.1",
        backend_port=int(os.getenv("BACKEND_PORT", "8001")),
        data_collection_script=os.getenv("DATA_COLLECTION_SCRIPT", "").strip(),
        training_script=os.getenv("TRAINING_SCRIPT", "").strip(),
        status_script=os.getenv("STATUS_SCRIPT", "").strip(),
        gpu_training_endpoint=os.getenv("GPU_TRAINING_ENDPOINT", "").strip(),
        gpu_status_endpoint=os.getenv("GPU_STATUS_ENDPOINT", "").strip(),
        vessl_access_token=os.getenv("VESSL_ACCESS_TOKEN", "").strip(),
        vessl_default_organization=os.getenv("VESSL_DEFAULT_ORGANIZATION", "").strip(),
        vessl_default_project=os.getenv("VESSL_DEFAULT_PROJECT", "").strip(),
    )
