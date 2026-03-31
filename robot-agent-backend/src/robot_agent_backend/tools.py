from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .config import Settings
from .models import AgentState


class RobotTools:
    """Mockable adapter for existing robot pipeline scripts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._last_training_status: dict[str, Any] = {
            "status": "idle",
            "progress": 0,
            "message": "아직 학습이 시작되지 않았습니다.",
        }

    def collect_data(self, session_name: str, count: int, state: AgentState) -> dict[str, Any]:
        if self.settings.data_collection_script:
            result = self._run_script(
                self.settings.data_collection_script,
                {
                    "session_name": session_name,
                    "count": count,
                    "state": state.model_dump(),
                },
            )
            new_count = int(result.get("new_data_count", state.data_count))
            collected = new_count - state.data_count
            self._last_training_status = {
                "status": "data_collection",
                "progress": result.get("progress", 100),
                "message": result.get("message", f"샘플 {collected}개를 수집했습니다."),
            }
            return result

        new_count = min(state.data_count + count, state.target_data)
        collected = new_count - state.data_count
        self._last_training_status = {
            "status": "data_collection",
            "progress": 100,
            "message": f"세션 '{session_name}'에 샘플 {collected}개를 수집했습니다.",
        }
        return {
            "session_name": session_name,
            "requested_count": count,
            "collected_count": collected,
            "new_data_count": new_count,
        }

    def start_training(self, config: dict[str, Any], state: AgentState) -> dict[str, Any]:
        if self.settings.training_script:
            result = self._run_script(
                self.settings.training_script,
                {
                    "config": config,
                    "state": state.model_dump(),
                },
            )
            self._last_training_status = {
                "status": result.get("status", "completed"),
                "progress": result.get("progress", 100),
                "message": result.get("message", "학습이 성공적으로 끝났습니다."),
            }
            return result

        old_loss = state.loss
        improved_loss = max(state.target_loss, round(old_loss * 0.55, 4))
        self._last_training_status = {
            "status": "completed",
            "progress": 100,
            "message": "학습이 성공적으로 끝났습니다.",
        }
        return {
            "config": config,
            "old_loss": old_loss,
            "new_loss": improved_loss,
            "epochs_completed": config.get("epochs", 1),
        }

    def get_training_status(self) -> dict[str, Any]:
        if self.settings.status_script:
            result = self._run_script(self.settings.status_script, {})
            self._last_training_status = {
                "status": result.get("status", "unknown"),
                "progress": result.get("progress", 0),
                "message": result.get("message", "외부 상태 스크립트 결과를 받았습니다."),
            }
            return result
        return dict(self._last_training_status)

    @staticmethod
    def _run_script(script_path: str, payload: dict[str, Any]) -> dict[str, Any]:
        script = Path(script_path).expanduser()
        completed = subprocess.run(
            [str(script), json.dumps(payload)],
            check=True,
            capture_output=True,
            text=True,
        )
        stdout = completed.stdout.strip()
        if not stdout:
            return {}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"스크립트 {script}는 JSON을 출력해야 합니다. 실제 출력: {stdout}") from exc
