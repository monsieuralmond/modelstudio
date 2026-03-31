from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from .agent import RobotTrainingAgent
from .config import Settings
from .events import StatusBroadcaster
from .models import (
    ActionType,
    AgentAction,
    AgentMode,
    AgentState,
    LogEntry,
    PipelineStatus,
    StartAgentRequest,
    StepAgentRequest,
)
from .tools import RobotTools


class AgentOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tools = RobotTools(settings)
        self.agent = RobotTrainingAgent(api_key=settings.openai_api_key, model=settings.openai_model)
        self.status = PipelineStatus(state=AgentState())
        self.broadcaster = StatusBroadcaster()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    async def start(self, payload: StartAgentRequest) -> PipelineStatus:
        async with self._lock:
            if self._task and not self._task.done():
                return self.status
            self.status = PipelineStatus(
                state=payload.state.model_copy(deep=True),
                mode=AgentMode.RUNNING,
                done=False,
                current_action=None,
                logs=[],
                last_tool_result={},
                session_name=payload.session_name,
                error=None,
            )
            self._append_log("system", "에이전트 반복 실행을 시작했습니다.")
            self._task = asyncio.create_task(self._run_loop(payload.training_config))
            snapshot = self.status.model_copy(deep=True)
        await self.broadcaster.broadcast_status(snapshot)
        return snapshot

    async def stop(self) -> PipelineStatus:
        task_to_cancel: asyncio.Task[None] | None = None
        async with self._lock:
            if self._task and not self._task.done():
                task_to_cancel = self._task
                self._task = None
        if task_to_cancel:
            task_to_cancel.cancel()
            with suppress(asyncio.CancelledError):
                await task_to_cancel
        async with self._lock:
            self.status.mode = AgentMode.STOPPED
            self.status.done = True
            self.status.current_action = None
            self._append_log("system", "사용자 요청으로 에이전트 실행을 멈췄습니다.")
            snapshot = self.status.model_copy(deep=True)
        await self.broadcaster.broadcast_status(snapshot)
        return snapshot

    async def status_snapshot(self) -> PipelineStatus:
        async with self._lock:
            return self.status.model_copy(deep=True)

    async def manual_step(self, payload: StepAgentRequest) -> PipelineStatus:
        async with self._lock:
            if payload.state is not None:
                self.status.state = payload.state.model_copy(deep=True)
            if payload.session_name:
                self.status.session_name = payload.session_name
            self.status.mode = AgentMode.RUNNING
            self.status.done = False

        config = payload.training_config or {"epochs": 3, "batch_size": 16, "learning_rate": 0.0005}
        await self._execute_single_step(config)
        return await self.status_snapshot()

    async def _run_loop(self, training_config: dict[str, Any]) -> None:
        try:
            while True:
                await self._execute_single_step(training_config)
                snapshot = await self.status_snapshot()
                if snapshot.done or snapshot.mode in {AgentMode.FINISHED, AgentMode.ERROR, AgentMode.STOPPED}:
                    break
                await asyncio.sleep(self.settings.agent_poll_interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with self._lock:
                self.status.mode = AgentMode.ERROR
                self.status.done = True
                self.status.error = str(exc)
                self._append_log("error", f"에이전트 실행 중 오류가 발생했습니다: {exc}")
                snapshot = self.status.model_copy(deep=True)
            await self.broadcaster.broadcast_status(snapshot)

    async def _execute_single_step(self, training_config: dict[str, Any]) -> None:
        async with self._lock:
            state = self.status.state.model_copy(deep=True)
            session_name = self.status.session_name
            if state.iteration >= state.max_iteration:
                self.status.mode = AgentMode.STOPPED
                self.status.done = True
                self.status.current_action = None
                self._append_log("system", "안전 제한 횟수에 도달해서 실행을 멈춥니다.")
                snapshot = self.status.model_copy(deep=True)
                should_stop = True
            else:
                should_stop = False
        if should_stop:
            await self.broadcaster.broadcast_status(snapshot)
            return

        action = self.agent.decide_action(state, session_name, training_config)

        async with self._lock:
            self.status.current_action = action.type
            self._append_log("agent", action.reasoning, action.type)

        await self._execute_action(action)

        async with self._lock:
            self.status.state.iteration += 1
            status_snapshot = self.tools.get_training_status()
            self.status.last_tool_result["training_status"] = status_snapshot
            self._append_log("tool", f"학습 상태: {status_snapshot['status']} - {status_snapshot['message']}", ActionType.GET_TRAINING_STATUS)
            if action.type == ActionType.FINISH:
                self.status.mode = AgentMode.FINISHED
                self.status.done = True
                self.status.current_action = None
            elif self.status.state.iteration >= self.status.state.max_iteration:
                self.status.mode = AgentMode.STOPPED
                self.status.done = True
                self.status.current_action = None
                self._append_log("system", "최대 반복 횟수에 도달해서 실행을 종료했습니다.")
            else:
                self.status.mode = AgentMode.RUNNING
            snapshot = self.status.model_copy(deep=True)
        await self.broadcaster.broadcast_status(snapshot)

    async def _execute_action(self, action: AgentAction) -> None:
        async with self._lock:
            state = self.status.state
            session_name = self.status.session_name

            if action.type == ActionType.COLLECT_DATA:
                result = self.tools.collect_data(
                    session_name=action.tool_input["session_name"],
                    count=int(action.tool_input["count"]),
                    state=state,
                )
                state.data_count = result["new_data_count"]
                self.status.last_tool_result = result
                self._append_log("tool", f"데이터가 부족해서 수집을 시작했습니다. 샘플 {result['collected_count']}개를 추가했습니다.", action.type)
                return

            if action.type == ActionType.START_TRAINING:
                result = self.tools.start_training(config=action.tool_input["config"], state=state)
                state.loss = result["new_loss"]
                self.status.last_tool_result = result
                self._append_log("tool", f"손실값이 높아서 다시 학습했습니다. 새 손실값은 {result['new_loss']}입니다.", action.type)
                return

            if action.type == ActionType.GET_TRAINING_STATUS:
                result = self.tools.get_training_status()
                self.status.last_tool_result = result
                self._append_log("tool", f"학습 진행 상황: {result['message']}", action.type)
                return

            if action.type == ActionType.FINISH:
                self.status.last_tool_result = {"message": "목표를 달성했습니다."}
                self._append_log("tool", "목표 조건을 만족해서 실행을 마칩니다.", action.type)
                return

            raise ValueError(f"지원하지 않는 액션 타입입니다: {action.type}")

    def _append_log(self, actor: str, message: str, action: ActionType | None = None) -> None:
        step = len(self.status.logs) + 1
        self.status.logs.append(LogEntry(step=step, actor=actor, message=message, action=action))
