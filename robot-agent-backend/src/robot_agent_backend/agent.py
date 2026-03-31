from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .models import ActionType, AgentAction, AgentState


SYSTEM_PROMPT = """너는 로봇 학습 파이프라인을 관리하는 오케스트레이션 에이전트다.
현재 상태를 보고 다음 행동을 결정해야 한다.

규칙:
- data_count < target_data 이면 collect_data를 선택한다.
- 그렇지 않고 loss > target_loss 이면 start_training을 선택한다.
- 둘 다 아니면 finish를 선택한다.
- get_training_status는 상태를 더 분명하게 보여줄 필요가 있을 때만 선택한다.
- reasoning은 반드시 한국어로 짧고 이해하기 쉽게 쓴다.
- 출력은 반드시 JSON만 한다.

JSON schema:
{
  "type": "collect_data" | "start_training" | "get_training_status" | "finish",
  "reasoning": "짧은 한국어 설명",
  "tool_input": {}
}
"""


class RobotTrainingAgent:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None

    def decide_action(self, state: AgentState, session_name: str, training_config: dict[str, Any]) -> AgentAction:
        if self.client is None:
            return self._deterministic_decision(state, session_name, training_config)

        prompt = {
            "session_name": session_name,
            "state": state.model_dump(),
            "training_config": training_config,
        }
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )
        payload = self._extract_json(response.output_text)
        action = AgentAction.model_validate(payload)
        return self._normalize_action(action, state, session_name, training_config)

    def _deterministic_decision(
        self,
        state: AgentState,
        session_name: str,
        training_config: dict[str, Any],
    ) -> AgentAction:
        if state.data_count < state.target_data:
            remaining = state.target_data - state.data_count
            collect_count = min(remaining, max(10, remaining))
            return AgentAction(
                type=ActionType.COLLECT_DATA,
                reasoning="데이터가 아직 부족해서 수집을 시작합니다.",
                tool_input={"session_name": session_name, "count": collect_count},
            )
        if state.loss > state.target_loss:
            return AgentAction(
                type=ActionType.START_TRAINING,
                reasoning="손실값이 높아서 다시 학습을 시작합니다.",
                tool_input={"config": training_config},
            )
        return AgentAction(
            type=ActionType.FINISH,
            reasoning="목표 데이터와 목표 손실값을 만족해서 파이프라인을 마칩니다.",
            tool_input={},
        )

    def _normalize_action(
        self,
        action: AgentAction,
        state: AgentState,
        session_name: str,
        training_config: dict[str, Any],
    ) -> AgentAction:
        if action.type == ActionType.COLLECT_DATA:
            remaining = max(state.target_data - state.data_count, 0)
            action.tool_input = {
                "session_name": action.tool_input.get("session_name", session_name),
                "count": int(action.tool_input.get("count", max(10, remaining or 10))),
            }
        elif action.type == ActionType.START_TRAINING:
            action.tool_input = {"config": action.tool_input.get("config", training_config)}
        else:
            action.tool_input = action.tool_input or {}
        return action

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("모델 응답에서 JSON을 찾지 못했습니다.")
        return json.loads(text[start : end + 1])
