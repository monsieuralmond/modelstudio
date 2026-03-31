from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .models import ActionType, AgentAction, AgentState


SYSTEM_PROMPT = """You are an orchestration agent for a robot training pipeline.
You must choose the next action based on pipeline state.

Rules:
- If data_count < target_data, choose collect_data.
- Else if loss > target_loss, choose start_training.
- Else choose finish.
- You may optionally choose get_training_status before training if useful, but only when it improves transparency.
- Always return JSON only.

JSON schema:
{
  "type": "collect_data" | "start_training" | "get_training_status" | "finish",
  "reasoning": "short operator-facing explanation",
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
                reasoning="Data is insufficient, starting data collection.",
                tool_input={"session_name": session_name, "count": collect_count},
            )
        if state.loss > state.target_loss:
            return AgentAction(
                type=ActionType.START_TRAINING,
                reasoning="Loss is too high, starting another training run.",
                tool_input={"config": training_config},
            )
        return AgentAction(
            type=ActionType.FINISH,
            reasoning="Target data and target loss are satisfied. Finishing the pipeline.",
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
            raise ValueError("JSON response not found in model output.")
        return json.loads(text[start : end + 1])
