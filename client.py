from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MyAction, MyObservation


class MyEnv(EnvClient[MyAction, MyObservation, State]):
    """WebSocket client for the support-ticket triage environment."""

    def _step_payload(self, action: MyAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MyObservation]:
        obs_data = payload.get("observation", {})
        observation = MyObservation(
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            objective=obs_data.get("objective", ""),
            ticket_id=obs_data.get("ticket_id", ""),
            customer_tier=obs_data.get("customer_tier", ""),
            subject=obs_data.get("subject", ""),
            body=obs_data.get("body", ""),
            allowed_teams=obs_data.get("allowed_teams", []),
            current_priority=obs_data.get("current_priority"),
            current_team=obs_data.get("current_team"),
            response_sent=obs_data.get("response_sent", False),
            last_feedback=obs_data.get("last_feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            metadata=payload.get("metadata", {}),
        )
