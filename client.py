from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import MyAction, MyObservation
except ImportError:
    from models import MyAction, MyObservation  # flat /tmp/workspace/ layout


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
            case_id=obs_data.get("case_id", ""),
            patient_description=obs_data.get("patient_description", ""),
            dialogue_history=obs_data.get("dialogue_history", []),
            allowed_decisions=obs_data.get("allowed_decisions", []),
            questions_remaining=obs_data.get("questions_remaining", 5),
            last_feedback=obs_data.get("last_feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata") or payload.get("metadata") or payload.get("info") or {},
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
