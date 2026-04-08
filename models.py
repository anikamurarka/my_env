from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class MyReward(BaseModel):
    """Structured reward breakdown for deterministic grading and debugging."""

    delta: float = Field(..., ge=-1.0, le=1.0, description="Reward earned on the latest step")
    total: float = Field(..., ge=0.0, le=1.0, description="Cumulative normalized score")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="Per-component cumulative scores")
    notes: List[str] = Field(default_factory=list, description="Why reward changed")


class MyAction(Action):
    """Agent action for maternal/newborn health triage."""

    decision: Optional[str] = Field(
        default=None,
        description="One of: reassure, schedule_follow_up, refer_facility, urgent_escalation, emergency_escalation",
    )
    question: Optional[str] = Field(
        default=None,
        description="A follow-up question to ask the patient or caregiver",
    )
    summary: Optional[str] = Field(
        default=None,
        description="A structured handoff note summarizing the situation",
    )
    resolve: bool = Field(
        default=False,
        description="Set true when you believe the triage is fully handled and final decision is made",
    )


class MyObservation(Observation):
    """Observation visible to the agent."""

    task_id: str = Field(..., description="Current task identifier")
    difficulty: str = Field(..., description="Task difficulty")
    objective: str = Field(..., description="What the agent must accomplish")
    case_id: str = Field(..., description="Patient case identifier")
    patient_description: str = Field(..., description="Initial patient description or complaint")
    dialogue_history: List[str] = Field(default_factory=list, description="History of Q&A with the patient/caregiver")
    allowed_decisions: List[str] = Field(default_factory=list)
    questions_remaining: int = 5
    last_feedback: str = ""
