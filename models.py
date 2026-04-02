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
    """Agent action for support-ticket triage."""

    priority: Optional[str] = Field(
        default=None,
        description="One of: low, medium, high, urgent",
    )
    team: Optional[str] = Field(
        default=None,
        description="One of: billing, technical_support, account_security, logistics",
    )
    response: Optional[str] = Field(
        default=None,
        description="A short customer reply draft",
    )
    resolve: bool = Field(
        default=False,
        description="Set true when you believe the ticket is fully handled",
    )


class MyObservation(Observation):
    """Observation visible to the agent."""

    task_id: str = Field(..., description="Current task identifier")
    difficulty: str = Field(..., description="Task difficulty")
    objective: str = Field(..., description="What the agent must accomplish")
    ticket_id: str = Field(..., description="Support ticket identifier")
    customer_tier: str = Field(..., description="free, pro, or enterprise")
    subject: str = Field(..., description="Ticket subject")
    body: str = Field(..., description="Ticket body")
    allowed_teams: List[str] = Field(default_factory=list)
    current_priority: Optional[str] = None
    current_team: Optional[str] = None
    response_sent: bool = False
    last_feedback: str = ""
