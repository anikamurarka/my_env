from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MyAction, MyObservation
except ImportError:  # pragma: no cover
    from models import MyAction, MyObservation


TASKS: List[Dict[str, Any]] = [
    {
        "id": "easy_billing_refund",
        "difficulty": "easy",
        "objective": "Classify the ticket priority, route it to the right team, send a helpful reply, then resolve it.",
        "ticket": {
            "ticket_id": "T-1001",
            "customer_tier": "pro",
            "subject": "Charged twice for monthly plan",
            "body": "Hi team, I upgraded this morning and now I see two charges on my card. Please refund the duplicate payment.",
        },
        "answer": {
            "priority": "high",
            "team": "billing",
            "keywords": ["sorry", "refund", "duplicate"],
        },
    },
    {
        "id": "medium_account_lockout",
        "difficulty": "medium",
        "objective": "Handle a customer who cannot access their account before an important deadline. Route accurately and reassure them with a clear next step.",
        "ticket": {
            "ticket_id": "T-2007",
            "customer_tier": "enterprise",
            "subject": "Account locked before board meeting",
            "body": "Our CFO cannot log in and password reset emails are not arriving. We have a board meeting in two hours and need access restored quickly.",
        },
        "answer": {
            "priority": "urgent",
            "team": "account_security",
            "keywords": ["urgent", "verify", "restore", "access"],
        },
    },
    {
        "id": "hard_multi_issue_vip",
        "difficulty": "hard",
        "objective": "Triage a VIP customer with multiple symptoms. Choose the most appropriate team and communicate both immediate containment and follow-up steps.",
        "ticket": {
            "ticket_id": "T-3014",
            "customer_tier": "enterprise",
            "subject": "Users billed twice and dashboard failing after SSO rollout",
            "body": "Since enabling SSO, some teammates cannot sign in, finance reports duplicate invoices, and leadership needs the dashboard for tomorrow's launch review. Please escalate.",
        },
        "answer": {
            "priority": "urgent",
            "team": "account_security",
            "keywords": ["escalate", "access", "billing", "dashboard", "sorry"],
        },
    },
]

ALLOWED_TEAMS = ["billing", "technical_support", "account_security", "logistics"]
ALLOWED_PRIORITIES = ["low", "medium", "high", "urgent"]
MAX_STEPS = 5


class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._task_cursor = -1
        self._forced_task_id: Optional[str] = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode: Dict[str, Any] = {}
        self._load_task(TASKS[0])

    @classmethod
    def list_tasks(cls) -> List[Dict[str, str]]:
        return [
            {
                "id": t["id"],
                "difficulty": t["difficulty"],
                "objective": t["objective"],
            }
            for t in TASKS
        ]

    def select_task(self, task_id: str) -> None:
        if task_id not in {t["id"] for t in TASKS}:
            raise ValueError(f"Unknown task_id: {task_id}")
        self._forced_task_id = task_id

    def reset(self) -> MyObservation:
        if self._forced_task_id is not None:
            task = next(t for t in TASKS if t["id"] == self._forced_task_id)
            self._forced_task_id = None
        else:
            self._task_cursor = (self._task_cursor + 1) % len(TASKS)
            task = TASKS[self._task_cursor]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._load_task(task)
        return self._observation("Environment reset. Start triaging the ticket.", reward=0.0, done=False)

    def _load_task(self, task: Dict[str, Any]) -> None:
        self._episode = {
            "task": deepcopy(task),
            "priority": None,
            "team": None,
            "response": "",
            "resolved": False,
            "score": 0.0,
            "breakdown": {"priority": 0.0, "team": 0.0, "response": 0.0, "resolution": 0.0},
            "history": [],
        }

    def _keyword_score(self, response: str, keywords: List[str]) -> float:
        text = response.lower()
        hits = sum(1 for kw in keywords if kw in text)
        return hits / max(len(keywords), 1)

    def _current_score(self) -> Dict[str, Any]:
        answer = self._episode["task"]["answer"]
        breakdown = {"priority": 0.0, "team": 0.0, "response": 0.0, "resolution": 0.0}

        if self._episode["priority"] == answer["priority"]:
            breakdown["priority"] = 0.25
        elif self._episode["priority"] in ALLOWED_PRIORITIES:
            breakdown["priority"] = 0.1

        if self._episode["team"] == answer["team"]:
            breakdown["team"] = 0.25
        elif self._episode["team"] in ALLOWED_TEAMS:
            breakdown["team"] = 0.1

        response_text = self._episode["response"].strip()
        if response_text:
            breakdown["response"] = round(0.25 * self._keyword_score(response_text, answer["keywords"]), 2)

        if self._episode["resolved"]:
            pre_resolve = breakdown["priority"] + breakdown["team"] + breakdown["response"]
            breakdown["resolution"] = 0.25 if pre_resolve >= 0.55 else 0.05

        total = round(sum(breakdown.values()), 2)
        return {"total": total, "breakdown": breakdown}

    def _observation(self, feedback: str, reward: float, done: bool) -> MyObservation:
        task = self._episode["task"]
        ticket = task["ticket"]
        score = self._current_score()
        return MyObservation(
            task_id=task["id"],
            difficulty=task["difficulty"],
            objective=task["objective"],
            ticket_id=ticket["ticket_id"],
            customer_tier=ticket["customer_tier"],
            subject=ticket["subject"],
            body=ticket["body"],
            allowed_teams=ALLOWED_TEAMS,
            current_priority=self._episode["priority"],
            current_team=self._episode["team"],
            response_sent=bool(self._episode["response"].strip()),
            last_feedback=feedback,
            reward=reward,
            done=done,
            metadata={
                "score": score["total"],
                "breakdown": score["breakdown"],
                "max_steps": MAX_STEPS,
            },
        )

    def step(self, action: MyAction) -> MyObservation:  # type: ignore[override]
        self._state.step_count += 1
        notes: List[str] = []

        if action.priority is not None:
            self._episode["priority"] = action.priority.lower().strip()
            notes.append(f"priority={self._episode['priority']}")
        if action.team is not None:
            self._episode["team"] = action.team.lower().strip()
            notes.append(f"team={self._episode['team']}")
        if action.response is not None:
            self._episode["response"] = action.response.strip()
            notes.append("response_updated")
        if action.resolve:
            self._episode["resolved"] = True
            notes.append("resolve_requested")

        prev_score = self._episode["score"]
        score = self._current_score()
        total = score["total"]
        delta = round(total - prev_score, 2)
        self._episode["score"] = total
        self._episode["breakdown"] = score["breakdown"]
        self._episode["history"].append({"step": self._state.step_count, "action": action.model_dump(), "score": total})

        invalid = False
        if action.priority and action.priority.lower() not in ALLOWED_PRIORITIES:
            delta -= 0.05
            invalid = True
            notes.append("invalid_priority")
        if action.team and action.team.lower() not in ALLOWED_TEAMS:
            delta -= 0.05
            invalid = True
            notes.append("invalid_team")
        if not any([action.priority is not None, action.team is not None, action.response is not None, action.resolve]):
            delta -= 0.03
            notes.append("empty_action")

        delta = round(max(min(delta, 1.0), -1.0), 2)
        done = bool(self._episode["resolved"] or self._state.step_count >= MAX_STEPS)

        if done and not self._episode["resolved"]:
            notes.append("episode_ended_at_step_limit")
        if done and self._episode["resolved"] and not invalid:
            notes.append("ticket_closed")

        feedback = "; ".join(notes) if notes else "State updated."
        return self._observation(feedback, reward=delta, done=done)

    @property
    def state(self) -> State:
        score = self._current_score()
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            metadata={
                "task_id": self._episode["task"]["id"],
                "score": score["total"],
                "breakdown": score["breakdown"],
                "history": self._episode["history"],
            },
        )

    def grade_current_episode(self) -> Dict[str, Any]:
        score = self._current_score()
        return {
            "task_id": self._episode["task"]["id"],
            "difficulty": self._episode["task"]["difficulty"],
            "score": score["total"],
            "breakdown": score["breakdown"],
            "done": self._episode["resolved"] or self._state.step_count >= MAX_STEPS,
        }
