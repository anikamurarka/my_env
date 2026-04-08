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
        "id": "easy_obvious_emergency",
        "difficulty": "easy",
        "objective": "Identify the critical danger sign and escalate immediately.",
        "case": {
            "case_id": "C-101",
            "patient_description": "I am 38 weeks pregnant and I have been bleeding heavily for the last hour.",
            "responses": {
                "pain": "I have some cramps but the bleeding is the scary part. It's soaking through pads.",
                "baby": "The baby is moving less than normal.",
            }
        },
        "answer": {
            "decision": "emergency_escalation",
            "keywords": ["bleeding", "heavy", "38 weeks", "emergency", "hemorrhage"],
        },
    },
    {
        "id": "medium_ambiguous_symptoms",
        "difficulty": "medium",
        "objective": "Ask the right follow-up question to identify preeclampsia risk, then escalate or follow-up appropriately.",
        "case": {
            "case_id": "C-205",
            "patient_description": "I had my baby 3 days ago. My feet are very swollen today.",
            "responses": {
                "headach": "Yes, I have a terrible headache that won't go away with paracetamol.",
                "vision": "A little bit blurry today.",
                "fever": "No fever, just the swelling and headache.",
                "pain": "No stomach pain.",
            }
        },
        "answer": {
            "decision": "urgent_escalation",
            "keywords": ["headache", "preeclampsia", "vision", "postpartum", "swollen"],
        },
    },
    {
        "id": "hard_noisy_comorbid",
        "difficulty": "hard",
        "objective": "Clarify vague complaints about newborn feeding/breathing and refer to facility.",
        "case": {
            "case_id": "C-314",
            "patient_description": "The baby is just sleeping all the time and feels a bit hot. Also making a funny noise.",
            "responses": {
                "nois": "Like a grunting sound when breathing in. His chest pulls in deep.",
                "breath": "Breathing very fast, maybe 70 times a minute.",
                "feed": "He hasn't taken milk since yesterday evening. Too sleepy.",
                "milk": "He hasn't taken milk since yesterday evening. Too sleepy.",
                "sleep": "Can't wake him up easily.",
            }
        },
        "answer": {
            "decision": "emergency_escalation",
            "keywords": ["grunting", "chest", "fast breathing", "feeding", "lethargic"],
        },
    },
]

ALLOWED_DECISIONS = ["reassure", "schedule_follow_up", "refer_facility", "urgent_escalation", "emergency_escalation"]
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
        return self._observation("Environment reset. Start triaging the patient case.", reward=0.0, done=False)

    def _load_task(self, task: Dict[str, Any]) -> None:
        self._episode = {
            "task": deepcopy(task),
            "decision": None,
            "questions_asked": 0,
            "dialogue_history": [],
            "summary": "",
            "resolved": False,
            "score": 0.0,
            "breakdown": {"questions": 0.0, "decision": 0.0, "summary": 0.0, "efficiency": 0.0},
            "history": [],
        }

    def _keyword_score(self, text: str, keywords: List[str]) -> float:
        t = text.lower()
        hits = sum(1 for kw in keywords if kw in t)
        return hits / max(len(keywords), 1)

    def _current_score(self) -> Dict[str, Any]:
        answer = self._episode["task"]["answer"]
        breakdown = {"questions": 0.0, "decision": 0.0, "summary": 0.0, "efficiency": 0.0}

        # Questions score: 0.1 for just asking something, up to 0.25 if good info elicited.
        if self._episode["questions_asked"] > 0:
            breakdown["questions"] = 0.25

        # Decision score
        if self._episode["decision"] == answer["decision"]:
            breakdown["decision"] = 0.40
        elif self._episode["decision"] in ALLOWED_DECISIONS:
            # Harsh penalty for reassure if it's an emergency
            if self._episode["decision"] == "reassure" and "escalation" in answer["decision"]:
                breakdown["decision"] = -0.50
            else:
                breakdown["decision"] = 0.10

        # Summary score
        summary_text = self._episode["summary"].strip()
        if summary_text:
            breakdown["summary"] = round(0.25 * self._keyword_score(summary_text, answer["keywords"]), 2)

        # Efficiency
        if self._episode["resolved"]:
            eff = 0.10
            if self._episode["questions_asked"] > 3:
                eff = 0.0  # slight penalty to efficiency for asking too many questions
            breakdown["efficiency"] = eff

        total = round(sum(breakdown.values()), 2)
        total = max(0.0, min(1.0, total))  # clamp
        return {"total": total, "breakdown": breakdown}

    def _observation(self, feedback: str, reward: float, done: bool) -> MyObservation:
        task = self._episode["task"]
        case = task["case"]
        score = self._current_score()
        return MyObservation(
            task_id=task["id"],
            difficulty=task["difficulty"],
            objective=task["objective"],
            case_id=case["case_id"],
            patient_description=case["patient_description"],
            dialogue_history=self._episode["dialogue_history"],
            allowed_decisions=ALLOWED_DECISIONS,
            questions_remaining=MAX_STEPS - self._episode["questions_asked"],
            last_feedback=feedback,
            reward=reward,
            done=done,
            metadata={
                "score": score["total"],
                "breakdown": score["breakdown"],
            },
        )

    def step(self, action: MyAction) -> MyObservation:  # type: ignore[override]
        self._state.step_count += 1
        notes: List[str] = []

        if action.decision is not None:
            self._episode["decision"] = action.decision.lower().strip()
            notes.append(f"decision={self._episode['decision']}")
            
        if action.question is not None and not self._episode["resolved"]:
            if self._episode["questions_asked"] < MAX_STEPS:
                self._episode["questions_asked"] += 1
                q_text = action.question.strip()
                ans_text = "I don't understand or I don't have that information."
                # naive keyword match for simulated patient
                for kw, ans in self._episode["task"]["case"]["responses"].items():
                    if kw.lower() in q_text.lower():
                        ans_text = ans
                        break
                self._episode["dialogue_history"].append(f"Agent: {q_text}")
                self._episode["dialogue_history"].append(f"Patient: {ans_text}")
                notes.append("question_asked")
            else:
                notes.append("question_rejected_budget_exceeded")

        if action.summary is not None:
            self._episode["summary"] = action.summary.strip()
            notes.append("summary_updated")
            
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
        if action.decision and action.decision.lower() not in ALLOWED_DECISIONS:
            delta -= 0.05
            invalid = True
            notes.append("invalid_decision")

        if not any([action.decision is not None, action.question is not None, action.summary is not None, action.resolve]):
            delta -= 0.03
            notes.append("empty_action")

        delta = round(max(min(delta, 1.0), -1.0), 2)
        done = bool(self._episode["resolved"] or self._state.step_count >= MAX_STEPS)

        if done and not self._episode["resolved"]:
            notes.append("episode_ended_at_step_limit")
        if done and self._episode["resolved"] and not invalid:
            notes.append("triage_completed")

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
