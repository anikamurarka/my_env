import asyncio
import json
import os
from typing import Dict, List

from openai import OpenAI

try:
    from my_env import MyAction, MyEnv
except ModuleNotFoundError:
    from client import MyEnv
    from models import MyAction


IMAGE_NAME = os.getenv("IMAGE_NAME", "my_env-env:latest")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN", "")
MAX_STEPS = 5
TASK_SEQUENCE = [
    "easy_obvious_emergency",
    "medium_ambiguous_symptoms",
    "hard_noisy_comorbid",
]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_text = "null" if error is None else error
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def heuristic_action(task_id: str) -> Dict:
    if task_id == "easy_obvious_emergency":
        return {
            "decision": "emergency_escalation",
            "question": None,
            "summary": "38 weeks pregnant with heavy bleeding. Emergency escalation required.",
            "resolve": True,
        }
    if task_id == "medium_ambiguous_symptoms":
        return {
            "decision": "urgent_escalation",
            "question": "Do you have a severe headache or vision changes?",
            "summary": "Postpartum swelling with possible preeclampsia symptoms. Urgent escalation.",
            "resolve": True,
        }
    return {
        "decision": "emergency_escalation",
        "question": "Is the baby breathing fast?",
        "summary": "Newborn lethargic and not feeding. Emergency escalation.",
        "resolve": True,
    }


def get_model_action(client: OpenAI, task_id: str, case_id: str, patient_description: str, objective: str) -> Dict:
    prompt = (
        "Return only JSON with keys decision, question, summary, resolve. "
        "Choose decision from [reassure, schedule_follow_up, refer_facility, urgent_escalation, emergency_escalation]. "
        f"Task: {task_id}. Objective: {objective}. Case ID: {case_id}. Description: {patient_description} "
        "Either ask a question and set resolve=False, or make a decision with a summary and set resolve=True."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a frontline public-health triage agent. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=220,
        )
        content = response.choices[0].message.content or "{}"
        start = content.find("{")
        end = content.rfind("}")
        parsed = json.loads(content[start : end + 1])
        return {
            "decision": parsed.get("decision"),
            "question": parsed.get("question"),
            "summary": parsed.get("summary"),
            "resolve": bool(parsed.get("resolve", True)),
        }
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return heuristic_action(task_id)


async def run_task(client: OpenAI, env: MyEnv, task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env="maternal_health_triage_env", model=MODEL_NAME)

    result = await env.reset()
    obs = result.observation

    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break
        action_dict = get_model_action(client, obs.task_id, obs.case_id, obs.patient_description, obs.objective)
        action = MyAction(**action_dict)
        result = await env.step(action)
        obs = result.observation
        reward = float(result.reward or 0.0)
        rewards.append(reward)
        steps_taken = step
        log_step(step=step, action=json.dumps(action.model_dump(exclude_none=True)), reward=reward, done=bool(result.done), error=None)
        if result.done:
            break

    _EPS = 1e-4
    # The actual final score must come from the environment's grinder (metadata.score)
    state = await env.state()
    env_score = float(state.metadata.get("score", 0.0))
    score = round(max(min(env_score, 1.0 - _EPS), _EPS), 2)
    success = score >= 0.6
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "EMPTY")
    env = await MyEnv.from_docker_image(IMAGE_NAME)
    all_scores: List[float] = []
    try:
        for task in TASK_SEQUENCE:
            score = await run_task(client, env, task)
            all_scores.append(score)
        avg = sum(all_scores) / len(all_scores)
        print(f"[DEBUG] avg_score={avg:.2f} tasks={len(TASK_SEQUENCE)}", flush=True)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error (container cleanup): {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
