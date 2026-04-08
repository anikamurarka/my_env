---
title: Maternal Health Triage Environment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - healthcare
  - triage
---

# Maternal and Newborn Risk Triage Environment

A real-world OpenEnv environment representing a frontline public-health triage tool. An agent must sequentially triage maternal and newborn cases under a fixed question budget, extracting vital symptom information, and then deciding whether to reassure, schedule a follow-up, refer to a nearest facility, or escalate for emergencies.

## Why this environment is useful

Healthcare-agent benchmarks are often limited to static medical QA. This environment implements a realistic constraint: limited question budget, noisy data, and potentially catastrophic penalties for unsafe reassurance. The focus is strictly on **safe escalation policy and structured documentation**, mimicking actual frontline health worker workflows (such as ASHA/ANM workers).

## Environment API

The environment follows the standard OpenEnv interface:

- `reset()` returns the initial observation for the next task
- `step(action)` applies the agent action and returns observation, reward, done, info
- `state()` returns episode state and grader-relevant metadata

Additional helper endpoints:

- `GET /tasks` lists all tasks
- `POST /set_task` forces the next `reset()` to use a specific task id
- `POST /grade` returns the deterministic grader output for the current episode

## Action space

`MyAction`

- `decision`: `reassure | schedule_follow_up | refer_facility | urgent_escalation | emergency_escalation`
- `question`: single follow-up question to ask the user (max 5 questions per case)
- `summary`: structured clinical handoff free-text note
- `resolve`: whether the agent is making their final decision

## Observation space

`MyObservation`

- task metadata: `task_id`, `difficulty`, `objective`
- patient details: `case_id`, `patient_description`
- interactions: `dialogue_history` (shows questions asked by the agent and answers obtained)
- constraints: `questions_remaining`
- feedback: `last_feedback`

## Tasks

### 1. easy_obvious_emergency
A 38-week pregnant patient complaining of heavy bleeding. Must be identified as life-threatening and immediately escalated without reassurance.

### 2. medium_ambiguous_symptoms
A postpartum patient with swollen feet. The agent must use the question budget to check for headaches or vision changes to assess preeclampsia risk before escalating.

### 3. hard_noisy_comorbid
A vaguely described newborn case with lethargy. Eliciting specific details like chest indrawing/fast breathing confirms the severity before referral.

## Reward design

The reward is dense and deterministic:

- Eliciting relevant missing information yields positive rewards
- Correct policy decision (identifying the correct escalation path) grants the largest reward
- Unsafe `reassure` actions on emergency cases receive severe negative penalties (-0.50 score hit)
- The structured summary text is evaluated for correct clinical keywords
- Wasting the budget on irrelevant questions damages the efficiency score

The final task score is clamped to `0.0–1.0`.

## Baseline inference

`inference.py` runs all three tasks sequentially and emits structured stdout logs using `[START]`, `[STEP]`, and `[END]` prefixes. It uses the OpenAI client and reads:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

If the model call fails, the script falls back to a deterministic heuristic so the baseline still reproduces.

## Local setup

```bash
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t my_env-env:latest -f server/Dockerfile .
docker run -p 8000:8000 my_env-env:latest
```

## Validation

```bash
openenv validate
```

## Running the baseline

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your_token>"
python inference.py
```

## Expected baseline behavior

The current baseline averages `0.57` using `Qwen2.5-72B-Instruct`, demonstrating that the sequential questioning loop is non-trivial compared to single-shot JSON completion.
