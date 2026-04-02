---
title: Support Ticket Triage Environment
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - customer-support
---

# Support Ticket Triage Environment

A real-world OpenEnv environment where an agent triages customer support tickets by setting priority, routing to the correct team, writing a response, and deciding when to resolve the ticket.

## Why this environment is useful

Support operations teams repeatedly solve this workflow in production systems. Good agents need to balance urgency, routing accuracy, customer communication quality, and safe resolution. This makes the environment practical for evaluating agent reliability rather than game-playing.

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

- `priority`: `low | medium | high | urgent`
- `team`: `billing | technical_support | account_security | logistics`
- `response`: short free-text reply to the customer
- `resolve`: whether the agent wants to close the ticket

## Observation space

`MyObservation`

- task metadata: `task_id`, `difficulty`, `objective`
- ticket details: `ticket_id`, `customer_tier`, `subject`, `body`
- current working state: `current_priority`, `current_team`, `response_sent`
- feedback: `last_feedback`
- metadata includes current normalized score and per-component breakdown

## Tasks

### 1. easy_billing_refund
Route a duplicate-charge complaint to billing, mark it high priority, send an apologetic refund-oriented reply, and resolve.

### 2. medium_account_lockout
Handle an enterprise login outage before a board meeting. The correct route is account security, with urgent priority and a reply focused on restoring access.

### 3. hard_multi_issue_vip
A VIP customer reports SSO login failures, duplicate invoices, and dashboard issues. The agent must prioritize correctly, pick the most safety-critical team, acknowledge multiple issues, and resolve only after taking the right steps.

## Reward design

The reward is dense and deterministic:

- partial credit for choosing a reasonable priority and team
- higher credit for exact matches
- response reward based on keyword coverage of required customer-facing concepts
- resolution reward only after enough progress has been made
- penalties for empty or invalid actions

The final task score is always clamped to `0.0–1.0`.

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
export MODEL_NAME="openai/gpt-oss-20b"
export HF_TOKEN="<your_token>"
python inference.py
```

## Expected baseline behavior

The provided baseline should finish in well under 20 minutes on a 2 vCPU / 8 GB machine because each task uses a tiny number of steps and lightweight JSON parsing.
