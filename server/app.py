from fastapi import Body, HTTPException

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with `uv sync`."
    ) from e

try:
    from ..models import MyAction, MyObservation
    from .my_env_environment import MyEnvironment
except ModuleNotFoundError:  # pragma: no cover
    from models import MyAction, MyObservation
    from server.my_env_environment import MyEnvironment


app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="support_ticket_triage_env",
    max_concurrent_envs=2,
)

_shared_env = MyEnvironment()


@app.get("/")
def root():
    return {"status": "ok", "env": "support_ticket_triage_env"}


@app.get("/tasks")
def list_tasks():
    return {"tasks": MyEnvironment.list_tasks()}


@app.post("/set_task")
def set_task(task_id: str = Body(..., embed=True)):
    try:
        _shared_env.select_task(task_id)
        return {"ok": True, "task_id": task_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/grade")
def grade_current_episode():
    return _shared_env.grade_current_episode()


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
