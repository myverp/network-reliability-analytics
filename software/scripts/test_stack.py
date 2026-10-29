"""Run isolated Compose integration checks; never uses the normal Compose project."""

import subprocess
import uuid

project = "nr-test-" + uuid.uuid4().hex[:12]
base = ["docker", "compose", "-f", "compose.test.yml", "-p", project]


def compose(*args, capture=False):
    return subprocess.run(base + list(args), check=True, text=True, stdout=subprocess.PIPE if capture else None).stdout


def check(*args):
    return compose("exec", "-T", "api", "python", "software/scripts/check_stack.py", *args, capture=True).strip()


try:
    compose("up", "--build", "--wait", "--wait-timeout", "180")
    # Startup runs Alembic on a fresh PostgreSQL database. Also check repeatability and model drift.
    compose("exec", "-T", "api", "alembic", "upgrade", "head")
    compose("exec", "-T", "api", "alembic", "check")
    check("success")
    check("worker-failure")
    compose("stop", "redis")
    analysis_id = check("outage")
    compose("start", "redis")
    compose("up", "--wait", "--wait-timeout", "60", "redis")
    compose("restart", "worker")
    check("retry", analysis_id)
    print("PASS: migrations, HTTP/PostgreSQL/Redis/RQ, worker failure, Redis outage and retry")
except BaseException:
    compose("logs", "--no-color", "--tail", "100")
    raise
finally:
    compose("down", "--volumes", "--remove-orphans")
