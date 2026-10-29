"""HTTP assertions run inside the disposable API container, against a real RQ worker."""

import json
import math
import sys
import time
from urllib.request import Request, urlopen


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    with urlopen(
        Request("http://api:8000" + path, data=data, headers={"Content-Type": "application/json"}), timeout=20
    ) as response:
        assert response.status == (202 if data is not None else 200)
        return json.load(response)


def wait_for(analysis_id, status):
    for _ in range(100):
        result = request(f"/api/v1/analyses/{analysis_id}")
        if result["status"] == status:
            return result
        if result["status"] not in ("queued", "running"):
            raise AssertionError(result)
        time.sleep(0.2)
    raise AssertionError(f"Timed out waiting for {status}: {result}")


triangle = {
    "sample_id": "integration-triangle",
    "node_count": 3,
    "edges": [
        {"source": 0, "target": 1, "reliability": 0.8},
        {"source": 1, "target": 2, "reliability": 0.8},
        {"source": 0, "target": 2, "reliability": 0.8},
    ],
}
mode = sys.argv[1]
if mode == "success":
    result = request("/api/v1/analyses", triangle)
    result = wait_for(result["id"], "completed")
    assert math.isclose(result["reliability"], 0.896)
    print(result["id"])
elif mode == "outage":
    result = request("/api/v1/analyses", triangle)
    assert result["status"] == "submission_failed", result
    assert request(f"/api/v1/analyses/{result['id']}")["status"] == "submission_failed"
    print(result["id"])
elif mode == "retry":
    analysis_id = sys.argv[2]
    result = request(f"/api/v1/analyses/{analysis_id}/retry", {})
    assert result["id"] == analysis_id
    result = wait_for(analysis_id, "completed")
    assert math.isclose(result["reliability"], 0.896)
    assert result["error"] is None
elif mode == "worker-failure":
    result = request(
        "/api/v1/analyses",
        {
            "sample_id": "too-many-edges",
            "node_count": 22,
            "edges": [{"source": i, "target": i + 1, "reliability": 0.9} for i in range(21)],
        },
    )
    result = wait_for(result["id"], "failed")
    assert "at most 20 edges" in result["error"]
    assert result["completed_at"] is not None
else:
    raise ValueError(mode)
