from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(cls.temp_dir.name) / "api_test.db"
        os.environ["NR_DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
        os.environ["NR_USE_INLINE_JOBS"] = "true"

        from functional_stability.api.config import get_settings

        get_settings.cache_clear()
        from functional_stability.api import database

        database.engine = database.create_db_engine()
        database.SessionLocal.configure(bind=database.engine)
        from functional_stability.api.database import Base
        from functional_stability.api.main import app

        Base.metadata.create_all(bind=database.engine)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        from functional_stability.api import database

        database.engine.dispose()
        cls.temp_dir.cleanup()

    def test_queue_failure_can_be_retried_without_new_analysis(self) -> None:
        with patch("functional_stability.api.jobs.enqueue_analysis", side_effect=ConnectionError("secret")):
            response = self.client.post("/api/v1/analyses", json={"sample_id": "retry", "node_count": 1})
        self.assertEqual(response.status_code, 202)
        result = response.json()
        self.assertEqual(result["status"], "submission_failed")
        self.assertNotIn("secret", result["error"])
        retried = self.client.post(f"/api/v1/analyses/{result['id']}/retry")
        self.assertEqual(retried.status_code, 202)
        self.assertEqual(retried.json()["id"], result["id"])
        self.assertEqual(retried.json()["status"], "completed")
        self.assertIsNone(retried.json()["error"])
        self.assertEqual(self.client.post(f"/api/v1/analyses/{result['id']}/retry").status_code, 409)

    def test_worker_failure_is_persisted_and_retryable(self) -> None:
        with patch(
            "functional_stability.api.jobs.all_terminal_reliability", side_effect=ValueError("calculation failed")
        ):
            result = self.client.post("/api/v1/analyses", json={"sample_id": "worker", "node_count": 1}).json()
        self.assertEqual(result["status"], "failed")
        self.assertIsNotNone(result["completed_at"])
        self.assertEqual(self.client.post(f"/api/v1/analyses/{result['id']}/retry").json()["status"], "completed")

    def test_duplicate_worker_delivery_does_not_recompute(self) -> None:
        from functional_stability.api.jobs import run_analysis

        result = self.client.post("/api/v1/analyses", json={"sample_id": "duplicate", "node_count": 1}).json()
        with patch("functional_stability.api.jobs.all_terminal_reliability") as calculate:
            run_analysis(result["id"])
            calculate.assert_not_called()

    def test_lost_enqueue_response_preserves_completed_result(self) -> None:
        from functional_stability.api.jobs import run_analysis

        def deliver_then_fail(analysis_id):
            run_analysis(str(analysis_id))
            raise ConnectionError("lost acknowledgement")

        with patch("functional_stability.api.jobs.enqueue_analysis", side_effect=deliver_then_fail):
            result = self.client.post("/api/v1/analyses", json={"sample_id": "lost-ack", "node_count": 1}).json()
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["error"])

    def test_health(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_create_analysis_calculates_triangle_reliability(self) -> None:
        response = self.client.post(
            "/api/v1/analyses",
            json={
                "sample_id": "triangle-001",
                "node_count": 3,
                "edges": [
                    {"source": 0, "target": 1, "reliability": 0.8},
                    {"source": 1, "target": 2, "reliability": 0.8},
                    {"source": 0, "target": 2, "reliability": 0.8},
                ],
            },
        )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["status"], "completed")
        self.assertAlmostEqual(payload["reliability"], 0.896)

    def test_invalid_edge_returns_consistent_validation_error(self) -> None:
        response = self.client.post(
            "/api/v1/analyses",
            json={
                "sample_id": "bad-loop",
                "node_count": 2,
                "edges": [{"source": 0, "target": 0, "reliability": 0.9}],
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "validation_error")

    def test_missing_analysis_returns_not_found(self) -> None:
        response = self.client.get("/api/v1/analyses/00000000-0000-0000-0000-000000000000")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
