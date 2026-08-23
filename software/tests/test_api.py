from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

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
