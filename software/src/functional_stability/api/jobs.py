"""Queue submission and worker-safe execution of reliability calculations."""

import logging
from datetime import UTC, datetime
from uuid import UUID

from redis import Redis
from rq import Queue
from sqlalchemy import select

from ..graph import Edge, Graph
from ..reliability import all_terminal_reliability
from .config import get_settings
from .database import SessionLocal
from .models import Analysis

logger = logging.getLogger(__name__)


def run_analysis(analysis_id: str) -> None:
    """Execute one analysis; called by RQ or inline in development."""
    with SessionLocal() as session:
        analysis = session.scalar(select(Analysis).where(Analysis.id == UUID(analysis_id)))
        if analysis is None:
            logger.warning("analysis job received an unknown id", extra={"analysis_id": analysis_id})
            return
        analysis.status = "running"
        session.commit()
        try:
            graph = Graph(
                analysis.node_count,
                tuple(Edge(**edge) for edge in analysis.edges),
            )
            settings = get_settings()
            if len(graph.edges) > settings.max_exact_edges:
                raise ValueError(
                    f"exact analysis supports at most {settings.max_exact_edges} edges; received {len(graph.edges)}"
                )
            analysis.reliability = all_terminal_reliability(graph)
            analysis.status = "completed"
            analysis.completed_at = datetime.now(UTC)
            logger.info("analysis completed", extra={"analysis_id": analysis_id})
        except Exception as exc:  # stored as a safe status for a submitted job
            analysis.status = "failed"
            analysis.error = str(exc)
            analysis.completed_at = datetime.now(UTC)
            logger.exception("analysis failed", extra={"analysis_id": analysis_id})
        session.commit()


def enqueue_analysis(analysis_id: UUID) -> None:
    settings = get_settings()
    if settings.use_inline_jobs:
        run_analysis(str(analysis_id))
        return
    if not settings.redis_url:
        raise RuntimeError("NR_REDIS_URL must be set when NR_USE_INLINE_JOBS=false")
    queue = Queue(connection=Redis.from_url(settings.redis_url))
    queue.enqueue(run_analysis, str(analysis_id), job_timeout="10m")
