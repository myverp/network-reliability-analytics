"""FastAPI application exposing the existing analysis core as a small service."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .config import get_settings
from .database import get_session
from .errors import NotFoundError
from .jobs import enqueue_analysis
from .logging import configure_logging
from .models import Analysis
from .schemas import AnalysisCreate, AnalysisRead, HealthRead

logger = logging.getLogger(__name__)


def _to_read(analysis: Analysis) -> AnalysisRead:
    return AnalysisRead(
        id=analysis.id,
        sample_id=analysis.sample_id,
        node_count=analysis.node_count,
        edge_count=len(analysis.edges),
        status=analysis.status,
        reliability=analysis.reliability,
        error=analysis.error,
        created_at=analysis.created_at,
        completed_at=analysis.completed_at,
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    yield


app = FastAPI(
    title="Network Reliability Analytics",
    version="0.1.0",
    description="Submit small network topologies and calculate exact all-terminal reliability.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            {"error": {"code": "validation_error", "message": "Invalid request", "details": exc.errors()}}
        ),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled application error")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "Unexpected server error"}},
    )


@app.get("/health", response_model=HealthRead, tags=["operations"])
def health(session: Session = Depends(get_session)) -> HealthRead:
    session.execute(text("SELECT 1"))
    settings = get_settings()
    return HealthRead(status="ok", environment=settings.environment)


@app.post("/api/v1/analyses", response_model=AnalysisRead, status_code=202, tags=["analyses"])
def create_analysis(payload: AnalysisCreate, session: Session = Depends(get_session)) -> AnalysisRead:
    analysis = Analysis(
        sample_id=payload.sample_id,
        node_count=payload.node_count,
        edges=[edge.model_dump() for edge in payload.edges],
    )
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
    enqueue_analysis(analysis.id)
    session.refresh(analysis)
    return _to_read(analysis)


@app.get("/api/v1/analyses", response_model=list[AnalysisRead], tags=["analyses"])
def list_analyses(limit: int = 20, session: Session = Depends(get_session)) -> list[AnalysisRead]:
    safe_limit = min(max(limit, 1), 100)
    analyses = session.scalars(select(Analysis).order_by(Analysis.created_at.desc()).limit(safe_limit)).all()
    return [_to_read(analysis) for analysis in analyses]


@app.get("/api/v1/analyses/{analysis_id}", response_model=AnalysisRead, tags=["analyses"])
def get_analysis(analysis_id: UUID, session: Session = Depends(get_session)) -> AnalysisRead:
    analysis = session.get(Analysis, analysis_id)
    if analysis is None:
        raise NotFoundError("analysis", str(analysis_id))
    return _to_read(analysis)
