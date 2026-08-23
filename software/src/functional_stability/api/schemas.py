"""Request and response schemas for the public API."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EdgeInput(BaseModel):
    source: int = Field(ge=0)
    target: int = Field(ge=0)
    reliability: float = Field(ge=0.0, le=1.0)


class AnalysisCreate(BaseModel):
    sample_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._-]+$")
    node_count: int = Field(ge=1, le=1000)
    edges: list[EdgeInput] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def validate_edges(self) -> "AnalysisCreate":
        seen: set[tuple[int, int]] = set()
        for edge in self.edges:
            if edge.source == edge.target:
                raise ValueError("self-loops are not supported")
            if edge.source >= self.node_count or edge.target >= self.node_count:
                raise ValueError("edge endpoint is outside node_count")
            key = tuple(sorted((edge.source, edge.target)))
            if key in seen:
                raise ValueError(f"duplicate edge {key}")
            seen.add(key)
        return self


class AnalysisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    sample_id: str
    node_count: int
    edge_count: int
    status: str
    reliability: float | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None


class HealthRead(BaseModel):
    status: str
    environment: str
