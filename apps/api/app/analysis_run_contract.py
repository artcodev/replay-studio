"""Persisted analysis-run telemetry contracts."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from .project_contract_base import ProjectContract


AnalysisRunStatus = Literal[
    "queued", "running", "cancelling", "cancelled", "succeeded", "failed"
]


class AnalysisRunCreate(ProjectContract):
    id: str | None = Field(default=None, max_length=160)
    scene_id: str | None = Field(default=None, max_length=120)
    segment_id: str | None = Field(default=None, max_length=160)
    kind: str = Field(min_length=1, max_length=60)
    status: AnalysisRunStatus = "queued"
    source_run_id: str | None = Field(default=None, max_length=160)
    input_fingerprint: str | None = Field(default=None, max_length=120)
    model: str | None = Field(default=None, max_length=240)
    progress: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    requested_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class AnalysisRunDocument(AnalysisRunCreate):
    id: str
    project_id: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AnalysisRunUpdate(ProjectContract):
    status: AnalysisRunStatus | None = None
    progress: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
