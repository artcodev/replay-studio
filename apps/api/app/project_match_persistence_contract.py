"""Normalized Match, snapshot, and external-reference persistence contracts."""

from datetime import datetime
from typing import Any

from pydantic import Field

from .project_contract_base import ProjectContract


class MatchUpsert(ProjectContract):
    id: str = Field(min_length=1, max_length=120)
    sport: str = Field(default="football", min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=240)
    competition: str | None = Field(default=None, max_length=240)
    season: str | None = Field(default=None, max_length=80)
    kickoff_at: str | None = Field(default=None, max_length=80)
    status: str | None = Field(default=None, max_length=80)
    home_team_name: str | None = Field(default=None, max_length=240)
    away_team_name: str | None = Field(default=None, max_length=240)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatchDocument(MatchUpsert):
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MatchSnapshotCreate(ProjectContract):
    provider: str = Field(min_length=1, max_length=80)
    external_event_id: str | None = Field(default=None, max_length=160)
    schema_version: int = Field(default=1, ge=1)
    fetched_at: str | None = Field(default=None, max_length=80)
    payload: dict[str, Any]


class MatchSnapshotDocument(MatchSnapshotCreate):
    id: str
    project_id: str
    match_id: str | None = None
    content_hash: str
    is_current: bool
    created_at: datetime | None = None


class MatchSnapshotSummary(ProjectContract):
    """Provider-neutral snapshot metadata safe for normal project responses."""

    id: str
    project_id: str
    match_id: str | None = None
    schema_version: int
    fetched_at: str | None = None
    content_hash: str
    is_current: bool
    created_at: datetime | None = None


class ExternalReferenceCreate(ProjectContract):
    resource_type: str = Field(min_length=1, max_length=60)
    resource_id: str = Field(min_length=1, max_length=200)
    provider: str = Field(min_length=1, max_length=80)
    external_type: str = Field(min_length=1, max_length=80)
    external_id: str = Field(min_length=1, max_length=240)
    payload: dict[str, Any] = Field(default_factory=dict)


class ExternalReferenceDocument(ExternalReferenceCreate):
    id: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class IntegrationDiagnostics(ProjectContract):
    """Explicit provenance surface, never nested in normal project responses."""

    project_id: str
    current_match_snapshot: MatchSnapshotDocument | None = None
    external_references: list[ExternalReferenceDocument] = Field(default_factory=list)
