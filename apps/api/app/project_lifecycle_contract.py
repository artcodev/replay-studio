"""Project-header lifecycle contracts."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from .project_contract_base import ProjectContract


class ProjectCreate(ProjectContract):
    id: str | None = Field(default=None, min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=240)
    status: str = Field(default="active", min_length=1, max_length=40)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectUpdate(ProjectContract):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    status: Literal["active", "archived"] | None = None
    metadata: dict[str, Any] | None = None
    expected_revision: int | None = Field(default=None, ge=1)


class ProjectSummary(ProjectContract):
    id: str
    title: str
    status: str
    revision: int
    match_id: str | None = None
    current_match_snapshot_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ProjectHeader(ProjectSummary):
    """Compact project state; Match content is queried through its repository."""
