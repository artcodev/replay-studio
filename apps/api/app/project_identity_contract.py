"""Project-wide canonical-person and membership contracts."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from .project_contract_base import ProjectContract


class ProjectPersonMembershipDocument(ProjectContract):
    id: str
    project_id: str
    project_person_id: str
    scene_id: str
    scene_person_id: str
    assignment_source: Literal["scene-local", "accepted-roster", "explicit"]
    identity_status: str | None = None
    identity_confidence: float | None = Field(default=None, ge=0, le=1)
    observation_count: int = Field(default=0, ge=0)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ProjectPersonDocument(ProjectContract):
    """Provider-neutral project identity returned by the public API."""

    id: str
    project_id: str
    roster_person_id: str | None = None
    display_name: str
    team_id: str | None = None
    role: str | None = None
    jersey_number: str | None = None
    status: Literal["active", "excluded"] = "active"
    identity_confidence: float | None = Field(default=None, ge=0, le=1)
    memberships: list[ProjectPersonMembershipDocument] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ProjectPersonSyncItem(ProjectContract):
    """Internal normalized input; it deliberately cannot carry provider ids."""

    scene_person_id: str = Field(min_length=1, max_length=160)
    roster_person_id: str | None = Field(default=None, max_length=160)
    display_name: str = Field(min_length=1, max_length=240)
    team_id: str | None = Field(default=None, max_length=160)
    role: str | None = Field(default=None, max_length=80)
    jersey_number: str | None = Field(default=None, max_length=40)
    status: Literal["active", "excluded"] = "active"
    identity_status: str | None = Field(default=None, max_length=40)
    identity_confidence: float | None = Field(default=None, ge=0, le=1)
    observation_count: int = Field(default=0, ge=0)


class ProjectIdentitySyncReport(ProjectContract):
    project_id: str
    scene_id: str
    people_created: int = 0
    people_updated: int = 0
    memberships_created: int = 0
    memberships_updated: int = 0
    memberships_preserved: int = 0
    unverified_roster_binding_count: int = 0
    people: list[ProjectPersonDocument] = Field(default_factory=list)
