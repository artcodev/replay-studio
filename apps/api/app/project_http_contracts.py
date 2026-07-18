from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from .project_contract_base import ProjectContract


class MatchCandidate(ProjectContract):
    id: str
    name: str
    date: str | None = None
    time: str | None = None
    status: str | None = None
    competition: str | None = None
    season: str | None = None
    home_team: dict[str, Any]
    away_team: dict[str, Any]
    score: dict[str, int | None]
    thumbnail: str | None = None


class MatchSelection(ProjectContract):
    match_id: str = Field(min_length=1, max_length=160)


class ProjectCompositionRequest(ProjectContract):
    segment_ids: list[str] = Field(min_length=2, max_length=12)
    title: str | None = Field(default=None, max_length=240)
    manual_alignment_anchors: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("segment_ids")
    @classmethod
    def distinct_segment_ids(cls, values: list[str]) -> list[str]:
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("segment ids must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("choose different project segments")
        return normalized


class ProjectPersonMembershipAssignment(ProjectContract):
    scene_id: str = Field(min_length=1, max_length=120)
    scene_person_id: str = Field(min_length=1, max_length=160)


class PublicProject(ProjectContract):
    id: str
    title: str
    revision: int
    match_id: str | None = None
    active_segment_id: str | None = None
    created_at: str
    updated_at: str


class PublicMatchTeam(ProjectContract):
    id: str
    name: str
    short_name: str | None = None
    badge_url: str | None = None


class PublicRosterPlayer(ProjectContract):
    id: str
    team_id: str
    name: str
    number: str | None = None
    position: str | None = None
    role: Literal["starter", "substitute", "squad", "unknown"] = "unknown"
    goalkeeper: bool = False


class PublicMatchEvent(ProjectContract):
    id: str
    kind: str
    minute: int | None = None
    added_time: int | None = None
    team_id: str | None = None
    player_id: str | None = None
    secondary_player_id: str | None = None
    label: str
    detail: str | None = None


class PublicSubstitution(ProjectContract):
    id: str
    team_id: str | None = None
    minute: int | None = None
    added_time: int | None = None
    player_out_id: str | None = None
    player_in_id: str | None = None
    label: str | None = None


class PublicMatchSync(ProjectContract):
    state: Literal["not-configured", "manual", "syncing", "synced", "failed"]
    synced_at: str | None = None
    stale: bool = False
    warnings: list[str] = Field(default_factory=list)


class PublicCanonicalMatch(ProjectContract):
    id: str
    revision: int
    snapshot_id: str
    snapshot_hash: str
    name: str | None = None
    competition: str | None = None
    season: str | None = None
    kickoff_at: str | None = None
    status: str | None = None
    score: dict[str, int | None]
    home_team: PublicMatchTeam
    away_team: PublicMatchTeam
    roster: list[PublicRosterPlayer] = Field(default_factory=list)
    events: list[PublicMatchEvent] = Field(default_factory=list)
    substitutions: list[PublicSubstitution] = Field(default_factory=list)
    sync: PublicMatchSync


class PublicProjectAsset(ProjectContract):
    id: str
    project_id: str
    timeline_scene_id: str | None = None
    filename: str
    duration: float | None = None
    status: Literal["uploading", "processing", "ready", "failed", "cancelled"]
    media_url: str | None = None
    poster_url: str | None = None
    created_at: str


class PublicProjectSegment(ProjectContract):
    id: str
    project_id: str
    asset_id: str
    source_segment_id: str
    scene_id: str | None = None
    label: str
    start: float
    end: float
    status: Literal["pending", "ready", "analyzing", "failed"]


class PublicAnalysisProgress(ProjectContract):
    completed: int = 0
    total: int = 0
    percent: int = 0
    label: str = ""
    detail: str | None = None
    eta_seconds: float | None = None


class PublicAnalysisJob(ProjectContract):
    id: str
    project_id: str
    segment_id: str | None = None
    kind: str
    status: Literal[
        "queued", "running", "cancelling", "cancelled", "succeeded", "failed"
    ]
    phase: str | None = None
    progress: PublicAnalysisProgress
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
