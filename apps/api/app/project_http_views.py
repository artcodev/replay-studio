from __future__ import annotations

from datetime import datetime
from typing import Any

from .project_http_contracts import (
    PublicAnalysisJob,
    PublicAnalysisProgress,
    PublicCanonicalMatch,
    PublicMatchEvent,
    PublicMatchSync,
    PublicMatchTeam,
    PublicProject,
    PublicRosterPlayer,
    PublicSubstitution,
)
from .project_identifiers import stable_identifier
from .project_lifecycle_contract import ProjectHeader, ProjectSummary
from .project_match_persistence_contract import (
    MatchSnapshotDocument,
    MatchSnapshotSummary,
)


def iso_string(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def public_project(project: ProjectSummary) -> PublicProject:
    return PublicProject(
        id=project.id,
        title=project.title,
        revision=project.revision,
        match_id=project.match_id,
        active_segment_id=(project.metadata or {}).get("activeSegmentId"),
        created_at=iso_string(project.created_at),
        updated_at=iso_string(project.updated_at),
    )


def canonical_match_view(
    project: ProjectHeader,
    canonical: dict[str, Any] | None,
    snapshot: MatchSnapshotSummary | MatchSnapshotDocument | None,
    *,
    manual: bool = False,
) -> PublicCanonicalMatch | None:
    if not canonical or snapshot is None:
        return None
    home = canonical.get("homeTeam") or {}
    away = canonical.get("awayTeam") or {}
    score = canonical.get("score") or {}
    sync = canonical.get("sync") or {}
    kickoff = " ".join(
        str(value)
        for value in (canonical.get("date"), canonical.get("time"))
        if value
    ) or None

    roster = [
        _roster_player(item)
        for item in canonical.get("roster") or []
        if isinstance(item, dict) and item.get("id") and item.get("teamId")
    ]
    events = [
        PublicMatchEvent(
            id=str(item.get("id") or stable_identifier("event", project.id, index)),
            kind=str(item.get("type") or item.get("kind") or "other"),
            minute=item.get("minute"),
            added_time=item.get("addedTime"),
            team_id=item.get("teamId"),
            player_id=item.get("playerId"),
            secondary_player_id=item.get("secondaryPlayerId"),
            label=str(item.get("label") or "Match event"),
            detail=item.get("detail"),
        )
        for index, item in enumerate(canonical.get("events") or [])
        if isinstance(item, dict)
    ]
    substitutions = [
        PublicSubstitution(
            id=str(item.get("id") or stable_identifier("sub", project.id, index)),
            team_id=item.get("teamId"),
            minute=item.get("minute"),
            added_time=item.get("addedTime"),
            player_out_id=item.get("playerOutId"),
            player_in_id=item.get("playerInId"),
            label=item.get("label"),
        )
        for index, item in enumerate(canonical.get("substitutions") or [])
        if isinstance(item, dict)
    ]
    return PublicCanonicalMatch(
        id=str(canonical.get("id") or project.match_id),
        revision=project.revision,
        snapshot_id=str(snapshot.id),
        snapshot_hash=str(snapshot.content_hash),
        name=canonical.get("name"),
        competition=canonical.get("competition"),
        season=canonical.get("season"),
        kickoff_at=kickoff,
        status=canonical.get("status") or "unknown",
        score={"home": score.get("home"), "away": score.get("away")},
        home_team=_match_team(home, project.id, "home"),
        away_team=_match_team(away, project.id, "away"),
        roster=roster,
        events=events,
        substitutions=substitutions,
        sync=PublicMatchSync(
            state="manual" if manual else "synced",
            synced_at=sync.get("syncedAt"),
            stale=bool(sync.get("stale")),
            warnings=[str(value) for value in sync.get("warnings") or []],
        ),
    )


def public_analysis_job(run: Any) -> PublicAnalysisJob:
    progress = run.progress or {}
    return PublicAnalysisJob(
        id=run.id,
        project_id=run.project_id,
        segment_id=run.segment_id,
        kind=run.kind,
        status=run.status,
        phase=progress.get("phase"),
        progress=PublicAnalysisProgress(
            completed=int(progress.get("completed") or 0),
            total=int(progress.get("total") or 0),
            percent=max(0, min(100, int(progress.get("overallPercent") or 0))),
            label=str(progress.get("label") or ""),
            detail=progress.get("detail"),
            eta_seconds=progress.get("etaSeconds"),
        ),
        created_at=iso_string(run.created_at),
        started_at=iso_string(run.started_at) or None,
        finished_at=iso_string(run.completed_at) or None,
        error=run.error,
    )


def _roster_player(item: dict[str, Any]) -> PublicRosterPlayer:
    role = str(item.get("lineupRole") or item.get("role") or "unknown")
    if role not in {"starter", "substitute", "squad", "unknown"}:
        role = "unknown"
    position = str(item.get("position") or "") or None
    return PublicRosterPlayer(
        id=str(item["id"]),
        team_id=str(item["teamId"]),
        name=str(item.get("name") or "Unknown player"),
        number=str(item["number"]) if item.get("number") is not None else None,
        position=position,
        role=role,
        goalkeeper=_is_goalkeeper_position(position),
    )


def _match_team(source: dict, project_id: str, side: str) -> PublicMatchTeam:
    return PublicMatchTeam(
        id=str(source.get("id") or stable_identifier("team", project_id, side)),
        name=str(source.get("name") or side.title()),
        short_name=source.get("shortName"),
        badge_url=source.get("badgeUrl") or source.get("badge"),
    )


def _is_goalkeeper_position(value: object) -> bool:
    position = str(value or "").strip().lower().replace(".", "")
    return position in {"g", "gk", "goalie", "goalkeeper", "keeper"} or (
        "goalkeeper" in position
    )
