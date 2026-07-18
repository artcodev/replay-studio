"""Replace startup-backfill match snapshots with the canonical match contract.

Revision ID: 20260718_0008
Revises: 20260718_0007
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from alembic import op
import sqlalchemy as sa


revision = "20260718_0008"
down_revision = "20260718_0007"
branch_labels = None
depends_on = None


def _payload_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def _snapshot_id(project_id: str, content_hash: str) -> str:
    material = "\x1f".join((project_id, content_hash))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return f"snapshot-{digest}"


def _canonical_payload(
    payload: object,
    *,
    fetched_at: str | None,
) -> dict[str, Any] | None:
    """Map the one retired startup-backfill shape into the live contract.

    This deliberately recognizes a narrow structural signature. It is a
    one-time data cutover, not a second runtime parser for historical shapes.
    """

    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        return None
    if any(
        key in payload
        for key in ("id", "homeTeam", "awayTeam", "roster", "events", "sync")
    ):
        return None
    event = payload.get("event")
    players = payload.get("players")
    timeline = payload.get("timeline")
    if (
        not isinstance(event, dict)
        or not isinstance(event.get("home"), dict)
        or not isinstance(event.get("away"), dict)
        or not isinstance(players, list)
        or not isinstance(timeline, list)
        or not str(payload.get("matchId") or "").strip()
    ):
        return None

    quality = payload.get("rosterQuality")
    quality = dict(quality) if isinstance(quality, dict) else {}
    quality_status = str(quality.get("status") or "")
    sync_state = (
        "ready"
        if quality_status == "automatic-ready"
        else "partial"
        if players or timeline
        else "unavailable"
    )
    warnings = payload.get("warnings")
    warnings = list(warnings) if isinstance(warnings, list) else []

    return {
        "schemaVersion": 1,
        "id": str(payload["matchId"]),
        "name": event.get("name"),
        "competition": event.get("competition"),
        "season": event.get("season"),
        "date": event.get("date"),
        "time": event.get("time"),
        "status": event.get("status"),
        "score": dict(event.get("score"))
        if isinstance(event.get("score"), dict)
        else {"home": None, "away": None},
        "homeTeam": dict(event["home"]),
        "awayTeam": dict(event["away"]),
        "roster": list(players),
        "lineup": list(payload.get("lineup"))
        if isinstance(payload.get("lineup"), list)
        else [],
        "events": list(timeline),
        "substitutions": list(payload.get("substitutions"))
        if isinstance(payload.get("substitutions"), list)
        else [],
        "rosterQuality": quality,
        "sync": {
            "state": sync_state,
            "syncedAt": payload.get("fetchedAt") or fetched_at,
            "stale": False,
            "warnings": warnings,
        },
    }


def _cutover(connection) -> None:
    tables = set(sa.inspect(connection).get_table_names())
    if not {"projects", "matches", "match_snapshots"}.issubset(tables):
        return

    projects = sa.table(
        "projects",
        sa.column("id", sa.String(120)),
        sa.column("revision", sa.Integer()),
        sa.column("match_id", sa.String(120)),
        sa.column("current_match_snapshot_id", sa.String(160)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    matches = sa.table(
        "matches",
        sa.column("id", sa.String(120)),
    )
    snapshots = sa.table(
        "match_snapshots",
        sa.column("id", sa.String(160)),
        sa.column("project_id", sa.String(120)),
        sa.column("match_id", sa.String(120)),
        sa.column("provider", sa.String(80)),
        sa.column("external_event_id", sa.String(160)),
        sa.column("schema_version", sa.Integer()),
        sa.column("fetched_at", sa.String(80)),
        sa.column("content_hash", sa.String(80)),
        sa.column("is_current", sa.Boolean()),
        sa.column("payload", sa.JSON()),
    )

    current_rows = connection.execute(
        sa.select(
            projects.c.id.label("project_id"),
            projects.c.match_id.label("project_match_id"),
            snapshots.c.id.label("snapshot_id"),
            snapshots.c.match_id.label("snapshot_match_id"),
            snapshots.c.provider,
            snapshots.c.external_event_id,
            snapshots.c.schema_version,
            snapshots.c.fetched_at,
            snapshots.c.payload,
        )
        .select_from(
            projects.join(
                snapshots,
                snapshots.c.id == projects.c.current_match_snapshot_id,
            )
        )
        .where(
            snapshots.c.provider == "canonical",
            snapshots.c.schema_version == 1,
        )
    ).mappings().all()

    for row in current_rows:
        canonical = _canonical_payload(row["payload"], fetched_at=row["fetched_at"])
        if canonical is None:
            continue

        payload_match_id = str(canonical["id"])
        selected_match_ids = {
            str(value)
            for value in (row["project_match_id"], row["snapshot_match_id"])
            if value is not None and str(value).strip()
        }
        if selected_match_ids != {payload_match_id}:
            raise RuntimeError(
                "Cannot cut over canonical match snapshot "
                f"{row['snapshot_id']} for project {row['project_id']}: "
                "project, snapshot and payload Match ids disagree"
            )
        match_exists = connection.scalar(
            sa.select(matches.c.id).where(matches.c.id == payload_match_id)
        )
        if match_exists is None:
            raise RuntimeError(
                "Cannot cut over canonical match snapshot "
                f"{row['snapshot_id']}: Match {payload_match_id} does not exist"
            )

        content_hash = _payload_hash(canonical)
        replacement_id = _snapshot_id(str(row["project_id"]), content_hash)
        replacement = connection.execute(
            sa.select(
                snapshots.c.id,
                snapshots.c.match_id,
                snapshots.c.payload,
            ).where(
                snapshots.c.project_id == row["project_id"],
                snapshots.c.content_hash == content_hash,
            )
        ).mappings().one_or_none()
        if replacement is None:
            colliding_id = connection.scalar(
                sa.select(snapshots.c.id).where(snapshots.c.id == replacement_id)
            )
            if colliding_id is not None:
                raise RuntimeError(
                    f"Canonical replacement snapshot id collision: {replacement_id}"
                )
            connection.execute(
                snapshots.insert().values(
                    id=replacement_id,
                    project_id=row["project_id"],
                    match_id=payload_match_id,
                    provider=row["provider"],
                    external_event_id=row["external_event_id"],
                    schema_version=1,
                    fetched_at=row["fetched_at"],
                    content_hash=content_hash,
                    is_current=False,
                    payload=canonical,
                )
            )
        else:
            replacement_id = str(replacement["id"])
            if (
                replacement["match_id"] != payload_match_id
                or replacement["payload"] != canonical
            ):
                raise RuntimeError(
                    "Existing canonical replacement does not match its content hash: "
                    f"{replacement_id}"
                )

        # MatchSnapshot rows are immutable. Only current-selection state and
        # the Project's two current pointers move in this transaction.
        connection.execute(
            snapshots.update()
            .where(snapshots.c.project_id == row["project_id"])
            .values(is_current=False)
        )
        connection.execute(
            snapshots.update()
            .where(snapshots.c.id == replacement_id)
            .values(is_current=True)
        )
        connection.execute(
            projects.update()
            .where(projects.c.id == row["project_id"])
            .values(
                current_match_snapshot_id=replacement_id,
                match_id=payload_match_id,
                revision=projects.c.revision + 1,
                updated_at=sa.func.now(),
            )
        )


def upgrade() -> None:
    _cutover(op.get_bind())


def downgrade() -> None:
    # The replacement is a canonical immutable snapshot and remains valid.
    # Re-selecting the retired shape would reintroduce the production bug.
    pass
