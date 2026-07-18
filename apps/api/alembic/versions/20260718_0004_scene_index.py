"""Add compact scene navigation and segment lookup metadata.

Revision ID: 20260718_0004
Revises: 20260718_0003
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260718_0004"
down_revision = "20260718_0003"
branch_labels = None
depends_on = None


def _column_names() -> set[str]:
    return {
        str(column["name"])
        for column in sa.inspect(op.get_bind()).get_columns("scenes")
    }


def _index(name: str, columns: list[str]) -> None:
    indexes = {
        str(item["name"])
        for item in sa.inspect(op.get_bind()).get_indexes("scenes")
    }
    if name not in indexes:
        op.create_index(name, "scenes", columns)


def _kind(scene: dict) -> str:
    payload = scene.get("payload") if isinstance(scene, dict) else None
    video = payload.get("videoAsset") if isinstance(payload, dict) else None
    if not isinstance(video, dict) or not video:
        return "demo"
    title = str(scene.get("title") or "").lower()
    filename = str(video.get("filename") or "").lower()
    if "smoke test" in title or "smoke" in filename:
        return "demo"
    if video.get("multiPass"):
        return "multi-pass"
    if video.get("parentSceneId") or video.get("selectedSegmentId"):
        return "segment"
    return "video"


def _metadata(scene: dict) -> dict[str, str | float | None]:
    try:
        duration = max(0.0, float(scene.get("duration") or 0.0))
    except (AttributeError, TypeError, ValueError):
        duration = 0.0
    payload = scene.get("payload") if isinstance(scene, dict) else None
    video = payload.get("videoAsset") if isinstance(payload, dict) else None
    video = video if isinstance(video, dict) else {}
    multi_pass = video.get("multiPass")
    multi_pass = multi_pass if isinstance(multi_pass, dict) else {}
    parent_scene_id = str(
        multi_pass.get("parentSceneId") or video.get("parentSceneId") or ""
    ).strip()
    selected_segment_id = str(video.get("selectedSegmentId") or "").strip()
    return {
        "duration": duration,
        "kind": _kind(scene),
        "parent_scene_id": parent_scene_id or None,
        "selected_segment_id": selected_segment_id or None,
    }


def upgrade() -> None:
    columns = _column_names()
    if "duration" not in columns:
        op.add_column(
            "scenes",
            sa.Column(
                "duration",
                sa.Float(),
                nullable=False,
                server_default=sa.text("0"),
            ),
        )
    if "kind" not in columns:
        op.add_column(
            "scenes",
            sa.Column(
                "kind",
                sa.String(40),
                nullable=False,
                server_default=sa.text("'demo'"),
            ),
        )
    if "parent_scene_id" not in columns:
        op.add_column(
            "scenes",
            sa.Column("parent_scene_id", sa.String(120), nullable=True),
        )
    if "selected_segment_id" not in columns:
        op.add_column(
            "scenes",
            sa.Column("selected_segment_id", sa.String(160), nullable=True),
        )

    # This is the single dense read used to cut existing databases over to
    # the compact index. Runtime code deliberately has no JSON fallback or
    # startup backfill after this versioned migration completes.
    scenes = sa.table(
        "scenes",
        sa.column("id", sa.String(120)),
        sa.column("payload", sa.JSON()),
        sa.column("duration", sa.Float()),
        sa.column("kind", sa.String(40)),
        sa.column("parent_scene_id", sa.String(120)),
        sa.column("selected_segment_id", sa.String(160)),
    )
    connection = op.get_bind()
    rows = connection.execute(
        sa.select(scenes.c.id, scenes.c.payload)
    ).mappings().all()
    for row in rows:
        scene = row["payload"] if isinstance(row["payload"], dict) else {}
        connection.execute(
            scenes.update()
            .where(scenes.c.id == str(row["id"]))
            .values(**_metadata(scene))
        )

    _index("ix_scenes_kind", ["kind"])
    _index(
        "ix_scenes_parent_segment",
        ["parent_scene_id", "selected_segment_id", "kind"],
    )
    _index("ix_scenes_updated_at", ["updated_at"])


def downgrade() -> None:
    existing_indexes = {
        str(item["name"])
        for item in sa.inspect(op.get_bind()).get_indexes("scenes")
    }
    for name in (
        "ix_scenes_updated_at",
        "ix_scenes_parent_segment",
        "ix_scenes_kind",
    ):
        if name in existing_indexes:
            op.drop_index(name, table_name="scenes")
    columns = _column_names()
    for name in (
        "selected_segment_id",
        "parent_scene_id",
        "kind",
        "duration",
    ):
        if name in columns:
            op.drop_column("scenes", name)
