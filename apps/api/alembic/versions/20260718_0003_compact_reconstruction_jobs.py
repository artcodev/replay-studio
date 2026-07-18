"""Add the compact authoritative reconstruction scheduler.

Revision ID: 20260718_0003
Revises: 20260717_0002
"""

from __future__ import annotations

import time

from alembic import op
import sqlalchemy as sa


revision = "20260718_0003"
down_revision = "20260717_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "reconstruction_jobs" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "reconstruction_jobs",
        sa.Column(
            "scene_id",
            sa.String(120),
            sa.ForeignKey("scenes.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("run_id", sa.String(120), nullable=False),
        sa.Column("input_fingerprint", sa.String(96), nullable=False),
        sa.Column("input_revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("requested_at", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )
    op.create_index(
        "ix_reconstruction_jobs_status",
        "reconstruction_jobs",
        ["status"],
    )

    # One-time cutover: seed only explicit current-run tokens. This migration
    # is the removal condition for the dense Scene scheduler; runtime code has
    # no fallback/backfill path and never discovers work from Scene JSON.
    scenes = sa.table(
        "scenes",
        sa.column("id", sa.String(120)),
        sa.column("payload", sa.JSON()),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    jobs = sa.table(
        "reconstruction_jobs",
        sa.column("scene_id", sa.String(120)),
        sa.column("run_id", sa.String(120)),
        sa.column("input_fingerprint", sa.String(96)),
        sa.column("input_revision", sa.Integer()),
        sa.column("status", sa.String(40)),
        sa.column("requested_at", sa.Float()),
        sa.column("updated_at", sa.Float()),
    )
    now = time.time()
    values: list[dict[str, object]] = []
    for row in op.get_bind().execute(
        sa.select(scenes.c.id, scenes.c.payload, scenes.c.updated_at)
    ).mappings():
        scene = row["payload"] if isinstance(row["payload"], dict) else {}
        video = scene.get("payload", {}).get("videoAsset", {})
        if video.get("multiPass"):
            continue
        reconstruction = video.get("reconstruction", {})
        status = str(reconstruction.get("status") or "")
        run_id = str(reconstruction.get("runId") or "")
        fingerprint = str(reconstruction.get("inputFingerprint") or "")
        if (
            status not in {"queued", "processing", "ready", "failed", "cancelled"}
            or not run_id
            or not fingerprint
        ):
            continue
        try:
            input_revision = max(
                1,
                int(reconstruction.get("runRevision") or 1),
            )
        except (TypeError, ValueError):
            input_revision = 1
        updated = row["updated_at"]
        timestamp = (
            float(updated.timestamp())
            if updated is not None and hasattr(updated, "timestamp")
            else now
        )
        values.append(
            {
                "scene_id": str(row["id"]),
                "run_id": run_id,
                "input_fingerprint": fingerprint,
                "input_revision": input_revision,
                "status": status,
                "requested_at": timestamp,
                "updated_at": timestamp,
            }
        )
    if values:
        op.bulk_insert(jobs, values)


def downgrade() -> None:
    if "reconstruction_jobs" in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table("reconstruction_jobs")
