"""Baseline legacy tables and add normalized project persistence.

Revision ID: 20260717_0001
Revises: None
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260717_0001"
down_revision = None
branch_labels = None
depends_on = None


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _index(name: str, table: str, columns: list[str], *, unique: bool = False) -> None:
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table)}
    if name not in indexes:
        op.create_index(name, table, columns, unique=unique)


def upgrade() -> None:
    tables = _tables()
    timestamp = sa.text("CURRENT_TIMESTAMP")

    # This first revision can initialize a fresh database or adopt the legacy
    # create_all database in place. It never drops or rewrites existing rows.
    if "scenes" not in tables:
        op.create_table(
            "scenes",
            sa.Column("id", sa.String(120), primary_key=True),
            sa.Column("title", sa.String(240), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
        )
    if "video_assets" not in tables:
        op.create_table(
            "video_assets",
            sa.Column("id", sa.String(120), primary_key=True),
            sa.Column("filename", sa.String(240), nullable=False),
            sa.Column("original_name", sa.String(240), nullable=False),
            sa.Column("content_type", sa.String(120), nullable=False),
            sa.Column("status", sa.String(40), nullable=False),
            sa.Column("stage", sa.String(80), nullable=False),
            sa.Column("progress", sa.Integer(), nullable=False),
            sa.Column("duration", sa.Float(), nullable=True),
            sa.Column("width", sa.Integer(), nullable=True),
            sa.Column("height", sa.Integer(), nullable=True),
            sa.Column("fps", sa.Float(), nullable=True),
            sa.Column("frame_count", sa.Integer(), nullable=False),
            sa.Column("scene_id", sa.String(120), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
        )
    if "reconstruction_leases" not in tables:
        op.create_table(
            "reconstruction_leases",
            sa.Column("scene_id", sa.String(120), primary_key=True),
            sa.Column("run_id", sa.String(120), nullable=False),
            sa.Column("input_fingerprint", sa.String(96), nullable=False),
            sa.Column("owner_id", sa.String(120), nullable=False),
            sa.Column("acquired_at", sa.Float(), nullable=False),
            sa.Column("heartbeat_at", sa.Float(), nullable=False),
            sa.Column("expires_at", sa.Float(), nullable=False),
        )
        _index(
            "ix_reconstruction_leases_expires_at",
            "reconstruction_leases",
            ["expires_at"],
        )

    tables = _tables()
    if "matches" not in tables:
        op.create_table(
            "matches",
            sa.Column("id", sa.String(120), primary_key=True),
            sa.Column("sport", sa.String(40), nullable=False),
            sa.Column("name", sa.String(240), nullable=False),
            sa.Column("competition", sa.String(240), nullable=True),
            sa.Column("season", sa.String(80), nullable=True),
            sa.Column("kickoff_at", sa.String(80), nullable=True),
            sa.Column("status", sa.String(80), nullable=True),
            sa.Column("home_team_name", sa.String(240), nullable=True),
            sa.Column("away_team_name", sa.String(240), nullable=True),
            sa.Column("metadata", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
        )
    if "projects" not in tables:
        op.create_table(
            "projects",
            sa.Column("id", sa.String(120), primary_key=True),
            sa.Column("title", sa.String(240), nullable=False),
            sa.Column("status", sa.String(40), nullable=False),
            sa.Column("revision", sa.Integer(), nullable=False),
            sa.Column("match_id", sa.String(120), sa.ForeignKey("matches.id", ondelete="SET NULL"), nullable=True),
            sa.Column("current_match_snapshot_id", sa.String(160), nullable=True),
            sa.Column("metadata", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
        )
        _index("ix_projects_status", "projects", ["status"])
        _index("ix_projects_match_id", "projects", ["match_id"])
    if "match_snapshots" not in tables:
        op.create_table(
            "match_snapshots",
            sa.Column("id", sa.String(160), primary_key=True),
            sa.Column("project_id", sa.String(120), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
            sa.Column("match_id", sa.String(120), sa.ForeignKey("matches.id", ondelete="SET NULL"), nullable=True),
            sa.Column("provider", sa.String(80), nullable=False),
            sa.Column("external_event_id", sa.String(160), nullable=True),
            sa.Column("schema_version", sa.Integer(), nullable=False),
            sa.Column("fetched_at", sa.String(80), nullable=True),
            sa.Column("content_hash", sa.String(80), nullable=False),
            sa.Column("is_current", sa.Boolean(), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
            sa.UniqueConstraint("project_id", "content_hash", name="uq_match_snapshot_project_content"),
        )
        _index("ix_match_snapshots_project_id", "match_snapshots", ["project_id"])
        _index("ix_match_snapshots_match_id", "match_snapshots", ["match_id"])
        _index("ix_match_snapshot_match_fetched", "match_snapshots", ["match_id", "fetched_at"])
    if "external_references" not in tables:
        op.create_table(
            "external_references",
            sa.Column("id", sa.String(160), primary_key=True),
            sa.Column("resource_type", sa.String(60), nullable=False),
            sa.Column("resource_id", sa.String(200), nullable=False),
            sa.Column("provider", sa.String(80), nullable=False),
            sa.Column("external_type", sa.String(80), nullable=False),
            sa.Column("external_id", sa.String(240), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
            sa.UniqueConstraint("resource_type", "resource_id", "provider", "external_type", "external_id", name="uq_external_reference_resource"),
        )
        _index("ix_external_reference_lookup", "external_references", ["provider", "external_type", "external_id"])
    if "project_scenes" not in tables:
        op.create_table(
            "project_scenes",
            sa.Column("project_id", sa.String(120), sa.ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("scene_id", sa.String(120), sa.ForeignKey("scenes.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("role", sa.String(40), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
            sa.UniqueConstraint("scene_id", name="uq_project_scene_owner"),
        )
    if "project_video_assets" not in tables:
        op.create_table(
            "project_video_assets",
            sa.Column("project_id", sa.String(120), sa.ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("video_asset_id", sa.String(120), sa.ForeignKey("video_assets.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("role", sa.String(40), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
            sa.UniqueConstraint("video_asset_id", name="uq_project_video_asset_owner"),
        )
    if "segments" not in tables:
        op.create_table(
            "segments",
            sa.Column("id", sa.String(160), primary_key=True),
            sa.Column("project_id", sa.String(120), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
            sa.Column("video_asset_id", sa.String(120), sa.ForeignKey("video_assets.id", ondelete="SET NULL"), nullable=True),
            sa.Column("scene_id", sa.String(120), sa.ForeignKey("scenes.id", ondelete="SET NULL"), nullable=True),
            sa.Column("source_segment_id", sa.String(160), nullable=False),
            sa.Column("label", sa.String(240), nullable=True),
            sa.Column("start_seconds", sa.Float(), nullable=False),
            sa.Column("end_seconds", sa.Float(), nullable=False),
            sa.Column("ordinal", sa.Integer(), nullable=False),
            sa.Column("replay_group", sa.String(80), nullable=True),
            sa.Column("replay_variant", sa.String(40), nullable=True),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
            sa.UniqueConstraint("project_id", "video_asset_id", "source_segment_id", name="uq_segment_project_asset_source"),
        )
        _index("ix_segments_project_id", "segments", ["project_id"])
        _index("ix_segments_video_asset_id", "segments", ["video_asset_id"])
        _index("ix_segments_scene_id", "segments", ["scene_id"])
        _index("ix_segment_project_ordinal", "segments", ["project_id", "ordinal"])
    if "analysis_runs" not in tables:
        op.create_table(
            "analysis_runs",
            sa.Column("id", sa.String(160), primary_key=True),
            sa.Column("project_id", sa.String(120), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
            sa.Column("scene_id", sa.String(120), sa.ForeignKey("scenes.id", ondelete="SET NULL"), nullable=True),
            sa.Column("segment_id", sa.String(160), sa.ForeignKey("segments.id", ondelete="SET NULL"), nullable=True),
            sa.Column("kind", sa.String(60), nullable=False),
            sa.Column("status", sa.String(40), nullable=False),
            sa.Column("source_run_id", sa.String(160), nullable=True),
            sa.Column("input_fingerprint", sa.String(120), nullable=True),
            sa.Column("model", sa.String(240), nullable=True),
            sa.Column("progress", sa.JSON(), nullable=False),
            sa.Column("diagnostics", sa.JSON(), nullable=False),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("requested_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=timestamp, nullable=False),
            sa.UniqueConstraint("project_id", "scene_id", "kind", "source_run_id", name="uq_analysis_run_source"),
        )
        _index("ix_analysis_runs_project_id", "analysis_runs", ["project_id"])
        _index("ix_analysis_runs_scene_id", "analysis_runs", ["scene_id"])
        _index("ix_analysis_runs_segment_id", "analysis_runs", ["segment_id"])
        _index("ix_analysis_runs_kind", "analysis_runs", ["kind"])
        _index("ix_analysis_runs_status", "analysis_runs", ["status"])
        _index("ix_analysis_run_project_status", "analysis_runs", ["project_id", "status"])


def downgrade() -> None:
    # Legacy tables deliberately survive downgrade; this revision may have
    # adopted a database that predates Alembic.
    for table in (
        "analysis_runs",
        "segments",
        "project_video_assets",
        "project_scenes",
        "external_references",
        "match_snapshots",
        "projects",
        "matches",
    ):
        if table in _tables():
            op.drop_table(table)
