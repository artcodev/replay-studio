from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class ProjectRow(Base):
    """Canonical owner of match, media, segments, scenes and analysis work."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    title: Mapped[str] = mapped_column(String(240), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="active", index=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    match_id: Mapped[str | None] = mapped_column(
        ForeignKey("matches.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # Kept as an application-owned pointer rather than a circular FK. The
    # snapshot itself has a strict FK back to this project and the store updates
    # both rows in one transaction.
    current_match_snapshot_id: Mapped[str | None] = mapped_column(
        String(160), nullable=True
    )
    metadata_payload: Mapped[dict] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class MatchRow(Base):
    """Provider-neutral logical match shared by one or more projects."""

    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    sport: Mapped[str] = mapped_column(String(40), nullable=False, default="football")
    name: Mapped[str] = mapped_column(String(240), nullable=False)
    competition: Mapped[str | None] = mapped_column(String(240), nullable=True)
    season: Mapped[str | None] = mapped_column(String(80), nullable=True)
    kickoff_at: Mapped[str | None] = mapped_column(String(80), nullable=True)
    status: Mapped[str | None] = mapped_column(String(80), nullable=True)
    home_team_name: Mapped[str | None] = mapped_column(String(240), nullable=True)
    away_team_name: Mapped[str | None] = mapped_column(String(240), nullable=True)
    metadata_payload: Mapped[dict] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class MatchSnapshotRow(Base):
    """Immutable provider response used to reproduce identity decisions."""

    __tablename__ = "match_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "project_id", "content_hash", name="uq_match_snapshot_project_content"
        ),
        Index("ix_match_snapshot_match_fetched", "match_id", "fetched_at"),
    )

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    match_id: Mapped[str | None] = mapped_column(
        ForeignKey("matches.id", ondelete="SET NULL"), nullable=True, index=True
    )
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    external_event_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    fetched_at: Mapped[str | None] = mapped_column(String(80), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ExternalReferenceRow(Base):
    """Namespaced upstream identifier for any normalized resource.

    A generic reference table avoids putting provider-specific columns on every
    domain table while keeping collisions such as player ``42`` from different
    providers unambiguous.
    """

    __tablename__ = "external_references"
    __table_args__ = (
        UniqueConstraint(
            "resource_type",
            "resource_id",
            "provider",
            "external_type",
            "external_id",
            name="uq_external_reference_resource",
        ),
        Index(
            "ix_external_reference_lookup",
            "provider",
            "external_type",
            "external_id",
        ),
    )

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    resource_type: Mapped[str] = mapped_column(String(60), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(200), nullable=False)
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    external_type: Mapped[str] = mapped_column(String(80), nullable=False)
    external_id: Mapped[str] = mapped_column(String(240), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class ProjectSceneRow(Base):
    __tablename__ = "project_scenes"
    __table_args__ = (
        UniqueConstraint("scene_id", name="uq_project_scene_owner"),
    )

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    scene_id: Mapped[str] = mapped_column(
        ForeignKey("scenes.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(40), nullable=False, default="scene")
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ProjectVideoAssetRow(Base):
    __tablename__ = "project_video_assets"
    __table_args__ = (
        UniqueConstraint("video_asset_id", name="uq_project_video_asset_owner"),
    )

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    video_asset_id: Mapped[str] = mapped_column(
        ForeignKey("video_assets.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(40), nullable=False, default="source")
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ProjectPersonRow(Base):
    """A project-wide identity shared by explicit scene memberships.

    ``roster_person_id`` is the provider-neutral player id from the canonical
    match snapshot. Raw provider ids intentionally have no column here.
    """

    __tablename__ = "project_people"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "roster_person_id",
            name="uq_project_person_roster_person",
        ),
        Index("ix_project_person_project_team", "project_id", "team_id"),
    )

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    roster_person_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    display_name: Mapped[str] = mapped_column(String(240), nullable=False)
    team_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    role: Mapped[str | None] = mapped_column(String(80), nullable=True)
    jersey_number: Mapped[str | None] = mapped_column(String(40), nullable=True)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="active")
    identity_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class ProjectPersonMembershipRow(Base):
    """Maps one scene-local canonical identity onto a project identity."""

    __tablename__ = "project_person_memberships"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "scene_id",
            "scene_person_id",
            name="uq_project_person_membership_scene_identity",
        ),
        Index(
            "ix_project_person_membership_person_scene",
            "project_person_id",
            "scene_id",
        ),
    )

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_person_id: Mapped[str] = mapped_column(
        ForeignKey("project_people.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scene_id: Mapped[str] = mapped_column(
        ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scene_person_id: Mapped[str] = mapped_column(String(160), nullable=False)
    assignment_source: Mapped[str] = mapped_column(
        String(40), nullable=False, default="scene-local"
    )
    identity_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    identity_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SegmentRow(Base):
    __tablename__ = "segments"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "video_asset_id",
            "source_segment_id",
            name="uq_segment_project_asset_source",
        ),
        Index("ix_segment_project_ordinal", "project_id", "ordinal"),
    )

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    video_asset_id: Mapped[str | None] = mapped_column(
        ForeignKey("video_assets.id", ondelete="SET NULL"), nullable=True, index=True
    )
    scene_id: Mapped[str | None] = mapped_column(
        ForeignKey("scenes.id", ondelete="SET NULL"), nullable=True, index=True
    )
    source_segment_id: Mapped[str] = mapped_column(String(160), nullable=False)
    label: Mapped[str | None] = mapped_column(String(240), nullable=True)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    replay_group: Mapped[str | None] = mapped_column(String(80), nullable=True)
    replay_variant: Mapped[str | None] = mapped_column(String(40), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class AnalysisRunRow(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "scene_id",
            "kind",
            "source_run_id",
            name="uq_analysis_run_source",
        ),
        Index("ix_analysis_run_project_status", "project_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scene_id: Mapped[str | None] = mapped_column(
        ForeignKey("scenes.id", ondelete="SET NULL"), nullable=True, index=True
    )
    segment_id: Mapped[str | None] = mapped_column(
        ForeignKey("segments.id", ondelete="SET NULL"), nullable=True, index=True
    )
    kind: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    source_run_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    input_fingerprint: Mapped[str | None] = mapped_column(String(120), nullable=True)
    model: Mapped[str | None] = mapped_column(String(240), nullable=True)
    progress: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    diagnostics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    requested_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
