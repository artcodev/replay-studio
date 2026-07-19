from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


class SceneRow(Base):
    __tablename__ = "scenes"
    __table_args__ = (
        Index(
            "ix_scenes_parent_segment",
            "parent_scene_id",
            "selected_segment_id",
            "kind",
        ),
        Index("ix_scenes_updated_at", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    title: Mapped[str] = mapped_column(String(240), nullable=False)
    duration: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )
    kind: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="demo",
        server_default="demo",
        index=True,
    )
    parent_scene_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    selected_segment_id: Mapped[str | None] = mapped_column(
        String(160),
        nullable=True,
    )
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ReconstructionJobRow(Base):
    """Compact, authoritative control record for the current scene run.

    Dense reconstruction output belongs in artifacts/the scene read model, not
    in the scheduler.  Keeping exactly one current row per scene lets idle
    discovery, fencing and cancellation operate without deserializing the
    multi-megabyte ``SceneRow.payload`` document.  Historical run telemetry
    remains in ``AnalysisRunRow`` and is deliberately not a scheduling source.
    """

    __tablename__ = "reconstruction_jobs"

    scene_id: Mapped[str] = mapped_column(
        ForeignKey("scenes.id", ondelete="CASCADE"), primary_key=True
    )
    run_id: Mapped[str] = mapped_column(String(120), nullable=False)
    input_fingerprint: Mapped[str] = mapped_column(String(96), nullable=False)
    input_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    # Attempt accounting bounds a deterministically crashing child and keeps
    # the last failure text on the authoritative record (see PipelineJobRow).
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    requested_at: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)


class ReconstructionLeaseRow(Base):
    """Process-independent fencing lease for one scene reconstruction run.

    The lease deliberately lives outside ``SceneRow.payload``. Heartbeats can
    therefore renew an active worker without changing the scene document
    revision underneath that same worker. ``SceneRepository`` still mirrors the
    current lease into response documents as runtime metadata.
    """

    __tablename__ = "reconstruction_leases"

    scene_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(120), nullable=False)
    input_fingerprint: Mapped[str] = mapped_column(String(96), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(120), nullable=False)
    acquired_at: Mapped[float] = mapped_column(Float, nullable=False)
    heartbeat_at: Mapped[float] = mapped_column(Float, nullable=False)
    expires_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)


class PipelineJobRow(Base):
    """Compact authoritative queue for non-reconstruction pipeline work.

    ``AnalysisRunRow`` is deliberately not referenced by the scheduler.  It is
    a telemetry/history sink and may be recreated without changing what work
    is runnable.  Job state contains only orchestration metadata (for example
    child scene ids and dependency statuses), never scene or frame payloads.
    """

    __tablename__ = "pipeline_jobs"
    __table_args__ = (
        UniqueConstraint(
            "kind", "subject_id", name="uq_pipeline_job_kind_subject"
        ),
        Index("ix_pipeline_jobs_status_available", "status", "available_at"),
    )

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    subject_id: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    state: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    available_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    requested_at: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)


class PipelineJobLeaseRow(Base):
    """Fencing lease for one generic pipeline job claim."""

    __tablename__ = "pipeline_job_leases"

    job_id: Mapped[str] = mapped_column(
        ForeignKey("pipeline_jobs.id", ondelete="CASCADE"), primary_key=True
    )
    owner_id: Mapped[str] = mapped_column(String(120), nullable=False)
    token: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    acquired_at: Mapped[float] = mapped_column(Float, nullable=False)
    heartbeat_at: Mapped[float] = mapped_column(Float, nullable=False)
    expires_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)


class VideoAssetRow(Base):
    __tablename__ = "video_assets"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    filename: Mapped[str] = mapped_column(String(240), nullable=False)
    original_name: Mapped[str] = mapped_column(String(240), nullable=False)
    content_type: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="queued")
    stage: Mapped[str] = mapped_column(String(80), nullable=False, default="queued")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Immutable derived media generation selected by the fenced pipeline
    # publication transaction.  The source upload remains at the asset root.
    generation_key: Mapped[str | None] = mapped_column(String(120), nullable=True)
    scene_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_database() -> None:
    """Bring the configured database to the current versioned schema.

    The first migration can adopt the pre-Alembic ``create_all`` development
    schema in place, so startup has only one schema-management path. Importing here
    avoids a module cycle while Alembic loads ``Base.metadata`` from its env.
    """

    from .schema_migrations import upgrade_database

    upgrade_database()
