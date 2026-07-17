from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


class SceneRow(Base):
    __tablename__ = "scenes"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    title: Mapped[str] = mapped_column(String(240), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ReconstructionLeaseRow(Base):
    """Process-independent fencing lease for one scene reconstruction run.

    The lease deliberately lives outside ``SceneRow.payload``. Heartbeats can
    therefore renew an active worker without changing the scene document
    revision underneath that same worker. ``SceneStore`` still mirrors the
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
    Base.metadata.create_all(bind=engine)
