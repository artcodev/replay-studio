"""Scene document and media transport contracts."""

from typing import Any, Literal

from pydantic import Field

from .transport_contract import TransportContract


class SceneDocument(TransportContract):
    id: str
    title: str
    version: int = 1
    revision: int = Field(default=0, ge=0)
    # Capability services validate timestamps against this concrete duration;
    # transport models do not impose a hidden global highlight length.
    duration: float = Field(gt=0, allow_inf_nan=False)
    payload: dict[str, Any]


class SceneSummary(TransportContract):
    id: str
    title: str
    duration: float
    kind: Literal["video", "segment", "multi-pass", "demo"]
    parent_scene_id: str | None = None
    updated_at: str | None = None


class CreateSceneRequest(TransportContract):
    event_id: str | None = None
    title: str | None = None
    provider: str | None = Field(default=None, min_length=1, max_length=80)


class VideoAsset(TransportContract):
    id: str
    filename: str
    original_name: str
    content_type: str
    status: Literal["queued", "processing", "ready", "failed", "cancelled"]
    stage: str
    progress: int = Field(ge=0, le=100)
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int = 0
    generation_key: str | None = None
    scene_id: str | None = None
    media_url: str | None = None
    poster_url: str | None = None
    error: str | None = None
    created_at: str | None = None
