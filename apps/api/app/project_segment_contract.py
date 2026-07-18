"""Video-scene ownership links and normalized segment contracts."""

from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from .project_contract_base import ProjectContract


class ProjectSceneLink(ProjectContract):
    scene_id: str
    role: str
    created_at: datetime | None = None


class ProjectVideoAssetLink(ProjectContract):
    video_asset_id: str
    role: str
    created_at: datetime | None = None


class SegmentUpsert(ProjectContract):
    id: str | None = Field(default=None, max_length=160)
    video_asset_id: str | None = Field(default=None, max_length=120)
    scene_id: str | None = Field(default=None, max_length=120)
    source_segment_id: str = Field(min_length=1, max_length=160)
    label: str | None = Field(default=None, max_length=240)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    ordinal: int = Field(default=0, ge=0)
    replay_group: str | None = Field(default=None, max_length=80)
    replay_variant: str | None = Field(default=None, max_length=40)
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_range(self) -> "SegmentUpsert":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class SegmentDocument(SegmentUpsert):
    id: str
    project_id: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
