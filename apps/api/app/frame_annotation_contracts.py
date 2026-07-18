"""Current-frame analysis and identity-correction HTTP contracts."""

from typing import Literal

from pydantic import Field

from .transport_contract import TransportContract


class FrameAnalysisRequest(TransportContract):
    scene_time: float = Field(ge=0, allow_inf_nan=False)


FrameAnnotationKind = Literal[
    "home-player",
    "away-player",
    "home-goalkeeper",
    "away-goalkeeper",
    "referee",
    "other",
    "ignore",
]
FrameIdentityAction = Literal["confirm", "exclude", "merge", "split"]
FrameIdentityScope = Literal["observation", "range", "identity"]


class FrameAnnotationBox(TransportContract):
    x: float = Field(ge=0, le=8000)
    y: float = Field(ge=0, le=8000)
    width: float = Field(ge=4, le=8000)
    height: float = Field(ge=4, le=8000)


class FramePersonAnnotationRequest(TransportContract):
    annotation_id: str | None = Field(default=None, min_length=1, max_length=80)
    scene_time: float = Field(ge=0, allow_inf_nan=False)
    bbox: FrameAnnotationBox
    kind: FrameAnnotationKind
    label: str | None = Field(default=None, max_length=120)
    external_player_id: str | None = Field(default=None, max_length=120)
    action: FrameIdentityAction
    scope: FrameIdentityScope
    merge_target_id: str | None = Field(default=None, min_length=1, max_length=120)
    # A correction may target both the stable identity and its optional
    # currently rendered track. They are distinct current-domain concepts.
    source_track_id: str | None = Field(default=None, min_length=1, max_length=120)
    canonical_person_id: str | None = Field(default=None, min_length=1, max_length=160)
    target_observation_id: str | None = Field(default=None, min_length=1, max_length=200)
    range_start: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    range_end: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class CanonicalRosterBindingRequest(TransportContract):
    """Bind a canonical identity to the selected match roster."""

    external_player_id: str | None = Field(default=None, min_length=1, max_length=120)
