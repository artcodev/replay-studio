"""Human-authored player-action transport contracts."""

from typing import Literal

from pydantic import ConfigDict, Field

from .transport_contract import TransportContract


PlayerActionType = Literal[
    "idle",
    "walk",
    "run",
    "sprint",
    "turn",
    "jump",
    "fall",
    "get-up",
    "first-touch",
    "drive",
    "pass",
    "cross",
    "shot",
    "header",
    "throw-in",
    "clearance",
    "tackle",
    "slide-tackle",
    "block",
    "interception",
    "feint",
]
PlayerActionKeypointKind = Literal[
    "wind-up",
    "contact",
    "release",
    "apex",
    "impact",
    "recovery",
]


class PlayerActionKeypointRequest(TransportContract):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: PlayerActionKeypointKind
    time: float = Field(ge=0, allow_inf_nan=False)


class PlayerActionUpsertRequest(TransportContract):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    id: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    canonical_person_id: str = Field(
        alias="canonicalPersonId",
        min_length=1,
        max_length=160,
    )
    type: PlayerActionType
    start_time: float = Field(alias="startTime", ge=0, allow_inf_nan=False)
    end_time: float = Field(alias="endTime", gt=0, allow_inf_nan=False)
    keypoints: list[PlayerActionKeypointRequest] = Field(default_factory=list, max_length=24)
