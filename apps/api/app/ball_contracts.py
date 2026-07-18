"""Manual ball-trajectory transport contracts."""

from typing import Literal

from pydantic import Field

from .transport_contract import TransportContract


BallTrajectoryMode = Literal["automatic", "manual"]


class ManualBallKeyframe(TransportContract):
    t: float = Field(ge=0, allow_inf_nan=False)
    x: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)


class BallTrajectoryRequest(TransportContract):
    mode: BallTrajectoryMode
    keyframes: list[ManualBallKeyframe] | None = Field(default=None, max_length=2000)
