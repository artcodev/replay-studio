"""Pitch-calibration transport contracts."""

from typing import Literal

from pydantic import Field

from .transport_contract import TransportContract


PitchCalibrationPreset = Literal[
    "penalty-area-left",
    "goal-area-left",
    "center-circle",
    "goal-area-right",
    "penalty-area-right",
]


class CalibrationImagePoint(TransportContract):
    x: float = Field(ge=-4000, le=8000)
    y: float = Field(ge=-4000, le=8000)


class CalibrationPitchPoint(TransportContract):
    x: float = Field(ge=-52.5, le=52.5)
    z: float = Field(ge=-34, le=34)


class PitchCalibrationAnchor(TransportContract):
    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=120)
    image: CalibrationImagePoint
    pitch: CalibrationPitchPoint


class PitchCalibrationDraftRequest(TransportContract):
    scene_time: float = Field(ge=0, allow_inf_nan=False)
    preset: PitchCalibrationPreset | None = None


class PitchSideRequest(TransportContract):
    side: Literal["left", "right"]


class PitchCalibrationPreviewRequest(TransportContract):
    scene_time: float = Field(ge=0, allow_inf_nan=False)
    preset: PitchCalibrationPreset
    anchors: list[PitchCalibrationAnchor] = Field(min_length=4, max_length=12)
