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

CalibrationDraftSource = Literal[
    "reconstruction",
    "frame-evidence",
    "saved",
    "approximate-seed",
    "manual-seed",
    "manual",
    "borrowed-previous",
    "borrowed-next",
    "borrowed-interpolation",
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
    # A UI provenance hint ("seed" / "projected" / "manual") echoed back from the
    # auto-proposed draft. The backend ignores it for the homography; it is
    # accepted so the client can round-trip draft anchors verbatim.
    source: str | None = Field(default=None, max_length=40)


class PitchCalibrationDraftRequest(TransportContract):
    scene_time: float = Field(ge=0, allow_inf_nan=False)
    preset: PitchCalibrationPreset | None = None


class PitchSideRequest(TransportContract):
    side: Literal["left", "right"]


class PitchCalibrationPreviewRequest(TransportContract):
    scene_time: float = Field(ge=0, allow_inf_nan=False)
    preset: PitchCalibrationPreset
    anchors: list[PitchCalibrationAnchor] = Field(min_length=4, max_length=12)


class PitchCalibrationSaveDraftRequest(PitchCalibrationPreviewRequest):
    source: CalibrationDraftSource
    # Poor automatic line-mask agreement is a warning for a manually aligned
    # frame, not an implicit veto. The operator must opt in explicitly so a
    # low-quality draft can never be persisted silently.
    accept_quality_warning: bool = False


class PitchCalibrationBorrowRequest(TransportContract):
    scene_time: float = Field(ge=0, allow_inf_nan=False)
    source: Literal["previous", "next", "interpolation"]
    preset: PitchCalibrationPreset | None = None
