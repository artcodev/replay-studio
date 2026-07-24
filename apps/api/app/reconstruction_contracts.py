"""Reconstruction execution transport contracts."""

from typing import Literal

from pydantic import Field

from .transport_contract import TransportContract


# "football.pt" is an owner-supplied football-tuned checkpoint dropped at the
# repository root; it is never auto-downloaded and fails with an installation
# hint when absent. Its class map is resolved by name (player/goalkeeper/
# referee/ball), so domain checkpoints keep officials and the ball.
ReconstructionModel = Literal[
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
    "football.pt",
]

BallDetectionBackend = Literal[
    "generic-ultralytics",
    "dedicated-ultralytics",
    "wasb-service",
]

# "skip-manual-authoritative" is valid only while the manual ball trajectory
# is the authoritative channel; dense ball inference is then not executed.
BallDetectionProfile = Literal[
    "automatic",
    "skip-manual-authoritative",
]

# "off" skips shirt-number OCR entirely: cheaper runs for manually bound
# rosters, at the cost of automatic jersey merge evidence.
JerseyOcrProfile = Literal[
    "automatic",
    "off",
]

# "pose-feet" projects RTMPose feet evidence from stored person crops instead
# of the bbox bottom-centre; ineligible crops fall back to bbox explicitly.
ContactPointProfile = Literal[
    "bbox-bottom",
    "pose-feet",
]

# Two separate process contracts share the durable job transport. "calibrate"
# computes and publishes only immutable calibration evidence. "full" has no
# calibration authority: it consumes the pinned artifact, then runs detection,
# identity, ball and publication. Mode is not itself a calibration input.
ReconstructionMode = Literal[
    "calibrate",
    "full",
]


class ReconstructionRequest(TransportContract):
    model: ReconstructionModel | None = None
    ball_backend: BallDetectionBackend | None = None
    ball_detection_profile: BallDetectionProfile | None = None
    jersey_ocr_profile: JerseyOcrProfile | None = None
    contact_point_profile: ContactPointProfile | None = None
    mode: ReconstructionMode | None = None
    # None means native source cadence. A positive lower value is an explicit
    # operator-selected sampling contract, not a process-global fallback.
    frame_rate: float | None = Field(default=None, gt=0.0, le=240.0)
    # None lets the command choose its canonical default/inherit an existing
    # calibration. Zero runs direct PnLCalib on every selected frame; a positive
    # value is an explicit sparse-performance tradeoff owned by this scene.
    direct_calibration_max_gap_seconds: float | None = Field(
        default=None,
        ge=0.0,
        le=5.0,
    )
