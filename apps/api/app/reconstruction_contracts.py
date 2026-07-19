"""Reconstruction execution transport contracts."""

from typing import Literal

from .transport_contract import TransportContract


ReconstructionModel = Literal[
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
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


class ReconstructionRequest(TransportContract):
    model: ReconstructionModel | None = None
    ball_backend: BallDetectionBackend | None = None
    ball_detection_profile: BallDetectionProfile | None = None
    jersey_ocr_profile: JerseyOcrProfile | None = None
