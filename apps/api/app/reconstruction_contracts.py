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


class ReconstructionRequest(TransportContract):
    model: ReconstructionModel | None = None
    ball_backend: BallDetectionBackend | None = None
