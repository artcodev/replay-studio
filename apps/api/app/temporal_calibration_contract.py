from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pitch_calibration_contract import PitchCalibration, pitch_side


def _round_optional(value: float | None, digits: int = 3) -> float | None:
    return round(float(value), digits) if value is not None and np.isfinite(value) else None


def _matrix_payload(matrix: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 10) for value in row] for row in matrix]


@dataclass(frozen=True)
class TemporalCalibrationFrame:
    sample_index: int
    source_frame_index: int
    scene_time: float
    width: int
    height: int


@dataclass(frozen=True)
class TemporalCalibrationResult:
    resolved_by_sample: dict[int, PitchCalibration]
    anchor_by_sample: dict[int, int]
    uncertainty_by_sample: dict[int, float]
    recovered_frame_count: int
    metric_person_sample_count: int
    contact_point_diagnostics: dict | None = None
    demoted_anchors: list[dict] | None = None


@dataclass(frozen=True)
class CalibrationHypothesis:
    id: str
    target_sample_index: int
    anchor_sample_index: int
    anchor_source_frame_index: int
    anchor_scene_time: float
    direction: str
    calibration: PitchCalibration
    score: float
    uncertainty_metres: float
    motion_confidence: float
    temporal_distance_seconds: float
    motion_edge_indices: tuple[int, ...]
    disagreement_metres: float | None = None
    rejection_reasons: tuple[str, ...] = ()

    def as_dict(self, rank: int, selected: bool = False) -> dict:
        origin = "direct" if self.direction == "direct" else f"temporal-{self.direction}"
        return {
            "id": self.id,
            "rank": rank,
            "selected": selected,
            "origin": origin,
            "score": round(float(self.score), 5),
            "scoreKind": "heuristic-temporal-hypothesis-score",
            "visiblePitchSide": pitch_side(self.calibration.rectangle),
            "anchorFrameIndices": [self.anchor_source_frame_index],
            "anchorSampleIndices": [self.anchor_sample_index],
            "motionEdgeIndices": list(self.motion_edge_indices),
            "temporalDistanceSeconds": round(float(self.temporal_distance_seconds), 4),
            "motionConfidence": round(float(self.motion_confidence), 5),
            "uncertaintyP95Metres": round(float(self.uncertainty_metres), 4),
            "disagreementMetres": _round_optional(self.disagreement_metres, 4),
            "imageToPitch": _matrix_payload(self.calibration.image_to_pitch),
            "rejectionReasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class TemporalCalibrationResolution:
    selected: CalibrationHypothesis | None
    hypotheses: tuple[CalibrationHypothesis, ...]
    projection_source: str
    ambiguity_margin: float | None = None
    rejection_reasons: tuple[str, ...] = ()

    def hypotheses_payload(self) -> list[dict]:
        selected_id = self.selected.id if self.selected is not None else None
        return [
            hypothesis.as_dict(rank, selected=hypothesis.id == selected_id)
            for rank, hypothesis in enumerate(self.hypotheses, start=1)
        ]
