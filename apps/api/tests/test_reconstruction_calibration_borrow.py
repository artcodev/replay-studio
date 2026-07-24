from __future__ import annotations

import pytest

from app.reconstruction_calibration_borrow import (
    _require_contiguous_camera_reference,
)
from app.reconstruction_errors import ReconstructionError


def _frame(source_index: int, motion_status: str) -> dict:
    return {
        "sourceFrameIndex": source_index,
        "cameraMotion": {"status": motion_status},
    }


def test_neighbor_calibration_can_only_cross_reliable_camera_motion():
    evidence = [
        _frame(317, "first-frame"),
        _frame(318, "estimated"),
        _frame(319, "estimated"),
    ]

    _require_contiguous_camera_reference(evidence, 0, 2)


@pytest.mark.parametrize("status", ["cut", "unreliable", ""])
def test_neighbor_calibration_is_blocked_across_a_reference_reset(status: str):
    evidence = [
        _frame(317, "first-frame"),
        _frame(318, status),
    ]

    with pytest.raises(
        ReconstructionError,
        match=r"Cannot borrow calibration across a camera-motion boundary.*#318",
    ):
        _require_contiguous_camera_reference(evidence, 0, 1)
