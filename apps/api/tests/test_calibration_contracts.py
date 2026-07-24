from __future__ import annotations

from app.calibration_contracts import (
    PitchCalibrationPreviewRequest,
    PitchCalibrationSaveDraftRequest,
)


def _anchor(index: int, *, source: str | None = None) -> dict:
    anchor = {
        "id": f"a{index}",
        "label": f"anchor {index}",
        "image": {"x": float(index), "y": float(index)},
        "pitch": {"x": 0.0, "z": 0.0},
    }
    if source is not None:
        anchor["source"] = source
    return anchor


def test_preview_request_round_trips_draft_anchor_source_hint():
    # The auto-proposed draft tags each anchor with a provenance "source"; the
    # client echoes it back verbatim when previewing/applying, so the request
    # must accept (and ignore) it rather than fail closed.
    request = PitchCalibrationPreviewRequest.model_validate(
        {
            "scene_time": 0.2,
            "preset": "center-circle",
            "anchors": [
                _anchor(0, source="projected"),
                _anchor(1, source="seed"),
                _anchor(2, source="manual"),
                _anchor(3),
            ],
        }
    )

    assert [anchor.source for anchor in request.anchors] == [
        "projected",
        "seed",
        "manual",
        None,
    ]


def test_save_request_requires_explicit_consent_for_quality_warning():
    payload = {
        "scene_time": 0.2,
        "preset": "center-circle",
        "anchors": [_anchor(index) for index in range(4)],
        "source": "manual",
    }

    regular = PitchCalibrationSaveDraftRequest.model_validate(payload)
    accepted = PitchCalibrationSaveDraftRequest.model_validate(
        {**payload, "accept_quality_warning": True}
    )

    assert regular.accept_quality_warning is False
    assert accepted.accept_quality_warning is True
