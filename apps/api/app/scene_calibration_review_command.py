from __future__ import annotations

"""Confirm a calibration inspection gate so the full reconstruction may run.

A calibrate run stops at an inspection gate whenever it could not resolve every
frame. The operator inspects the unresolved frames and either fixes them (which
changes the inputs and forces a recalibration) or accepts the gap. This command
records that acceptance: it flips the gate to ``confirmed`` so the subsequent
full run is no longer refused. It never touches the job/lease machine — the gate
is a scene-level workflow state.
"""

from datetime import UTC, datetime

from .reconstruction_errors import ReconstructionError
from .reconstruction_coordinate_policy import (
    EXPLICIT_IMAGE_FALLBACK,
    unresolved_review_sample_indices,
)
from .reconstruction_calibration_fingerprint import calibration_input_fingerprint
from .scene_repository import scenes


def confirm_calibration_review(scene: dict) -> dict:
    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    if reconstruction.get("status") in {"queued", "processing"}:
        raise ReconstructionError(
            "Wait for calibration to finish before confirming the review"
        )
    if reconstruction.get("stage") != "calibration":
        raise ReconstructionError("There is no calibration review to confirm")
    review = reconstruction.get("calibrationReview") or {}
    if not review:
        raise ReconstructionError(
            "No published calibration review exists for the current scene; "
            "run calibration successfully before authorizing image fallback"
        )
    status = str(review.get("status") or "")
    if status not in {"review", "ready", "confirmed"}:
        raise ReconstructionError(
            "The calibration review is not in a confirmable state"
        )
    # The gate must describe the current inputs. If a manual anchor (or any other
    # calibration-affecting edit) changed them during review, the gate is stale
    # and the operator must recalibrate rather than confirm evidence that no
    # longer corresponds to the scene.
    if str(review.get("calibrationInputFingerprint") or "") != (
        calibration_input_fingerprint(scene)
    ):
        raise ReconstructionError(
            "The scene inputs changed since calibration; run calibration again "
            "before confirming"
        )
    fallback_samples = unresolved_review_sample_indices(review)
    reconstruction["calibrationReview"] = {
        **review,
        "status": "confirmed",
        "confirmedAt": datetime.now(UTC).isoformat(),
        **(
            {
                "fallbackPolicy": EXPLICIT_IMAGE_FALLBACK,
                "fallbackSampleIndices": list(fallback_samples),
            }
            if fallback_samples
            else {}
        ),
    }
    video["reconstruction"] = reconstruction
    return scenes.put(scene)


__all__ = ("confirm_calibration_review",)
