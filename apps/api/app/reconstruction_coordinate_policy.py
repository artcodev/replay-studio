from __future__ import annotations

"""Explicit authorization for non-metric person association."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from .reconstruction_errors import ReconstructionError


TrackingCoordinatePolicy = Literal[
    "metric-required",
    "explicit-image-fallback",
]

METRIC_REQUIRED: TrackingCoordinatePolicy = "metric-required"
EXPLICIT_IMAGE_FALLBACK: TrackingCoordinatePolicy = "explicit-image-fallback"
TRACKING_COORDINATE_POLICIES = {METRIC_REQUIRED, EXPLICIT_IMAGE_FALLBACK}


def _sample_indices(values: object) -> tuple[int, ...]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return ()
    result: set[int] = set()
    for value in values:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(result))


def unresolved_review_sample_indices(review: Mapping[str, Any]) -> tuple[int, ...]:
    samples = review.get("unresolvedSamples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        return ()
    return _sample_indices(
        sample.get("sampleIndex")
        for sample in samples
        if isinstance(sample, Mapping) and not bool(sample.get("resolved"))
    )


def resolve_full_run_coordinate_authorization(
    reconstruction: Mapping[str, Any],
    *,
    calibration_input_fingerprint: str,
) -> tuple[TrackingCoordinatePolicy, dict[str, Any] | None]:
    """Require a current calibration gate before a full reconstruction."""

    if reconstruction.get("stage") == "calibration":
        review = reconstruction.get("calibrationReview")
        if not isinstance(review, Mapping):
            raise ReconstructionError(
                "Calibration must finish before reconstruction can start"
            )
        if str(review.get("calibrationInputFingerprint") or "") != (
            calibration_input_fingerprint
        ):
            raise ReconstructionError(
                "Calibration inputs changed; run calibration again before reconstructing"
            )
        status = str(review.get("status") or "")
        if status == "ready":
            return METRIC_REQUIRED, None
        if status == "review":
            raise ReconstructionError(
                "Calibration has unresolved frames: fix them, or explicitly authorize "
                "image fallback in the calibration review"
            )
        if status != "confirmed":
            raise ReconstructionError(
                "Calibration review is not ready for reconstruction"
            )
        fallback_samples = unresolved_review_sample_indices(review)
        if not fallback_samples:
            return METRIC_REQUIRED, None
        return EXPLICIT_IMAGE_FALLBACK, {
            "policy": EXPLICIT_IMAGE_FALLBACK,
            "calibrationInputFingerprint": calibration_input_fingerprint,
            "sampleIndices": list(fallback_samples),
            "confirmedAt": review.get("confirmedAt"),
        }

    # A completed full run may be rebuilt with the same authorization while its
    # calibration fingerprint remains current. This preserves explicit consent
    # without silently extending it to changed source or pitch geometry.
    if str(reconstruction.get("calibrationInputFingerprint") or "") == (
        calibration_input_fingerprint
    ):
        policy = str(reconstruction.get("trackingCoordinatePolicy") or "")
        if policy == METRIC_REQUIRED:
            return METRIC_REQUIRED, None
        consent = reconstruction.get("calibrationFallbackConsent")
        if policy == EXPLICIT_IMAGE_FALLBACK and isinstance(consent, Mapping):
            if str(consent.get("calibrationInputFingerprint") or "") == (
                calibration_input_fingerprint
            ):
                return EXPLICIT_IMAGE_FALLBACK, dict(consent)

    raise ReconstructionError(
        "Run pitch calibration before reconstructing the scene"
    )


def validate_runtime_calibration_coverage(
    *,
    policy: str,
    consent: Mapping[str, Any] | None,
    calibration_input_fingerprint: str,
    sampled_frame_count: int,
    resolved_sample_indices: Sequence[int],
) -> dict[str, Any]:
    """Fail closed unless every missing calibration was explicitly accepted."""

    if policy not in TRACKING_COORDINATE_POLICIES:
        raise ReconstructionError("Unknown tracking coordinate policy")
    resolved = {int(value) for value in resolved_sample_indices}
    missing = sorted(set(range(sampled_frame_count)) - resolved)
    if not missing:
        return {
            "policy": policy,
            "resolvedFrameCount": sampled_frame_count,
            "fallbackFrameCount": 0,
            "fallbackSampleIndices": [],
        }
    if policy != EXPLICIT_IMAGE_FALLBACK or not isinstance(consent, Mapping):
        raise ReconstructionError(
            f"Metric calibration is missing on {len(missing)} sampled frame(s); "
            "fix calibration before tracking"
        )
    if str(consent.get("policy") or "") != EXPLICIT_IMAGE_FALLBACK or str(
        consent.get("calibrationInputFingerprint") or ""
    ) != calibration_input_fingerprint:
        raise ReconstructionError(
            "Image fallback consent does not match the current reconstruction inputs"
        )
    allowed = set(_sample_indices(consent.get("sampleIndices")))
    unapproved = sorted(set(missing) - allowed)
    if unapproved:
        raise ReconstructionError(
            "Calibration produced unapproved unresolved frames; review calibration "
            "again before tracking"
        )
    return {
        "policy": policy,
        "resolvedFrameCount": sampled_frame_count - len(missing),
        "fallbackFrameCount": len(missing),
        "fallbackSampleIndices": missing,
    }


__all__ = [
    "EXPLICIT_IMAGE_FALLBACK",
    "METRIC_REQUIRED",
    "TRACKING_COORDINATE_POLICIES",
    "TrackingCoordinatePolicy",
    "resolve_full_run_coordinate_authorization",
    "unresolved_review_sample_indices",
    "validate_runtime_calibration_coverage",
]
