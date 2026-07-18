from __future__ import annotations

"""Numerically bounded homography normalization and interpolation."""

import numpy as np


def normalise_interpolation_homography(
    matrix: np.ndarray,
    frame_size: tuple[int, int],
) -> tuple[np.ndarray | None, str | None]:
    """Normalise and reject homographies that are unsafe at pitch-foot probes."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        return None, "matrix-shape-invalid"
    if not np.isfinite(value).all():
        return None, "matrix-non-finite"
    scale = float(value[2, 2])
    if abs(scale) < 1e-10:
        return None, "matrix-scale-degenerate"
    value = value / scale
    try:
        singular_values = np.linalg.svd(value, compute_uv=False)
    except np.linalg.LinAlgError:
        return None, "matrix-svd-failed"
    if (
        not np.isfinite(singular_values).all()
        or float(singular_values[-1]) <= 1e-12 * max(1.0, float(singular_values[0]))
        or float(singular_values[0] / singular_values[-1]) > 1e12
    ):
        return None, "matrix-near-singular"

    width, height = frame_size
    probes = np.asarray(
        [
            [width * x_fraction, height * y_fraction, 1.0]
            for y_fraction in (0.58, 0.78, 0.96)
            for x_fraction in (0.08, 0.50, 0.92)
        ],
        dtype=np.float64,
    ).T
    projected = value @ probes
    denominator_scale = np.linalg.norm(value[2, :]) * np.linalg.norm(probes, axis=0)
    if np.any(
        np.abs(projected[2, :])
        <= 1e-9 * np.maximum(1.0, denominator_scale)
    ):
        return None, "matrix-probe-at-infinity"
    projected = projected[:2, :] / projected[2:3, :]
    if not np.isfinite(projected).all():
        return None, "matrix-projection-non-finite"
    return value, None


def interpolate_homography_bounded(
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    frame_size: tuple[int, int],
) -> tuple[np.ndarray | None, str | None]:
    """Linearly blend consistently scaled nearby H matrices, then revalidate."""

    if not 0.0 < alpha < 1.0:
        return None, "alpha-outside-open-interval"
    lower_value, lower_reason = normalise_interpolation_homography(lower, frame_size)
    if lower_value is None:
        return None, f"lower-{lower_reason}"
    upper_value, upper_reason = normalise_interpolation_homography(upper, frame_size)
    if upper_value is None:
        return None, f"upper-{upper_reason}"
    candidate = lower_value * (1.0 - alpha) + upper_value * alpha
    candidate, candidate_reason = normalise_interpolation_homography(
        candidate,
        frame_size,
    )
    if candidate is None:
        return None, f"interpolated-{candidate_reason}"
    return candidate, None
