from __future__ import annotations

import numpy as np

from .provider_contract import BallCandidate, ProviderUnavailable
from .wasb_configuration import INPUT_HEIGHT, INPUT_WIDTH


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.asarray([-direction[1], direction[0]], dtype=np.float32)


def affine_transforms(
    image_width: int,
    image_height: int,
    output_width: int = INPUT_WIDTH,
    output_height: int = INPUT_HEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Match upstream ``dataloaders.dataset_loader.get_transform`` exactly."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - image declares OpenCV
        raise ProviderUnavailable(f"OpenCV runtime is unavailable: {exc}") from exc

    center = np.asarray([image_width / 2.0, image_height / 2.0], dtype=np.float32)
    scale = float(max(image_height, image_width))
    source_direction = np.asarray([0.0, -scale * 0.5], dtype=np.float32)
    target_center = np.asarray(
        [output_width * 0.5, output_height * 0.5], dtype=np.float32
    )
    target_direction = np.asarray([0.0, -output_width * 0.5], dtype=np.float32)

    source = np.zeros((3, 2), dtype=np.float32)
    target = np.zeros((3, 2), dtype=np.float32)
    source[0] = center
    source[1] = center + source_direction
    source[2] = _third_point(source[0], source[1])
    target[0] = target_center
    target[1] = target_center + target_direction
    target[2] = _third_point(target[0], target[1])
    return (
        cv2.getAffineTransform(source, target),
        cv2.getAffineTransform(target, source),
    )


def component_candidates(
    heatmap: np.ndarray,
    inverse_affine: np.ndarray,
    image_size: tuple[int, int],
    *,
    score_threshold: float,
    max_candidates: int,
) -> list[BallCandidate]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - image declares OpenCV
        raise ProviderUnavailable(f"OpenCV runtime is unavailable: {exc}") from exc

    mask = (heatmap > score_threshold).astype(np.uint8)
    component_count, labels = cv2.connectedComponents(mask)
    image_width, image_height = image_size
    candidates: list[BallCandidate] = []
    for component_index in range(1, component_count):
        ys, xs = np.where(labels == component_index)
        if xs.size == 0:
            continue
        weights = heatmap[ys, xs].astype(np.float64)
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            continue
        heatmap_x = float(np.sum(xs * weights) / weight_sum)
        heatmap_y = float(np.sum(ys * weights) / weight_sum)
        source = inverse_affine @ np.asarray(
            [heatmap_x, heatmap_y, 1.0], dtype=np.float64
        )
        source_x = float(source[0])
        source_y = float(source[1])
        if not np.isfinite((source_x, source_y)).all():
            continue
        if (
            source_x < 0
            or source_y < 0
            or source_x >= image_width
            or source_y >= image_height
        ):
            continue
        peak = float(weights.max())
        candidates.append(
            BallCandidate(
                x=source_x,
                y=source_y,
                confidence=peak,
                heatmap_peak=peak,
                component_score=weight_sum,
                component_area=int(xs.size),
                metadata={"heatmapComponent": component_index},
            )
        )
    return sorted(
        candidates,
        key=lambda item: item.confidence,
        reverse=True,
    )[:max_candidates]
