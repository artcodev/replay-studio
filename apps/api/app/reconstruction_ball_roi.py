from __future__ import annotations

"""Adaptive region-of-interest policy for dense ball detection."""

from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Mapping

from .ball_detection_contract import BallDetectionBatch, BallDetector
from .reconstruction_errors import ReconstructionError


@dataclass(frozen=True, slots=True)
class AdaptiveBallRoiConfig:
    full_scan_interval: int
    max_regions: int
    padding_pixels: int
    algorithm_version: str


def adaptive_ball_roi_config(
    detector: BallDetector,
    detector_input: Mapping[str, object] | None,
) -> AdaptiveBallRoiConfig | None:
    """Resolve the opt-in strategy for the dedicated local detector only."""

    if detector.backend_name != "dedicated-ultralytics":
        return None
    if not callable(getattr(detector, "detect_regions", None)):
        return None
    if not isinstance(detector_input, Mapping):
        return None
    if str(detector_input.get("backend") or "") != "dedicated-ultralytics":
        return None
    raw = detector_input.get("adaptiveRoi")
    if not isinstance(raw, Mapping) or not bool(raw.get("enabled", False)):
        return None
    interval = int(raw.get("fullScanIntervalFrames") or 1)
    max_regions = int(raw.get("maxRegions") or 0)
    padding = int(raw.get("paddingPixels") or 0)
    algorithm_version = str(raw.get("algorithmVersion") or "")
    reacquire_policy = str(raw.get("reacquirePolicy") or "")
    if interval <= 1:
        return None
    if max_regions <= 0 or padding <= 0:
        raise ReconstructionError(
            "Dedicated ball adaptive ROI requires positive maxRegions and paddingPixels"
        )
    if algorithm_version != "adaptive-roi-v1":
        raise ReconstructionError(
            f"Unsupported dedicated ball adaptive ROI algorithm: {algorithm_version or 'missing'}"
        )
    if reacquire_policy != "same-frame-global-on-miss":
        raise ReconstructionError(
            "Dedicated ball adaptive ROI must use same-frame-global-on-miss"
        )
    return AdaptiveBallRoiConfig(
        full_scan_interval=interval,
        max_regions=max_regions,
        padding_pixels=padding,
        algorithm_version=algorithm_version,
    )


def region_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


def ball_roi_regions(
    seeds: list[dict],
    image_size: tuple[int, int],
    *,
    max_regions: int,
    padding_pixels: int,
) -> list[tuple[float, float, float, float]]:
    """Build fixed-context ROIs and deduplicate strongly overlapping seeds.

    Border windows are shifted rather than clipped whenever the image is large
    enough. This keeps the intended 640px context and therefore the same tiny
    object scale as one global tile.
    """

    image_width, image_height = image_size
    target_width = min(float(image_width), float(padding_pixels * 2))
    target_height = min(float(image_height), float(padding_pixels * 2))
    regions: list[tuple[float, float, float, float]] = []
    ordered = sorted(
        seeds,
        key=lambda item: float(item.get("confidence") or 0.0),
        reverse=True,
    )
    for seed in ordered:
        detector_metadata = seed.get("detectorMetadata")
        tile = (
            detector_metadata.get("tile")
            if isinstance(detector_metadata, Mapping)
            else None
        )
        region: tuple[float, float, float, float] | None = None
        if isinstance(tile, Mapping):
            try:
                tile_left = float(tile["x"])
                tile_top = float(tile["y"])
                tile_width = float(tile["width"])
                tile_height = float(tile["height"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if (
                    isfinite(tile_left)
                    and isfinite(tile_top)
                    and tile_width > 0
                    and tile_height > 0
                ):
                    region = (
                        max(0.0, tile_left),
                        max(0.0, tile_top),
                        min(float(image_width), tile_left + tile_width),
                        min(float(image_height), tile_top + tile_height),
                    )
        if region is None:
            try:
                center_x = float(seed["x"])
                center_y = float(seed["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (isfinite(center_x) and isfinite(center_y)):
                continue
            left = min(max(0.0, center_x - target_width / 2.0), image_width - target_width)
            top = min(max(0.0, center_y - target_height / 2.0), image_height - target_height)
            region = (left, top, left + target_width, top + target_height)
        if region[2] <= region[0] or region[3] <= region[1]:
            continue
        # Nearby hypotheses often describe the same tiny object. Running two
        # nearly identical crops buys no recall, while the higher-confidence
        # region already contains both seed centres with generous context.
        if any(region_iou(region, kept) >= 0.5 for kept in regions):
            continue
        regions.append(region)
        if len(regions) >= max_regions:
            break
    return regions


def scene_camera_cut_between(scene: dict, previous_time: float, time: float) -> bool:
    for item in scene.get("payload", {}).get("cameraCuts") or []:
        try:
            cut_time = float(item["t"])
        except (KeyError, TypeError, ValueError):
            continue
        if previous_time + 1e-6 < cut_time <= time + 1e-6:
            return True
    return False


def adaptive_ball_detect(
    detector: BallDetector,
    path: Path,
    *,
    frame_index: int,
    frame_count: int,
    timestamp: float,
    context_paths: tuple[Path, Path],
    seeds: list[dict],
    image_size: tuple[int, int] | None,
    config: AdaptiveBallRoiConfig,
    force_global_reason: str | None = None,
) -> BallDetectionBatch:
    periodic = frame_index % config.full_scan_interval == 0
    final_frame = frame_index == frame_count - 1
    regions = (
        ball_roi_regions(
            seeds,
            image_size,
            max_regions=config.max_regions,
            padding_pixels=config.padding_pixels,
        )
        if image_size is not None and seeds
        else []
    )
    global_reason = force_global_reason
    if global_reason is None and frame_index == 0:
        global_reason = "initial-frame"
    elif global_reason is None and final_frame:
        global_reason = "final-frame"
    elif global_reason is None and periodic:
        global_reason = "periodic"
    elif global_reason is None and not seeds:
        global_reason = "no-seed"
    elif global_reason is None and not regions:
        global_reason = "no-valid-roi"

    if global_reason is not None:
        batch = detector.detect(
            path,
            frame_index=frame_index,
            timestamp=timestamp,
            context_frames=context_paths,
        )
        return replace(
            batch,
            metadata={
                **dict(batch.metadata),
                "scanMode": "global",
                "globalScanReason": global_reason,
                "adaptiveRoiAlgorithm": config.algorithm_version,
                "seedCandidateCount": len(seeds),
            },
        )

    detect_regions = getattr(detector, "detect_regions")
    roi_error: str | None = None
    try:
        roi_batch = detect_regions(
            path,
            regions,
            frame_index=frame_index,
            timestamp=timestamp,
            context_frames=context_paths,
        )
    except Exception as exc:
        # A crop-specific decoding/batching problem must not trip the outer
        # backend circuit before the normal detector gets one global attempt.
        roi_error = f"{type(exc).__name__}: {exc}"
        roi_batch = BallDetectionBatch(
            candidates=(),
            image_size=image_size or (0, 0),
            backend=detector.backend_name,
            metadata={"scanMode": "roi", "roiError": roi_error},
        )
    roi_metadata = {
        **dict(roi_batch.metadata),
        "scanMode": "roi",
        "adaptiveRoiAlgorithm": config.algorithm_version,
        "seedCandidateCount": len(seeds),
        "requestedRegionCount": len(regions),
    }
    if roi_batch.candidates:
        return replace(roi_batch, metadata=roi_metadata)

    # Accuracy boundary: a crop miss does not create an artificial hole and
    # does not wait until the next periodic scan. Reacquire globally on this
    # exact timestamp, preserving the dense temporal contract.
    global_batch = detector.detect(
        path,
        frame_index=frame_index,
        timestamp=timestamp,
        context_frames=context_paths,
    )
    return replace(
        global_batch,
        metadata={
            **dict(global_batch.metadata),
            "scanMode": "global-reacquire",
            "globalScanReason": "roi-error" if roi_error else "roi-miss",
            "adaptiveRoiAlgorithm": config.algorithm_version,
            "seedCandidateCount": len(seeds),
            "roiAttempt": roi_metadata,
        },
    )


def adaptive_ball_roi_diagnostics(
    batches: list[dict],
    config: AdaptiveBallRoiConfig | None,
) -> dict:
    if config is None:
        return {"enabled": False}
    modes = [
        str(item.get("scanMode") or (item.get("metadata") or {}).get("scanMode") or "")
        for item in batches
    ]
    metadata = [
        item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        for item in batches
    ]
    global_crop_count = sum(
        int(item.get("tileCount") or 0)
        for mode, item in zip(modes, metadata, strict=True)
        if mode in {"global", "global-reacquire"}
    )
    roi_crop_count = 0
    for mode, item in zip(modes, metadata, strict=True):
        if mode == "roi":
            roi_crop_count += int(item.get("roiRegionCount") or item.get("tileCount") or 0)
        elif mode == "global-reacquire":
            attempt = item.get("roiAttempt")
            if isinstance(attempt, Mapping):
                roi_crop_count += int(
                    attempt.get("roiRegionCount")
                    or attempt.get("tileCount")
                    or attempt.get("requestedRegionCount")
                    or 0
                )
    global_tile_counts = [
        int(item.get("tileCount") or 0)
        for mode, item in zip(modes, metadata, strict=True)
        if mode in {"global", "global-reacquire"} and int(item.get("tileCount") or 0) > 0
    ]
    reference_full_scan_crops = max(global_tile_counts, default=0)
    baseline_crop_count = reference_full_scan_crops * len(batches)
    total_model_crop_count = global_crop_count + roi_crop_count
    return {
        "enabled": True,
        "algorithmVersion": config.algorithm_version,
        "fullScanIntervalFrames": config.full_scan_interval,
        "maxRegions": config.max_regions,
        "paddingPixels": config.padding_pixels,
        "globalScanFrameCount": sum(mode == "global" for mode in modes),
        "roiScanFrameCount": sum(mode == "roi" for mode in modes),
        "roiReacquireFrameCount": sum(mode == "global-reacquire" for mode in modes),
        "globalInferenceFrameCount": sum(
            mode in {"global", "global-reacquire"} for mode in modes
        ),
        "roiInferenceFrameCount": sum(
            mode in {"roi", "global-reacquire"} for mode in modes
        ),
        "globalCropCount": global_crop_count,
        "roiCropCount": roi_crop_count,
        "totalModelCropCount": total_model_crop_count,
        "referenceFullScanCropCount": reference_full_scan_crops,
        "estimatedFullScanBaselineCropCount": baseline_crop_count,
        "estimatedCropReductionRatio": (
            round(1.0 - total_model_crop_count / baseline_crop_count, 4)
            if baseline_crop_count > 0
            else None
        ),
    }

