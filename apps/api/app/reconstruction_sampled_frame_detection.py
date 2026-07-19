from __future__ import annotations

"""Single-pass sampled-frame detection and calibration accumulation."""

from pathlib import Path

import app.person_base_detection_cache as person_base_detection_cache

from .person_crop_store import (
    attach_crop_records,
    extract_and_store_person_crops,
    person_crop_store_runtime,
)
from .person_detection_cache import frame_content_sha256
from .reconstruction_inputs import source_frame_index
from .reconstruction_person_annotations import (
    apply_person_annotations,
    frame_annotations,
)
from .reconstruction_progress import ReconstructionProgress
from .reconstruction_reid_evidence import capture_detection_observations
from .reconstruction_sampled_calibration import SampledCalibrationAccumulator
from .reconstruction_sampled_detection_preparation import (
    SampledDetectionRuntime,
)
from .reconstruction_sampled_frame_contract import (
    SampledCalibrationInputs,
    SampledFrameAnalysis,
)


def analyze_sampled_frames(
    scene: dict,
    frames: list[tuple[Path, float]],
    runtime: SampledDetectionRuntime,
    calibration_inputs: SampledCalibrationInputs,
    progress: ReconstructionProgress,
) -> SampledFrameAnalysis:
    person_frames = []
    generic_ball_frames = []
    person_counts: list[int] = []
    ball_counts: list[int] = []
    calibration = SampledCalibrationAccumulator(scene, calibration_inputs)
    crop_store_directory, crop_policy = person_crop_store_runtime()
    crop_store_diagnostics: dict = {"hits": 0, "stores": 0, "storeErrors": 0}

    for sample_index, (path, time) in enumerate(frames):
        # This is the only sampled-frame decode boundary. Camera/calibration
        # consumes the returned image directly rather than reading it again.
        image, people, balls = person_base_detection_cache.cached_base_frame_detections(
            runtime.model,
            path,
            runtime.person_cache_directory,
            runtime.person_detector_input,
            runtime.person_cache_diagnostics,
        )
        source_index = source_frame_index(path)
        people = apply_person_annotations(
            image,
            people,
            frame_annotations(scene, source_index),
        )
        capture_detection_observations(people, source_index)
        try:
            frame_digest = frame_content_sha256(path)
        except OSError:
            frame_digest = None
        if frame_digest is not None:
            crop_records = extract_and_store_person_crops(
                crop_store_directory,
                image=image,
                frame_sha256=frame_digest,
                detections=people,
                policy=crop_policy,
                diagnostics=crop_store_diagnostics,
            )
            attach_crop_records(people, crop_records, frame_sha256=frame_digest)
        calibration.add_frame(
            sample_index=sample_index,
            source_index=source_index,
            scene_time=time,
            image=image,
            people=people,
        )
        person_counts.append(len(people))
        ball_counts.append(len(balls))
        person_frames.append((people, time))
        generic_ball_frames.append((balls, time))
        progress.update(
            "detection",
            3,
            "Detecting people and camera evidence",
            f"Sample {sample_index + 1}/{len(frames)} · "
            f"{len(people)} people · {len(balls)} generic ball fallback candidate(s).",
            62,
            84,
            completed=sample_index + 1,
            total=len(frames),
            eta_padding=5.0,
        )

    runtime.person_cache_diagnostics.update(
        {
            "status": (
                "degraded"
                if runtime.person_cache_diagnostics["errors"]
                else "ready"
            ),
            "hitRatio": round(
                runtime.person_cache_diagnostics["hits"] / max(1, len(frames)),
                4,
            ),
            "baseBoundary": (
                "pre-annotation/pre-calibration/pre-tracking/pre-reid/pre-ocr"
            ),
            "personCropStore": crop_store_diagnostics,
        }
    )
    return SampledFrameAnalysis(
        person_frames=person_frames,
        generic_ball_frames=generic_ball_frames,
        person_counts=person_counts,
        ball_counts=ball_counts,
        calibration=calibration.result(),
    )


__all__ = ["analyze_sampled_frames"]
