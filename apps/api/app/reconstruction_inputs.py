from __future__ import annotations

"""Immutable reconstruction input discovery and lazy model loading."""

import os
from collections.abc import Mapping
from math import ceil, floor, isfinite
from pathlib import Path
from threading import Lock

from .checkpoint_identity import file_content_sha256
from .config import get_settings
from .reconstruction_errors import ReconstructionError
from .scene_frame_exclusions import excluded_source_frame_indices
from .video_media_paths import video_generation_directory


_models: dict[tuple[str, str | None], object] = {}
_model_lock = Lock()

# Owner-supplied checkpoints are never auto-downloaded: a missing file must
# fail with an installation hint instead of an Ultralytics hub lookup. The
# owner manages the licensing of any weights placed here.
CUSTOM_MODEL_FILES = frozenset({"football.pt"})


def _custom_model_candidates(model_name: str) -> list[Path]:
    filename = Path(model_name).name
    candidates: list[Path] = []
    override = str(get_settings().football_detector_weights or "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    # The repository root works for local runs; ./models is baked into the
    # api image (mirroring the dedicated ball weights), so the same relative
    # layout resolves to /app/models inside the container.
    candidates.append(Path(filename))
    candidates.append(Path("models") / filename)
    resolved: list[Path] = []
    for candidate in candidates:
        absolute = (
            candidate if candidate.is_absolute() else Path.cwd() / candidate
        ).resolve()
        if absolute not in resolved:
            resolved.append(absolute)
    return resolved


def resolve_custom_model_checkpoint(model_name: str) -> Path | None:
    if model_name not in CUSTOM_MODEL_FILES:
        return None
    return next(
        (
            candidate
            for candidate in _custom_model_candidates(model_name)
            if candidate.is_file()
        ),
        None,
    )


def require_model_weights_available(model_name: str) -> None:
    if model_name not in CUSTOM_MODEL_FILES:
        return
    if resolve_custom_model_checkpoint(model_name) is None:
        searched = "; ".join(
            str(candidate) for candidate in _custom_model_candidates(model_name)
        )
        raise ReconstructionError(
            f"The {model_name} checkpoint is not installed: copy the "
            "owner-supplied weights to models/"
            f"{Path(model_name).name} (baked into the api image on rebuild) "
            "or point FOOTBALL_DETECTOR_WEIGHTS at the file. "
            f"Searched: {searched}"
        )


def resolve_analysis_frame_rate(source_fps: float) -> float:
    """Return the authoritative cadence materialized from the source video.

    Immutable generations always retain the source cadence. A cheaper cadence
    is an explicit per-scene reconstruction input selected later; silently
    throwing source frames away during ingest makes a higher-quality
    calibration impossible without re-importing the asset.
    """

    source = float(source_fps or 0.0)
    if not isfinite(source) or source <= 0.0:
        raise ReconstructionError("The source video has no valid frame rate")
    return source


def resolve_sampling_frame_rate(
    scene: Mapping,
    requested_frame_rate: float | None = None,
) -> float:
    """Resolve one explicit consumer cadence without silently thinning frames.

    ``None`` means native source cadence unless the queued reconstruction has
    already pinned a selection. A requested cadence must be available in the
    immutable generation; an older capped generation fails closed and asks for
    regeneration instead of pretending that its reduced cadence is native.
    """

    payload = scene.get("payload")
    video = payload.get("videoAsset") if isinstance(payload, Mapping) else None
    if not isinstance(video, Mapping):
        raise ReconstructionError("Scene has no source video")
    reconstruction = video.get("reconstruction")
    queued_rate = (
        reconstruction.get("samplingFrameRate")
        if isinstance(reconstruction, Mapping)
        else None
    )
    source_rate = float(video.get("fps") or 0.0)
    materialized_rate = float(video.get("analysisFps") or 0.0)
    if not isfinite(source_rate) or source_rate <= 0.0:
        raise ReconstructionError("The source video has no valid frame rate")
    if not isfinite(materialized_rate) or materialized_rate <= 0.0:
        raise ReconstructionError("The analysis-frame generation has no valid frame rate")
    selected_value = requested_frame_rate
    if selected_value is None:
        selected_value = queued_rate if queued_rate is not None else source_rate
    selected = float(selected_value)
    if not isfinite(selected) or selected <= 0.0:
        raise ReconstructionError("The selected analysis frame rate must be positive")
    tolerance = 1e-3
    if selected > source_rate + tolerance:
        raise ReconstructionError(
            f"Selected analysis cadence {selected:g} FPS exceeds the source "
            f"cadence {source_rate:g} FPS"
        )
    if selected > materialized_rate + tolerance:
        raise ReconstructionError(
            f"Native analysis requires {selected:g} FPS, but the published "
            f"generation contains only {materialized_rate:g} FPS. Regenerate "
            "source-resolution analysis frames before calibration."
        )
    return min(selected, source_rate, materialized_rate)


def _model_cache_key(name: str) -> tuple[str, str | None]:
    candidate = Path(name).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    checkpoint = candidate.resolve()
    if checkpoint.is_file():
        return str(checkpoint), file_content_sha256(checkpoint)
    return name, None


def load_model(model_name: str | None = None):
    name = model_name or get_settings().reconstruction_model
    require_model_weights_available(name)
    resolved_custom = resolve_custom_model_checkpoint(name)
    if resolved_custom is not None:
        name = str(resolved_custom)
    cache_key = _model_cache_key(name)
    with _model_lock:
        if cache_key not in _models:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/replay-studio-matplotlib")
            from ultralytics import YOLO

            source = cache_key[0]
            stale_keys = [key for key in _models if key[0] == source]
            for stale_key in stale_keys:
                _models.pop(stale_key, None)
            _models[cache_key] = YOLO(name)
    return _models[cache_key]


# The sampled range is derived from source_end * fps and may therefore point
# one or two indexes past the final materialized frame. Only that rounding
# suffix may be absent; any other missing frame is a truncated generation.
_MISSING_FRAME_SUFFIX_TOLERANCE = 2


def require_source_resolution_analysis_frames(scene: Mapping) -> dict:
    """Fail closed when a Scene still points at the retired 1280px input.

    A Docker rebuild cannot mutate an already published immutable generation.
    Calibration/reconstruction must never quietly consume those legacy pixels;
    the operator explicitly regenerates the asset from its uploaded source.
    """

    video = scene.get("payload", {}).get("videoAsset") or {}
    value = video.get("analysisFrameInput")
    try:
        schema_version = int(value.get("schemaVersion") or 0) if isinstance(value, Mapping) else 0
        width = int(value.get("width") or 0) if isinstance(value, Mapping) else 0
        height = int(value.get("height") or 0) if isinstance(value, Mapping) else 0
    except (TypeError, ValueError):
        schema_version = width = height = 0
    valid = (
        isinstance(value, Mapping)
        and schema_version == 1
        and value.get("source") == "uploaded-video"
        and value.get("coordinateSpace") == "source-video-pixels"
        and value.get("resize") == "none"
        and width > 0
        and height > 0
    )
    if not valid:
        raise ReconstructionError(
            "This Scene still uses a legacy derived-frame generation. "
            "Regenerate source-resolution analysis frames from the calibration "
            "workspace; the cutover invalidates old pixel calibration and does "
            "not start calibration automatically."
        )
    return dict(value)


def _frame_paths_for_rate(
    scene: dict,
    *,
    selected_fps: float,
) -> list[tuple[Path, float]]:
    video = scene["payload"]["videoAsset"]
    require_source_resolution_analysis_frames(scene)
    analysis_fps = float(video.get("analysisFps") or 0.0)
    source_start = float(video.get("sourceStart") or 0.0)
    source_end = float(video.get("sourceEnd") or source_start + scene["duration"])
    # The first sample must lie AT or AFTER the segment start. Truncating
    # down here used to select one frame from before the boundary — and
    # segments are cut at shot changes, so that frame belonged to the
    # previous camera shot and produced a confidently wrong first-frame
    # calibration.
    first = max(1, ceil(source_start * analysis_fps - 1e-6) + 1)
    last = max(first, int(source_end * analysis_fps) + 1)
    generation_key = str(video.get("generationKey") or "")
    if not generation_key:
        raise ReconstructionError("Scene has no published video generation")
    frames = video_generation_directory(str(video["id"]), generation_key) / "frames"
    if selected_fps >= analysis_fps - 1e-3:
        expected_indexes = list(range(first, last + 1))
    else:
        # Integer stride is wrong whenever materialized_fps / selected_fps is
        # not integral (25 -> 10 became 12.5 FPS). Select the nearest available
        # source frame to each exact target timestamp. The first index is still
        # clamped to the first frame at/after the shot boundary, while the
        # alternating later gaps preserve the requested average cadence.
        expected_indexes = []
        sample_number = 0
        while True:
            target_time = source_start + sample_number / selected_fps
            if target_time > source_end + 1e-9:
                break
            index = max(
                first,
                min(last, floor(target_time * analysis_fps + 0.5) + 1),
            )
            if not expected_indexes or index != expected_indexes[-1]:
                expected_indexes.append(index)
            sample_number += 1
    resolved: list[tuple[Path, float]] = []
    missing: list[int] = []
    for index in expected_indexes:
        path = frames / f"frame_{index:05d}.jpg"
        if path.exists():
            resolved.append(
                (path, max(0.0, (index - 1) / analysis_fps - source_start))
            )
        else:
            missing.append(index)
    if not resolved:
        raise ReconstructionError(
            "No sampled frames exist for the published video generation; "
            "re-import the asset before reconstructing"
        )
    if missing:
        tail_start = expected_indexes[-len(missing)]
        is_rounding_tail = (
            len(missing) <= _MISSING_FRAME_SUFFIX_TOLERANCE
            and missing == expected_indexes[-len(missing) :]
            and tail_start > expected_indexes[0]
        )
        if not is_rounding_tail:
            raise ReconstructionError(
                f"The published video generation is missing "
                f"{len(missing)} of {len(expected_indexes)} sampled frames "
                f"(first missing: frame_{missing[0]:05d}.jpg); a silently "
                "thinned sample is not reconstructed — re-import the asset"
            )
    return resolved


def native_frame_paths(scene: dict) -> list[tuple[Path, float]]:
    """Return every materialized source-cadence frame in the scene range.

    This read intentionally includes operator-excluded frames. It is used by
    the exact-frame inspector and by the reversible exclusion command; analysis
    consumers must use :func:`frame_paths` instead.
    """

    video = scene.get("payload", {}).get("videoAsset") or {}
    analysis_fps = float(video.get("analysisFps") or 0.0)
    selected_fps = resolve_sampling_frame_rate(
        scene,
        requested_frame_rate=analysis_fps,
    )
    return _frame_paths_for_rate(scene, selected_fps=selected_fps)


def frame_paths(
    scene: dict,
    *,
    sampling_frame_rate: float | None = None,
) -> list[tuple[Path, float]]:
    """Return the canonical sampled inputs after scene exclusions."""

    selected_fps = resolve_sampling_frame_rate(scene, sampling_frame_rate)
    sampled = _frame_paths_for_rate(scene, selected_fps=selected_fps)
    excluded = excluded_source_frame_indices(scene)
    if not excluded:
        return sampled
    included = [
        (path, scene_time)
        for path, scene_time in sampled
        if source_frame_index(path) not in excluded
    ]
    if not included:
        raise ReconstructionError(
            "Every sampled frame is excluded from this scene segment"
        )
    return included


def source_frame_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])
