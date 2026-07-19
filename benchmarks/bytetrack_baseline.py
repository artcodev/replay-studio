"""ByteTrack baseline over the on-disk person-detection cache.

Reuses the exact cached detections of an already reconstructed segment (zero
video decoding, zero YOLO inference) and runs the `supervision` ByteTrack
tracker over them, printing per-run association counters. This is an offline
aid for the visual acceptance workflow: "our tracker produced 3x more
fragments than ByteTrack on identical detections" is an actionable signal
even without ground truth. It is not a quality benchmark.

Usage (from the repository root):

    pip install supervision  # separate tool dependency, NOT an app/ import
    ./.venv/bin/python benchmarks/bytetrack_baseline.py \
        --frames-dir data/media/<asset>/.pipeline-runs/<generation>/frames \
        --asset-media-dir data/media/<asset> \
        --analysis-fps 10 --step 1

`--step` mirrors the reconstruction sampling stride (analysisFps / sampleFps).
Frames whose detections were never cached are reported and skipped.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from hashlib import sha256
from pathlib import Path


def _frame_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cached_detections(asset_media_dir: Path) -> dict[str, list[dict]]:
    """Map frameContentSha256 -> people from every cached envelope."""

    by_sha: dict[str, tuple[float, list[dict]]] = {}
    cache_root = asset_media_dir / "person-detections"
    for envelope_path in sorted(cache_root.glob("*/*.json")):
        try:
            envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        contract = envelope.get("contract") or {}
        payload = envelope.get("payload") or {}
        frame_sha = str(contract.get("frameContentSha256") or "")
        people = payload.get("people")
        if not frame_sha or not isinstance(people, list):
            continue
        modified_at = envelope_path.stat().st_mtime
        existing = by_sha.get(frame_sha)
        # Several detector contracts may exist for one frame (model change):
        # prefer the most recently written envelope.
        if existing is None or modified_at > existing[0]:
            by_sha[frame_sha] = (modified_at, people)
    return {sha: people for sha, (_, people) in by_sha.items()}


def _detections_to_xyxy(people: list[dict]) -> list[list[float]]:
    boxes: list[list[float]] = []
    for person in people:
        x = float(person["x"])
        y = float(person["y"])
        width = float(person["width"])
        height = float(person["height"])
        boxes.append([x - width / 2, y - height, x + width / 2, y])
    return boxes


def run(
    frames_dir: Path,
    asset_media_dir: Path,
    *,
    analysis_fps: float,
    step: int,
    min_track_length: int,
) -> dict:
    try:
        import numpy as np
        import supervision as sv
    except ImportError as exc:  # separate tool dependency by design
        raise SystemExit(
            "This offline tool needs `pip install supervision numpy` in the "
            f"current environment (missing: {exc.name})."
        )

    cached = _load_cached_detections(asset_media_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))[:: max(1, step)]
    if not frame_paths:
        raise SystemExit(f"No frame_*.jpg files under {frames_dir}")

    tracker = sv.ByteTrack(frame_rate=int(round(analysis_fps)))
    track_frames: dict[int, list[int]] = defaultdict(list)
    missed_frames: list[str] = []
    for frame_index, path in enumerate(frame_paths):
        people = cached.get(_frame_sha256(path))
        if people is None:
            missed_frames.append(path.name)
            continue
        boxes = _detections_to_xyxy(people)
        detections = sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            confidence=np.asarray(
                [float(person["confidence"]) for person in people],
                dtype=np.float32,
            ),
            class_id=np.zeros(len(people), dtype=int),
        )
        tracked = tracker.update_with_detections(detections)
        for tracker_id in tracked.tracker_id or []:
            track_frames[int(tracker_id)].append(frame_index)

    lengths = sorted(len(frames) for frames in track_frames.values())
    stable = [length for length in lengths if length >= min_track_length]
    interior_gaps = sum(
        sum(
            1
            for previous, following in zip(frames, frames[1:])
            if following - previous > 1
        )
        for frames in track_frames.values()
    )
    return {
        "framesConsidered": len(frame_paths),
        "framesWithCachedDetections": len(frame_paths) - len(missed_frames),
        "framesWithoutCache": missed_frames[:10],
        "trackCount": len(lengths),
        "stableTrackCount": len(stable),
        "minTrackLength": min_track_length,
        "meanTrackLength": round(statistics.mean(lengths), 2) if lengths else 0,
        "medianTrackLength": statistics.median(lengths) if lengths else 0,
        "interiorGapCount": interior_gaps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--asset-media-dir", type=Path, required=True)
    parser.add_argument("--analysis-fps", type=float, default=10.0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--min-track-length", type=int, default=5)
    arguments = parser.parse_args()
    report = run(
        arguments.frames_dir,
        arguments.asset_media_dir,
        analysis_fps=arguments.analysis_fps,
        step=arguments.step,
        min_track_length=arguments.min_track_length,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
