"""Shadow comparison of a challenger person detector against cached baseline.

The baseline is what production actually saw: the person-detection disk cache
of an already reconstructed segment (exact envelopes, exact confidences). The
challenger is any Ultralytics-format checkpoint — e.g. a football-tuned YOLO —
whose weights the owner downloaded manually; this tool never downloads
weights itself and stays silent about their licensing.

Output is built for the visual acceptance workflow, not for accuracy claims:
agreement counters stratified by bbox height (distant players are the
interesting stratum), timing, and worst-N disagreement overlays where
baseline-only boxes are orange, challenger-only boxes are red and agreements
are green. Whether the red boxes are recall gains or false positives is a
judgement the owner makes by looking at them.

Usage (from the repository root):

    ./.venv/bin/python benchmarks/detector_shadow_comparison.py \
        --frames-dir data/media/<asset>/.pipeline-runs/<generation>/frames \
        --asset-media-dir data/media/<asset> \
        --challenger-weights /path/to/football-yolo.pt \
        --output-dir /tmp/shadow-overlays --step 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from hashlib import sha256
from pathlib import Path


PERSON_CLASS_NAMES = {"person", "player", "goalkeeper", "referee", "staff"}
HEIGHT_STRATA = ((0, 25), (25, 40), (40, 70), (70, 10_000))


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
        if existing is None or modified_at > existing[0]:
            by_sha[frame_sha] = (modified_at, people)
    return {sha: people for sha, (_, people) in by_sha.items()}


def _baseline_boxes(people: list[dict]) -> list[list[float]]:
    boxes: list[list[float]] = []
    for person in people:
        x = float(person["x"])
        y = float(person["y"])
        width = float(person["width"])
        height = float(person["height"])
        boxes.append([x - width / 2, y - height, x + width / 2, y])
    return boxes


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _greedy_match(
    baseline: list[list[float]],
    challenger: list[list[float]],
    iou_threshold: float,
) -> tuple[set[int], set[int]]:
    """Return matched index sets, highest-IoU pairs first."""

    pairs = sorted(
        (
            (_iou(box_a, box_b), index_a, index_b)
            for index_a, box_a in enumerate(baseline)
            for index_b, box_b in enumerate(challenger)
        ),
        key=lambda item: -item[0],
    )
    matched_baseline: set[int] = set()
    matched_challenger: set[int] = set()
    for iou, index_a, index_b in pairs:
        if iou < iou_threshold:
            break
        if index_a in matched_baseline or index_b in matched_challenger:
            continue
        matched_baseline.add(index_a)
        matched_challenger.add(index_b)
    return matched_baseline, matched_challenger


def _stratum(height: float) -> str:
    for low, high in HEIGHT_STRATA:
        if low <= height < high:
            return f"{low}-{high if high < 10_000 else 'inf'}px"
    return "unknown"


def _person_class_ids(names: dict) -> tuple[set[int], dict[int, str]]:
    lowered = {int(index): str(name) for index, name in names.items()}
    person_ids = {
        index
        for index, name in lowered.items()
        if name.lower() in PERSON_CLASS_NAMES
    }
    return person_ids, lowered


def _draw_overlay(image, baseline, challenger, matched_a, matched_b):
    import cv2

    for index, box in enumerate(baseline):
        color = (80, 200, 80) if index in matched_a else (0, 165, 255)
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            1 if index in matched_a else 2,
        )
    for index, box in enumerate(challenger):
        if index in matched_b:
            continue
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 0, 255),
            2,
        )
    legend = "green=agreement  orange=baseline-only  red=challenger-only"
    cv2.putText(
        image,
        legend,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def run(
    frames_dir: Path,
    asset_media_dir: Path,
    challenger_weights: Path,
    *,
    step: int,
    device: str,
    imgsz: int,
    challenger_conf: float,
    iou_match: float,
    output_dir: Path | None,
    overlay_count: int,
) -> dict:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:  # separate tool dependency by design
        raise SystemExit(
            f"This offline tool needs ultralytics + opencv (missing: {exc.name})."
        )
    if not challenger_weights.is_file():
        raise SystemExit(f"Challenger weights not found: {challenger_weights}")

    cached = _load_cached_detections(asset_media_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))[:: max(1, step)]
    if not frame_paths:
        raise SystemExit(f"No frame_*.jpg files under {frames_dir}")

    load_started = time.perf_counter()
    model = YOLO(str(challenger_weights))
    person_ids, class_names = _person_class_ids(model.names or {})
    model_load_seconds = time.perf_counter() - load_started
    class_filter_active = bool(person_ids)

    totals = {
        "framesCompared": 0,
        "framesWithoutCache": 0,
        "baselineBoxCount": 0,
        "challengerBoxCount": 0,
        "matchedCount": 0,
        "baselineOnlyCount": 0,
        "challengerOnlyCount": 0,
    }
    strata: dict[str, dict[str, int]] = {}
    challenger_class_counts: dict[str, int] = {}
    inference_ms: list[float] = []
    disagreements: list[tuple[int, Path, list, list, set, set]] = []

    for path in frame_paths:
        people = cached.get(_frame_sha256(path))
        if people is None:
            totals["framesWithoutCache"] += 1
            continue
        image = cv2.imread(str(path))
        if image is None:
            totals["framesWithoutCache"] += 1
            continue
        inference_started = time.perf_counter()
        result = model.predict(
            image,
            imgsz=imgsz,
            conf=challenger_conf,
            device=device,
            verbose=False,
        )[0]
        inference_ms.append((time.perf_counter() - inference_started) * 1000.0)

        challenger_boxes: list[list[float]] = []
        for box, class_id in zip(
            result.boxes.xyxy.tolist(), result.boxes.cls.tolist()
        ):
            class_name = class_names.get(int(class_id), str(int(class_id)))
            challenger_class_counts[class_name] = (
                challenger_class_counts.get(class_name, 0) + 1
            )
            if class_filter_active and int(class_id) not in person_ids:
                continue
            challenger_boxes.append([float(value) for value in box])

        baseline_boxes = _baseline_boxes(people)
        matched_a, matched_b = _greedy_match(
            baseline_boxes, challenger_boxes, iou_match
        )
        totals["framesCompared"] += 1
        totals["baselineBoxCount"] += len(baseline_boxes)
        totals["challengerBoxCount"] += len(challenger_boxes)
        totals["matchedCount"] += len(matched_a)
        totals["baselineOnlyCount"] += len(baseline_boxes) - len(matched_a)
        totals["challengerOnlyCount"] += len(challenger_boxes) - len(matched_b)

        for index, box in enumerate(baseline_boxes):
            row = strata.setdefault(
                _stratum(box[3] - box[1]),
                {"baseline": 0, "baselineMatched": 0, "challengerOnly": 0},
            )
            row["baseline"] += 1
            row["baselineMatched"] += int(index in matched_a)
        for index, box in enumerate(challenger_boxes):
            if index in matched_b:
                continue
            row = strata.setdefault(
                _stratum(box[3] - box[1]),
                {"baseline": 0, "baselineMatched": 0, "challengerOnly": 0},
            )
            row["challengerOnly"] += 1

        disagreement = (
            len(baseline_boxes)
            - len(matched_a)
            + len(challenger_boxes)
            - len(matched_b)
        )
        if disagreement and output_dir is not None:
            disagreements.append(
                (
                    disagreement,
                    path,
                    baseline_boxes,
                    challenger_boxes,
                    matched_a,
                    matched_b,
                )
            )

    overlay_paths: list[str] = []
    if output_dir is not None and disagreements:
        output_dir.mkdir(parents=True, exist_ok=True)
        disagreements.sort(key=lambda item: -item[0])
        for _count, path, baseline_boxes, challenger_boxes, matched_a, matched_b in (
            disagreements[:overlay_count]
        ):
            image = cv2.imread(str(path))
            if image is None:
                continue
            overlay = _draw_overlay(
                image, baseline_boxes, challenger_boxes, matched_a, matched_b
            )
            target = output_dir / f"{path.stem}_shadow.jpg"
            cv2.imwrite(str(target), overlay)
            overlay_paths.append(str(target))

    for row in strata.values():
        row["baselineMatchedRatio"] = round(
            row["baselineMatched"] / row["baseline"], 3
        ) if row["baseline"] else None
    return {
        **totals,
        "iouMatchThreshold": iou_match,
        "challengerConfThreshold": challenger_conf,
        "imageSize": imgsz,
        "device": device,
        "classFilterActive": class_filter_active,
        "challengerClassCounts": dict(sorted(challenger_class_counts.items())),
        "byBaselineHeightStratum": dict(sorted(strata.items())),
        "timing": {
            "modelLoadSeconds": round(model_load_seconds, 2),
            "meanInferenceMs": (
                round(statistics.mean(inference_ms), 1) if inference_ms else None
            ),
            "p95InferenceMs": (
                round(sorted(inference_ms)[int(0.95 * (len(inference_ms) - 1))], 1)
                if inference_ms
                else None
            ),
        },
        "overlays": overlay_paths,
    }


def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--asset-media-dir", type=Path, required=True)
    parser.add_argument("--challenger-weights", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--challenger-conf", type=float, default=0.25)
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overlay-count", type=int, default=12)
    arguments = parser.parse_args()
    report = run(
        arguments.frames_dir,
        arguments.asset_media_dir,
        arguments.challenger_weights,
        step=arguments.step,
        device=arguments.device,
        imgsz=arguments.imgsz,
        challenger_conf=arguments.challenger_conf,
        iou_match=arguments.iou_match,
        output_dir=arguments.output_dir,
        overlay_count=arguments.overlay_count,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
