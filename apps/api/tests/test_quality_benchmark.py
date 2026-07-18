from copy import deepcopy

import pytest

from app.quality_benchmark_contract import BenchmarkValidationError
from app.quality_benchmark_report import evaluate_benchmark
from app.quality_benchmark_validation import validate_manifest


def _manifest(frames: list[dict]) -> dict:
    return {
        "schemaVersion": "1.0",
        "benchmark": {
            "id": "synthetic-unit",
            "name": "Synthetic unit fixture",
            "status": "labelled",
            "coordinateSystem": {
                "image": "pixels-top-left",
                "pitch": "metres-centre-origin-xz",
            },
        },
        "evaluation": {
            "personIouThreshold": 0.5,
            "ballPointThresholdPx": 10.0,
        },
        "samples": [
            {
                "id": "sample-a",
                "source": {"assetId": "asset-a", "frameRate": 25.0},
                "groundTruth": {"frames": frames},
            }
        ],
    }


def _predictions(frames: list[dict]) -> dict:
    return {
        "schemaVersion": "1.0",
        "benchmarkId": "synthetic-unit",
        "run": {"id": "run-a", "pipelineVersion": "test"},
        "samples": [{"sampleId": "sample-a", "frames": frames}],
    }


def _person(identity: str, bbox: list[float]) -> dict:
    return {"id": identity, "bbox": bbox}


def _prediction(identity: str, bbox: list[float], confidence: float = 0.9) -> dict:
    return {"trackId": identity, "bbox": bbox, "confidence": confidence}


def test_perfect_labelled_sample_scores_only_supported_metrics() -> None:
    manifest = _manifest(
        [
            {
                "frameIndex": 0,
                "persons": [
                    _person("home-7", [10, 10, 20, 40]),
                    _person("away-4", [80, 10, 20, 40]),
                ],
                "ball": {"visible": True, "center": [55, 45]},
                "calibrationPoints": [
                    {"id": "centre", "image": [50, 50], "pitch": [0, 0]},
                    {"id": "box", "image": [90, 70], "pitch": [36, 20]},
                ],
            },
            {
                "frameIndex": 1,
                "persons": [
                    _person("home-7", [12, 10, 20, 40]),
                    _person("away-4", [78, 10, 20, 40]),
                ],
                "ball": {"visible": True, "center": [57, 44]},
            },
        ]
    )
    predictions = _predictions(
        [
            {
                "frameIndex": 0,
                "persons": [
                    _prediction("track-red", [10, 10, 20, 40]),
                    _prediction("track-blue", [80, 10, 20, 40]),
                ],
                "ball": {"center": [55, 45], "confidence": 0.9},
                "calibrationPoints": [
                    {"id": "centre", "image": [50, 50], "pitch": [0, 0]},
                    {"id": "box", "image": [90, 70], "pitch": [36, 20]},
                ],
            },
            {
                "frameIndex": 1,
                "persons": [
                    _prediction("track-red", [12, 10, 20, 40]),
                    _prediction("track-blue", [78, 10, 20, 40]),
                ],
                "ball": {"center": [57, 44], "confidence": 0.8},
            },
        ]
    )

    report = evaluate_benchmark(manifest, predictions)
    metrics = report["metrics"]

    assert report["status"] == "evaluated"
    assert metrics["personDetection"]["precision"] == 1.0
    assert metrics["personDetection"]["recall"] == 1.0
    assert metrics["personDetection"]["averagePrecisionAtIou"] == 1.0
    assert metrics["calibration"]["reprojectionError"]["maximum"] == 0.0
    assert metrics["calibration"]["metricProjectionError"]["maximum"] == 0.0
    assert metrics["ball"]["recall"] == 1.0
    assert metrics["ball"]["pointError"]["maximum"] == 0.0
    assert metrics["identity"]["idf1"] == 1.0
    assert metrics["identity"]["fragmentCount"] == 0
    assert metrics["identity"]["hota"] is None
    assert metrics["identity"]["gsHota"] is None


def test_person_detection_counts_miss_false_positive_and_confidence_ranked_ap() -> None:
    manifest = _manifest(
        [{"frameIndex": 0, "persons": [_person("p1", [0, 0, 10, 10])]}]
    )
    predictions = _predictions(
        [
            {
                "frameIndex": 0,
                "persons": [
                    _prediction("phantom", [50, 50, 10, 10], confidence=0.99),
                    _prediction("real", [0, 0, 10, 10], confidence=0.80),
                ],
            }
        ]
    )

    detection = evaluate_benchmark(manifest, predictions)["metrics"]["personDetection"]

    assert detection["truePositive"] == 1
    assert detection["falsePositive"] == 1
    assert detection["falseNegative"] == 0
    assert detection["precision"] == 0.5
    assert detection["recall"] == 1.0
    assert detection["averagePrecisionAtIou"] == 0.5


def test_calibration_reports_sparse_coverage_and_both_error_spaces() -> None:
    manifest = _manifest(
        [
            {
                "frameIndex": 3,
                "calibrationPoints": [
                    {"id": "a", "image": [10, 20], "pitch": [0, 0]},
                    {"id": "b", "image": [50, 70], "pitch": [20, 30]},
                ],
            }
        ]
    )
    predictions = _predictions(
        [
            {
                "frameIndex": 3,
                "calibrationPoints": [
                    {"id": "a", "image": [13, 24], "pitch": [3, 4]},
                ],
            }
        ]
    )

    calibration = evaluate_benchmark(manifest, predictions)["metrics"]["calibration"]

    assert calibration["groundTruthPointCount"] == 2
    assert calibration["reprojectionCoverage"] == 0.5
    assert calibration["metricProjectionCoverage"] == 0.5
    assert calibration["reprojectionError"]["mean"] == 5.0
    assert calibration["metricProjectionError"]["mean"] == 5.0


def test_ball_far_prediction_is_both_false_positive_and_miss() -> None:
    manifest = _manifest(
        [
            {"frameIndex": 0, "ball": {"visible": True, "center": [0, 0]}},
            {"frameIndex": 1, "ball": {"visible": False}},
        ]
    )
    predictions = _predictions(
        [
            {"frameIndex": 0, "ball": {"center": [30, 40], "confidence": 0.9}},
            {"frameIndex": 1, "ball": {"center": [4, 5], "confidence": 0.8}},
        ]
    )

    ball = evaluate_benchmark(manifest, predictions)["metrics"]["ball"]

    assert ball["visibleGroundTruthCount"] == 1
    assert ball["truePositive"] == 0
    assert ball["falseNegative"] == 1
    assert ball["falsePositive"] == 2
    assert ball["precision"] == 0.0
    assert ball["recall"] == 0.0
    assert ball["pointError"]["mean"] == 50.0


def test_identity_fragmentation_is_derived_after_spatial_matching() -> None:
    manifest = _manifest(
        [
            {"frameIndex": 0, "persons": [_person("player-a", [0, 0, 10, 20])]},
            {"frameIndex": 1, "persons": [_person("player-a", [1, 0, 10, 20])]},
            {"frameIndex": 2, "persons": [_person("player-a", [2, 0, 10, 20])]},
        ]
    )
    predictions = _predictions(
        [
            {"frameIndex": 0, "persons": [_prediction("track-9", [0, 0, 10, 20])]},
            {"frameIndex": 1, "persons": []},
            {"frameIndex": 2, "persons": [_prediction("track-9", [2, 0, 10, 20])]},
        ]
    )

    identity = evaluate_benchmark(manifest, predictions)["metrics"]["identity"]

    assert identity["idf1"] == pytest.approx(0.8)
    assert identity["fragmentCount"] == 1
    assert identity["fragmentationRate"] == 1.0
    assert identity["idSwitchCount"] == 0


def test_empty_draft_is_unavailable_instead_of_passing() -> None:
    manifest = _manifest([])
    manifest["benchmark"]["status"] = "draft"

    report = evaluate_benchmark(manifest, _predictions([]))

    assert report["status"] == "unavailable"
    assert report["benchmarkStatus"] == "draft"
    assert report["metrics"]["personDetection"]["precision"] is None
    assert report["metrics"]["ball"]["recall"] is None
    assert report["metrics"]["calibration"]["reprojectionError"]["mean"] is None
    assert report["metrics"]["identity"]["idf1"] is None


def test_manifest_rejects_ambiguous_duplicate_frames_and_unsupported_version() -> None:
    duplicate = _manifest([{"frameIndex": 0}, {"frameIndex": 0}])

    with pytest.raises(BenchmarkValidationError, match="duplicate frameIndex"):
        validate_manifest(duplicate)

    unsupported = deepcopy(_manifest([]))
    unsupported["schemaVersion"] = "2.0"
    with pytest.raises(BenchmarkValidationError, match="Unsupported benchmark"):
        validate_manifest(unsupported)


def test_predictions_for_unknown_sample_are_rejected() -> None:
    predictions = _predictions([])
    predictions["samples"].append({"sampleId": "not-in-manifest", "frames": []})

    with pytest.raises(BenchmarkValidationError, match="outside the manifest"):
        evaluate_benchmark(_manifest([]), predictions)
