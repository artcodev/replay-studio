from app.identity_metrics import evaluate_identity_assignments


def _rows(mapping_by_frame: list[dict[str, str | None]]) -> list[dict]:
    result = []
    for frame_index, mapping in enumerate(mapping_by_frame):
        for ground_truth, predicted in mapping.items():
            result.append(
                {
                    "frameIndex": frame_index,
                    "groundTruthId": ground_truth,
                    "predictedId": predicted,
                }
            )
    return result


def test_missing_labels_never_fabricate_identity_accuracy() -> None:
    result = evaluate_identity_assignments(None)

    assert result["groundTruthAvailable"] is False
    assert result["idf1"] is None
    assert result["hota"] is None
    assert result["gsHota"] is None


def test_arbitrary_but_consistent_labels_score_perfect_idf1() -> None:
    result = evaluate_identity_assignments(
        _rows(
            [
                {"player-a": "canonical-91", "player-b": "canonical-12"},
                {"player-a": "canonical-91", "player-b": "canonical-12"},
            ]
        )
    )

    assert result["idf1"] == 1.0
    assert result["idSwitchCount"] == 0
    assert result["fragmentCount"] == 0
    assert result["globalAssignment"] == {
        "player-a": "canonical-91",
        "player-b": "canonical-12",
    }


def test_smooth_trajectory_identity_swap_fails_identity_metric() -> None:
    # Geometry can remain perfectly smooth while the names swap at frame 2.
    result = evaluate_identity_assignments(
        _rows(
            [
                {"player-a": "left", "player-b": "right"},
                {"player-a": "left", "player-b": "right"},
                {"player-a": "right", "player-b": "left"},
                {"player-a": "right", "player-b": "left"},
            ]
        )
    )

    assert result["idf1"] == 0.5
    assert result["idSwitchCount"] == 2


def test_missing_and_duplicate_assignments_are_counted_explicitly() -> None:
    result = evaluate_identity_assignments(
        [
            {"frameIndex": 0, "groundTruthId": "a", "predictedId": "same"},
            {"frameIndex": 0, "groundTruthId": "b", "predictedId": "same"},
            {"frameIndex": 1, "groundTruthId": "a", "predictedId": None},
            {"frameIndex": 1, "groundTruthId": None, "predictedId": "phantom"},
        ]
    )

    assert result["idFalseNegatives"] == 2
    assert result["idFalsePositives"] == 2
    assert result["duplicateAssignmentFrameCount"] == 1
    assert result["duplicateOverlapSeconds"] is None
    assert result["duplicateOverlapTimebase"] == "frame-index-without-fps"


def test_duplicate_overlap_uses_explicit_frame_rate_for_frame_labels() -> None:
    rows = [
        {"frameIndex": 10, "groundTruthId": "a", "predictedId": "same"},
        {"frameIndex": 10, "groundTruthId": "b", "predictedId": "same"},
        {"frameIndex": 11, "groundTruthId": "a", "predictedId": "a"},
        {"frameIndex": 11, "groundTruthId": "b", "predictedId": "b"},
    ]

    result = evaluate_identity_assignments(rows, frame_rate=10.0)

    assert result["duplicateAssignmentFrameCount"] == 1
    assert result["duplicateOverlapSeconds"] == 0.1
    assert result["duplicateOverlapTimebase"] == "frame-index+explicit-fps"
    assert result["identityAssignmentFrameRate"] == 10.0


def test_scene_time_is_already_seconds_without_a_frame_rate() -> None:
    rows = [
        {"sceneTime": 0.0, "groundTruthId": "a", "predictedId": "same"},
        {"sceneTime": 0.0, "groundTruthId": "b", "predictedId": "same"},
        {"sceneTime": 0.04, "groundTruthId": "a", "predictedId": "a"},
        {"sceneTime": 0.04, "groundTruthId": "b", "predictedId": "b"},
    ]

    result = evaluate_identity_assignments(rows)

    assert result["duplicateOverlapSeconds"] == 0.04
    assert result["duplicateOverlapTimebase"] == "seconds"


def test_scene_time_cadence_is_not_overridden_by_incidental_frame_rate() -> None:
    rows = [
        {"sceneTime": 0.0, "groundTruthId": "a", "predictedId": "same"},
        {"sceneTime": 0.0, "groundTruthId": "b", "predictedId": "same"},
        {"sceneTime": 0.04, "groundTruthId": "a", "predictedId": "a"},
        {"sceneTime": 0.04, "groundTruthId": "b", "predictedId": "b"},
    ]

    result = evaluate_identity_assignments(rows, frame_rate=10.0)

    assert result["duplicateOverlapSeconds"] == 0.04
    assert result["duplicateOverlapTimebase"] == "seconds"


def test_mixed_seconds_and_explicit_frame_indices_share_time_buckets() -> None:
    rows = [
        {"sceneTime": 1.0, "groundTruthId": "a", "predictedId": "same"},
        {"frameIndex": 25, "groundTruthId": "b", "predictedId": "same"},
        {"sceneTime": 2.0, "groundTruthId": "a", "predictedId": "a"},
        {"frameIndex": 50, "groundTruthId": "b", "predictedId": "b"},
    ]

    result = evaluate_identity_assignments(rows, frame_rate=25.0)

    assert result["duplicateAssignmentFrameCount"] == 1
    assert result["duplicateOverlapSeconds"] == 0.04
    assert result["duplicateOverlapTimebase"] == "mixed"


def test_explicit_fps_does_not_infer_overlap_from_sparse_label_cadence() -> None:
    rows = [
        {"frameIndex": 10, "groundTruthId": "a", "predictedId": "same"},
        {"frameIndex": 10, "groundTruthId": "b", "predictedId": "same"},
        {"frameIndex": 20, "groundTruthId": "a", "predictedId": "same"},
        {"frameIndex": 20, "groundTruthId": "b", "predictedId": "same"},
    ]

    result = evaluate_identity_assignments(rows, frame_rate=10.0)

    assert result["duplicateAssignmentFrameCount"] == 2
    assert result["duplicateOverlapSeconds"] == 0.2


def test_single_duplicate_frame_has_duration_when_fps_is_explicit() -> None:
    result = evaluate_identity_assignments(
        [
            {"frameIndex": 10, "groundTruthId": "a", "predictedId": "same"},
            {"frameIndex": 10, "groundTruthId": "b", "predictedId": "same"},
        ],
        frame_rate=25.0,
    )

    assert result["duplicateAssignmentFrameCount"] == 1
    assert result["duplicateOverlapSeconds"] == 0.04


def test_fragment_counts_only_reappearance_after_a_missing_run() -> None:
    result = evaluate_identity_assignments(
        [
            {"frameIndex": 0, "groundTruthId": "a", "predictedId": "same"},
            {"frameIndex": 1, "groundTruthId": "a", "predictedId": None},
            {"frameIndex": 2, "groundTruthId": "a", "predictedId": "same"},
        ]
    )

    assert result["fragmentCount"] == 1
    assert result["idSwitchCount"] == 0


def test_conflicting_duplicate_ground_truth_rows_return_invalid_diagnostic() -> None:
    result = evaluate_identity_assignments(
        [
            {"frameIndex": 0, "groundTruthId": "a", "predictedId": None},
            {"frameIndex": 0, "groundTruthId": "a", "predictedId": "same"},
        ]
    )

    assert result["status"] == "invalid"
    assert result["idf1"] is None
    assert result["invalidAssignmentCount"] == 1
    assert result["invalidAssignments"][0]["predictedIds"] == ["<missing>", "same"]


def test_identical_duplicate_ground_truth_rows_are_deduplicated() -> None:
    row = {"frameIndex": 0, "groundTruthId": "a", "predictedId": "same"}

    result = evaluate_identity_assignments([row, dict(row)])

    assert result["status"] == "evaluated"
    assert result["sampleCount"] == 1
    assert result["idf1"] == 1.0
