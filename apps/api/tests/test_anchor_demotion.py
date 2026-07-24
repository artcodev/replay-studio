from __future__ import annotations

from pathlib import Path

from app.reconstruction_calibration_resolution import demote_outlier_direct_anchors


def _frames(count: int, cadence: float = 0.1) -> list[tuple[Path, float]]:
    return [
        (Path(f"frame_{index + 42:05d}.jpg"), round(index * cadence, 3))
        for index in range(count)
    ]


def _evidence(p95_by_sample: dict[int, float], count: int) -> list[dict]:
    evidence: list[dict] = []
    for index in range(count):
        entry: dict = {"status": "accepted", "rejectionReasons": []}
        if index in p95_by_sample:
            entry["alignmentMetrics"] = {"residualP95": p95_by_sample[index]}
        evidence.append(entry)
    return evidence


def test_outlier_anchors_are_demoted_and_marked():
    # The measured cohort of this real clip: three tail-heavy anchors must
    # go, the healthy ones stay.
    p95 = {0: 8.595, 10: 5.73, 20: 9.183, 30: 4.649, 40: 3.82, 50: 8.595}
    anchors = {sample: object() for sample in p95}
    evidence = _evidence(p95, 65)

    surviving, demotions = demote_outlier_direct_anchors(
        dict(anchors),
        evidence,
        _frames(65),
        manual_direct={},
        max_gap_seconds=2.0,
        residual_floor_pixels=6.5,
        best_quartile_ratio=1.6,
    )

    # s50 also exceeds the threshold, but demoting it would leave the tail
    # frames without any anchor inside the 2s gap — the coverage guard
    # keeps it.
    assert set(surviving) == {10, 30, 40, 50}
    assert {item["sampleIndex"] for item in demotions} == {0, 20}
    assert evidence[0]["status"] == "rejected"
    assert "direct-anchor-residual-p95-outlier" in evidence[0]["rejectionReasons"]
    assert evidence[10]["status"] == "accepted"
    assert evidence[50]["status"] == "accepted"
    assert all(item["thresholdPixels"] > 6.5 for item in demotions)


def test_demotion_never_strips_the_only_reachable_anchor():
    # Removing the bad tail anchor would leave the final frames without any
    # anchor inside the temporal gap: it must be kept.
    p95 = {0: 4.0, 10: 4.2, 20: 4.1, 60: 30.0}
    anchors = {sample: object() for sample in p95}
    evidence = _evidence(p95, 65)

    surviving, demotions = demote_outlier_direct_anchors(
        dict(anchors),
        evidence,
        _frames(65),
        manual_direct={},
        max_gap_seconds=2.0,
        residual_floor_pixels=6.5,
        best_quartile_ratio=1.6,
    )

    assert 60 in surviving
    assert demotions == []
    assert evidence[60]["status"] == "accepted"


def test_manual_anchor_coverage_allows_demoting_its_bad_neighbour():
    # Same layout, but a manual anchor keeps the tail covered, so the bad
    # automatic anchor can be demoted safely. Manual anchors themselves are
    # never demotion candidates.
    p95 = {0: 4.0, 10: 4.2, 20: 4.1, 60: 30.0}
    anchors = {sample: object() for sample in p95}
    evidence = _evidence(p95, 65)

    surviving, demotions = demote_outlier_direct_anchors(
        dict(anchors),
        evidence,
        _frames(65),
        manual_direct={58: object()},
        max_gap_seconds=2.0,
        residual_floor_pixels=6.5,
        best_quartile_ratio=1.6,
    )

    assert 60 not in surviving
    assert {item["sampleIndex"] for item in demotions} == {60}


def test_small_or_disabled_cohorts_are_untouched():
    p95 = {0: 30.0, 10: 4.0}
    anchors = {sample: object() for sample in p95}
    evidence = _evidence(p95, 20)

    surviving, demotions = demote_outlier_direct_anchors(
        dict(anchors),
        evidence,
        _frames(20),
        manual_direct={},
        max_gap_seconds=2.0,
        residual_floor_pixels=6.5,
        best_quartile_ratio=1.6,
    )
    assert set(surviving) == {0, 10} and demotions == []

    p95 = {0: 30.0, 10: 4.0, 20: 4.0, 30: 4.0}
    surviving, demotions = demote_outlier_direct_anchors(
        {sample: object() for sample in p95},
        _evidence(p95, 40),
        _frames(40),
        manual_direct={},
        max_gap_seconds=2.0,
        residual_floor_pixels=6.5,
        best_quartile_ratio=0.0,
    )
    assert len(surviving) == 4 and demotions == []
