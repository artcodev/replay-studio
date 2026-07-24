import numpy as np

from app.reconstruction_team_classification import (
    include_goalkeeper_candidates,
    team_clusters,
)
from app.reconstruction_track_state import TrackState


def _feature(*, hue_bin: int | None = None, white: float = 0.0) -> np.ndarray:
    value = np.zeros(12, dtype=np.float32)
    if hue_bin is not None:
        value[hue_bin] = 1.0
    value[8] = white
    value[10] = 0.8
    value[11] = 0.8
    return value


def _track(
    track_id: int,
    samples: list[np.ndarray],
    *,
    pitch_x: float = 0.0,
    screen_x: float = 480.0,
) -> TrackState:
    track = TrackState(id=track_id)
    for index, sample in enumerate(samples):
        track.points.append(
            {
                "t": index * 0.04,
                "px": screen_x,
                "py": 300.0,
                "pitchX": pitch_x,
                "pitchZ": 0.0,
                "_appearanceFeature": sample,
            }
        )
    track.feature_sum = np.sum(samples, axis=0)
    track.feature_count = len(samples)
    return track


def test_team_clustering_trims_frame_level_color_outliers():
    red = _feature(hue_bin=0)
    white = _feature(white=0.9)
    grass_leak = _feature(hue_bin=3)
    tracks = [
        *[
            _track(index, [red] * 8 + [grass_leak] * 2)
            for index in range(1, 4)
        ],
        *[
            _track(index, [white] * 8 + [grass_leak] * 2)
            for index in range(4, 7)
        ],
        _track(7, [_feature(hue_bin=2)] * 10),
    ]
    diagnostics = {}

    mapping, colors = team_clusters(tracks, diagnostics=diagnostics)

    assert {mapping[index] for index in range(1, 4)} == {"home"}
    assert {mapping[index] for index in range(4, 7)} == {"away"}
    assert colors["home"].startswith("#e1")
    assert colors["away"] == "#e8edf2"
    assert diagnostics["method"] == "robust-track-hsv-kmeans"
    assert all(
        item["discardedOutlierCount"] == 2
        for item in diagnostics["assignments"]
    )


def test_goalkeeper_recovery_uses_metric_pitch_side_not_screen_edge():
    home = _track(1, [_feature(hue_bin=0)] * 10, pitch_x=-20.0)
    keeper = _track(
        2,
        [_feature(hue_bin=2)] * 10,
        pitch_x=-49.0,
        # Deliberately central screen position: the old pixel heuristic missed
        # this keeper after a crop/pan despite the calibrated field position.
        screen_x=480.0,
    )

    mapping = include_goalkeeper_candidates(
        [home, keeper],
        {home.id: "home"},
        960,
    )

    assert mapping[keeper.id] == "home"
    assert keeper.role == "goalkeeper"


def test_goalkeeper_recovery_accepts_the_whole_penalty_area_not_only_goal_line():
    home = _track(1, [_feature(hue_bin=0)] * 10, pitch_x=-20.0)
    keeper = _track(2, [_feature(hue_bin=2)] * 10, pitch_x=-38.0)

    mapping = include_goalkeeper_candidates(
        [home, keeper],
        {home.id: "home"},
        960,
    )

    assert mapping[keeper.id] == "home"
    assert keeper.role == "goalkeeper"


def test_small_third_cluster_can_inherit_the_nearest_team_prototype():
    red = _feature(hue_bin=0)
    white = _feature(white=0.90)
    white_variant = _feature(white=0.82)
    tracks = [
        *[_track(index, [red] * 8) for index in range(1, 4)],
        *[_track(index, [white] * 8) for index in range(4, 7)],
        _track(7, [white_variant] * 8),
    ]
    diagnostics = {}

    mapping, _ = team_clusters(tracks, diagnostics=diagnostics)

    assert mapping[7] == mapping[4] == "away"
    row = next(
        item
        for item in diagnostics["assignments"]
        if item["trackletId"] == "tracklet-0007"
    )
    assert row["status"] == "accepted-nearest-team-prototype"
