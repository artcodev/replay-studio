import numpy as np

from app.reconstruction import (
    Detection,
    TrackState,
    _merge_raw_track_states,
    _refresh_split_track_state,
)


def _detection(
    frame_index: int,
    *,
    sharpness: float,
    vector: tuple[float, float],
    confidence: float = 0.9,
    evidence_fingerprint: str | None = None,
) -> Detection:
    return Detection(
        x=100.0 + frame_index,
        y=200.0,
        width=32.0,
        height=84.0,
        confidence=confidence,
        feature=np.zeros(12, dtype=np.float32),
        source_frame_index=frame_index,
        observation_id=f"observation-{frame_index}",
        reid_feature=np.asarray(vector, dtype=np.float32),
        reid_evidence_fingerprint=evidence_fingerprint,
        reid_quality={
            "cropWidth": 32,
            "cropHeight": 84,
            "sharpness": sharpness,
            "borderClipped": False,
        },
    )


def test_later_sharp_views_replace_early_blurry_first_twelve() -> None:
    track = TrackState(id=1)
    for frame_index in range(12):
        track.append(
            _detection(frame_index, sharpness=1.0, vector=(1.0, 0.0)),
            frame_index,
            frame_index * 0.5,
        )
    for frame_index in range(12, 15):
        track.append(
            _detection(frame_index, sharpness=500.0, vector=(0.0, 1.0)),
            frame_index,
            frame_index * 0.5,
        )

    assert track.reid_observation_count == 15
    assert track.reid_feature_count == 12
    selected_frames = {item["frameIndex"] for item in track.reid_selected_metadata}
    assert {12, 13, 14}.issubset(selected_frames)
    assert len(selected_frames.intersection(range(12))) == 9


def test_reid_samples_are_temporally_independent_and_best_crop_wins() -> None:
    track = TrackState(id=2)
    track.append(_detection(1, sharpness=1.0, vector=(1.0, 0.0)), 1, 0.10)
    track.append(_detection(2, sharpness=500.0, vector=(0.0, 1.0)), 2, 0.20)
    track.append(_detection(3, sharpness=500.0, vector=(1.0, 0.0)), 3, 0.70)

    assert [item["frameIndex"] for item in track.reid_selected_metadata] == [2, 3]
    assert all(
        right["time"] - left["time"] >= 0.45
        for left, right in zip(
            track.reid_selected_metadata,
            track.reid_selected_metadata[1:],
        )
    )


def test_identical_pixel_fingerprint_is_one_reid_support_even_at_different_times() -> None:
    track = TrackState(id=5)
    track.append(
        _detection(
            1,
            sharpness=100.0,
            vector=(1.0, 0.0),
            evidence_fingerprint="pixel-evidence-v1:same",
        ),
        1,
        0.10,
    )
    track.append(
        _detection(
            20,
            sharpness=100.0,
            vector=(1.0, 0.0),
            evidence_fingerprint="pixel-evidence-v1:same",
        ),
        20,
        2.00,
    )

    assert track.reid_observation_count == 1
    assert track.reid_feature_count == 1
    assert track.reid_duplicate_evidence_count == 1
    assert len(track.reid_evidence_fingerprints) == 1


def test_manual_track_merge_deduplicates_same_pixel_fingerprint() -> None:
    target = TrackState(id=6)
    source = TrackState(id=7)
    target.append(
        _detection(
            1,
            sharpness=100.0,
            vector=(1.0, 0.0),
            evidence_fingerprint="pixel-evidence-v1:same",
        ),
        1,
        0.10,
    )
    source.append(
        _detection(
            20,
            sharpness=100.0,
            vector=(1.0, 0.0),
            evidence_fingerprint="pixel-evidence-v1:same",
        ),
        20,
        2.00,
    )

    _merge_raw_track_states(target, source)

    assert target.reid_observation_count == 1
    assert target.reid_feature_count == 1
    assert target.reid_duplicate_evidence_count == 1


def test_split_partition_drops_reid_samples_from_other_observations() -> None:
    track = TrackState(id=3)
    for frame_index in range(4):
        track.append(
            _detection(frame_index, sharpness=100.0, vector=(1.0, float(frame_index))),
            frame_index,
            frame_index * 0.5,
        )

    track.points = track.points[2:]
    _refresh_split_track_state(track)

    assert track.reid_observation_ids == {"observation-2", "observation-3"}
    assert track.reid_observation_count == 2
    assert [item["frameIndex"] for item in track.reid_selected_metadata] == [2, 3]


def test_split_partition_without_point_features_drops_inherited_appearance() -> None:
    track = TrackState(
        id=4,
        feature_sum=np.ones(12, dtype=np.float32),
        feature_count=3,
        points=[
            {
                "t": 0.0,
                "frameIndex": 0,
                "bbox": {"x": 1.0, "y": 2.0, "width": 10.0, "height": 20.0},
            }
        ],
    )

    _refresh_split_track_state(track)

    assert track.feature_sum is None
    assert track.feature_count == 0
