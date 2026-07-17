from app.video_processing import rank_reconstruction_shots


def test_rank_reconstruction_shots_marks_top_eligible_segments():
    segments = [
        {"id": "short", "duration": 3.9, "score": 1.0},
        {"id": "steady", "duration": 6.0, "score": 0.8},
        {"id": "best", "duration": 5.0, "score": 0.95},
        {"id": "third", "duration": 7.0, "score": 0.7},
    ]

    ranked = rank_reconstruction_shots(segments, limit=2)

    assert [segment["id"] for segment in ranked] == ["best", "steady"]
    assert [segment["recommended"] for segment in segments] == [False, True, True, False]
