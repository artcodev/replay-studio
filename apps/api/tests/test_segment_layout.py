from app.segment_layout import build_segment_layout


def test_score_changes_group_highlight_shots_around_goal_events():
    segments = [
        {"id": f"shot-{index:02d}", "start": start, "end": end}
        for index, (start, end) in enumerate(
            [(0, 4), (4, 10), (10, 17), (17, 23), (23, 29), (29, 34), (34, 37), (37, 42), (42, 46), (46, 50), (50, 54)],
            start=1,
        )
    ]
    costs = {
        ("shot-02", "shot-03"): 0.05,
        ("shot-04", "shot-05"): 0.07,
        ("shot-05", "shot-06"): 0.08,
        ("shot-08", "shot-09"): 0.09,
        ("shot-09", "shot-10"): 0.08,
        ("shot-10", "shot-11"): 0.08,
    }

    layout = build_segment_layout(segments, [9, 23, 50], costs, scoreboard_coverage=0.9)

    assert [group["segmentIds"] for group in layout["groups"]] == [
        ["shot-01", "shot-02", "shot-03"],
        ["shot-04", "shot-05", "shot-06", "shot-07"],
        ["shot-08", "shot-09", "shot-10", "shot-11"],
    ]
    assert [segment["layout"]["label"] for segment in segments] == [
        "1-A", "1-B", "1-C", "2-A", "2-B", "2-C", "2-D", "3-A", "3-B", "3-C", "3-D"
    ]
    assert segments[2]["layout"]["role"] == "replay"
    assert segments[6]["layout"]["role"] == "continuation"


def test_layout_falls_back_to_reviewable_shot_order_groups():
    segments = [
        {"id": f"shot-{index}", "start": float(index), "end": float(index + 1)}
        for index in range(1, 8)
    ]

    layout = build_segment_layout(segments, [])

    assert layout["method"] == "shot-order-fallback"
    assert [group["segmentIds"] for group in layout["groups"]] == [
        ["shot-1", "shot-2", "shot-3"],
        ["shot-4", "shot-5", "shot-6"],
        ["shot-7"],
    ]
