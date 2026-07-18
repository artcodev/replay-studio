from __future__ import annotations

from app.sample import make_video_scene


def test_video_scene_starts_without_fabricated_tracks() -> None:
    scene = make_video_scene(
        scene_id="video-test",
        title="Source clip",
        duration=8.25,
        video_asset={"id": "asset-test", "mediaUrl": "/media"},
    )

    assert scene["duration"] == 8.25
    assert scene["payload"]["tracks"] == []
    assert scene["payload"]["videoAsset"]["id"] == "asset-test"
