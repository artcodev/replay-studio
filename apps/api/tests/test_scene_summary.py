from app.store import scene_kind


def test_scene_kind_hides_internal_scenes_from_project_navigation():
    root = {"payload": {"videoAsset": {"id": "asset-1", "segments": []}}}
    segment = {"payload": {"videoAsset": {"id": "asset-1", "parentSceneId": "video-1"}}}
    multi_pass = {
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "parentSceneId": "video-1",
                "multiPass": {"status": "ready"},
            }
        }
    }

    assert scene_kind(root) == "video"
    assert scene_kind(segment) == "segment"
    assert scene_kind(multi_pass) == "multi-pass"
    assert scene_kind({"payload": {}}) == "demo"
    assert scene_kind({"title": "Ingestion smoke test", "payload": {"videoAsset": {"id": "test"}}}) == "demo"
