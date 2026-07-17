from __future__ import annotations

import asyncio
from copy import deepcopy

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.store as store_module
from app.database import Base
from app.main import _migrate_legacy_project_match_bindings, app
from app.multi_pass import create_multi_pass_scene
from app.reconstruction import ReconstructionError, queue_reconstruction
from app.sample import make_video_scene
from app.schemas import EventBundle
from app.store import (
    RECONSTRUCTION_INPUT_CHANGED_ERROR,
    SceneRevisionConflict,
    SceneStore,
    reconstruction_input_fingerprint,
)
from app.video_processing import materialize_segment_scene


def _root() -> dict:
    return make_video_scene(
        scene_id="project-root",
        title="Spain vs Belgium",
        duration=12.0,
        video_asset={
            "id": "asset-project",
            "filename": "match.mp4",
            "originalName": "match.mp4",
            "analysisFps": 10.0,
            "processingState": "frames-ready",
            "segments": [],
        },
    )


def _child(scene_id: str, segment_id: str, *, status: str = "ready") -> dict:
    scene = make_video_scene(
        scene_id=scene_id,
        title=scene_id,
        duration=4.0,
        video_asset={
            "id": "asset-project",
            "filename": "match.mp4",
            "originalName": "match.mp4",
            "analysisFps": 10.0,
            "sourceStart": 0.0,
            "sourceEnd": 4.0,
            "parentSceneId": "project-root",
            "selectedSegmentId": segment_id,
            "processingState": "tracks-ready",
            "segments": [],
            "reconstruction": {
                "status": status,
                "model": "yolo26m.pt",
                "runId": f"run-{scene_id}",
                "runRevision": 2,
                "frameAnnotations": [],
            },
        },
    )
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["inputFingerprint"] = reconstruction_input_fingerprint(scene)
    return scene


def _multi_pass_child(
    scene_id: str = "multi-project",
    *,
    status: str = "ready",
) -> dict:
    scene = _child(scene_id, "segment-reference", status=status)
    video = scene["payload"]["videoAsset"]
    video["multiPass"] = {
        "id": "angles-project",
        "status": "ready",
        "matchBindingState": "current",
        "parentSceneId": "project-root",
        "selectedSegmentIds": ["segment-1", "segment-2"],
        "referenceSceneId": "shot-1",
        "passes": [],
        "warnings": [],
    }
    video["processingState"] = "multi-pass-ready"
    scene["payload"]["tracks"] = [{"id": "composite-track"}]
    video["reconstruction"]["inputFingerprint"] = reconstruction_input_fingerprint(
        scene
    )
    return scene


def _bundle() -> EventBundle:
    return EventBundle.model_validate(
        {
            "source": "thesportsdb",
            "event": {
                "id": "event-spain-belgium",
                "name": "Spain vs Belgium",
                "home": {"id": "spain", "name": "Spain"},
                "away": {"id": "belgium", "name": "Belgium"},
            },
            "players": [
                {
                    "id": "player-1",
                    "name": "Player One",
                    "team_id": "spain",
                    "number": "1",
                }
            ],
            "fetched_at": "2026-07-17T12:00:00Z",
        }
    )


def _manual_binding(player_count: int = 52) -> dict:
    return {
        "schemaVersion": 2,
        "source": "manual",
        "eventId": "manual-spain-belgium",
        "event": {
            "id": "manual-spain-belgium",
            "name": "Spain vs Belgium",
            "home": {"id": "spain", "name": "Spain"},
            "away": {"id": "belgium", "name": "Belgium"},
        },
        "teams": {
            "home": {"id": "spain", "name": "Spain"},
            "away": {"id": "belgium", "name": "Belgium"},
        },
        "players": [
            {
                "id": f"manual-player-{index}",
                "name": f"Manual Player {index}",
                "team_id": "spain" if index < player_count / 2 else "belgium",
            }
            for index in range(player_count)
        ],
        "lineup": [],
        "timeline": [],
        "substitutions": [],
        "rosterQuality": {
            "status": "automatic-ready",
            "playerCount": player_count,
            "homePlayerCount": player_count // 2,
            "awayPlayerCount": player_count // 2,
            "automaticIdentityEligible": True,
            "manualIdentityEligible": True,
            "reasons": [],
        },
        "fetchedAt": "2026-07-17T12:00:00Z",
        "provenance": {"kind": "manual-json"},
    }


async def _async_request(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _request(method: str, path: str, **kwargs):
    return asyncio.run(_async_request(method, path, **kwargs))


@pytest.fixture
def isolated_store(monkeypatch) -> SceneStore:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(store_module, "SessionLocal", session_local)
    return SceneStore()


def test_binding_from_child_updates_root_and_every_sibling_atomically(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    root = isolated_store.put(_root())
    first = isolated_store.put(_child("shot-1", "segment-1"))
    second = isolated_store.put(_child("shot-2", "segment-2"))
    background_runs: list[str] = []

    async def event_bundle(_event_id: str):
        return _bundle()

    monkeypatch.setattr("app.main.sports_provider.event_bundle", event_bundle)
    monkeypatch.setattr(
        "app.main.reconstruct_scene_by_id",
        lambda scene_id, *_: background_runs.append(scene_id),
    )
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("frame", 0.0)])

    response = _request(
        "POST",
        "/api/scenes/shot-1/match-binding",
        json={"event_id": "event-spain-belgium"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["scene"]["id"] == "shot-1"
    saved_root = isolated_store.get(root["id"])
    saved_first = isolated_store.get(first["id"])
    saved_second = isolated_store.get(second["id"])
    assert saved_root is not None and saved_first is not None and saved_second is not None
    assert saved_root["payload"]["matchBinding"]["inherited"] is False
    assert saved_first["payload"]["matchBinding"]["inherited"] is True
    assert saved_second["payload"]["matchBinding"]["inherited"] is True
    for scene in (saved_root, saved_first, saved_second):
        binding = scene["payload"]["matchBinding"]
        assert binding["scope"] == "project"
        assert binding["projectSceneId"] == "project-root"
        assert binding["eventId"] == "event-spain-belgium"
        assert scene["payload"]["teams"][0]["name"] == "Spain"
        assert scene["payload"]["teams"][1]["name"] == "Belgium"
    assert saved_first["payload"]["videoAsset"]["reconstruction"]["status"] == "queued"
    assert saved_second["payload"]["videoAsset"]["reconstruction"]["status"] == "queued"
    assert set(background_runs) == {"shot-1", "shot-2"}


def test_project_binding_never_queues_multi_pass_as_single_pass(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    root = isolated_store.put(_root())
    isolated_store.put(_child("shot-1", "segment-1"))
    multi = isolated_store.put(_multi_pass_child())
    background_runs: list[str] = []

    async def event_bundle(_event_id: str):
        return _bundle()

    monkeypatch.setattr("app.main.sports_provider.event_bundle", event_bundle)
    monkeypatch.setattr(
        "app.main.reconstruct_scene_by_id",
        lambda scene_id, *_: background_runs.append(scene_id),
    )
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("frame", 0.0)])

    response = _request(
        "POST",
        "/api/scenes/shot-1/match-binding",
        json={"event_id": "event-spain-belgium"},
    )

    assert response.status_code == 200, response.text
    saved_multi = isolated_store.get(multi["id"])
    video = saved_multi["payload"]["videoAsset"]
    assert video["reconstruction"]["status"] == "ready"
    assert video["reconstruction"]["qualityVerdict"] == "review"
    assert video["multiPass"]["status"] == "ready"
    assert video["multiPass"]["matchBindingState"] == "stale"
    assert video["multiPass"]["warnings"]
    assert background_runs == ["shot-1"]
    assert saved_multi["payload"]["matchBinding"]["projectSceneId"] == root["id"]


def test_single_pass_reconstruct_endpoint_and_queue_reject_multi_pass(
    isolated_store: SceneStore,
) -> None:
    multi = isolated_store.put(_multi_pass_child(status="failed"))

    response = _request("POST", f"/api/scenes/{multi['id']}/reconstruct")

    assert response.status_code == 409
    assert "multi-pass composite" in response.json()["detail"]
    with pytest.raises(ReconstructionError, match="Multi-pass composites"):
        queue_reconstruction(multi)


def test_single_pass_recovery_monitor_ignores_multi_pass_documents(
    isolated_store: SceneStore,
) -> None:
    multi = _multi_pass_child(status="queued")
    reconstruction = multi["payload"]["videoAsset"]["reconstruction"]
    reconstruction["inputFingerprint"] = reconstruction_input_fingerprint(multi)
    multi = isolated_store.put(multi)
    reconstruction = multi["payload"]["videoAsset"]["reconstruction"]

    assert isolated_store.list_recoverable_reconstruction_runs() == []
    assert isolated_store.fail_unrecoverable_reconstruction_runs() == 0
    assert not isolated_store.claim_reconstruction_run(
        multi["id"],
        reconstruction["runId"],
        reconstruction["inputFingerprint"],
        "single-pass-worker",
    )


def test_active_sibling_rejects_project_binding_before_provider_fetch(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    root = isolated_store.put(_root())
    first = isolated_store.put(_child("shot-1", "segment-1"))
    isolated_store.put(_child("shot-2", "segment-2", status="processing"))
    fetched = False

    async def event_bundle(_event_id: str):
        nonlocal fetched
        fetched = True
        return _bundle()

    monkeypatch.setattr("app.main.sports_provider.event_bundle", event_bundle)
    response = _request(
        "POST",
        "/api/scenes/shot-1/match-binding",
        json={"event_id": "event-spain-belgium"},
    )

    assert response.status_code == 409
    assert fetched is False
    assert isolated_store.get(root["id"])["payload"]["matchBinding"] is None
    assert isolated_store.get(first["id"])["payload"]["matchBinding"] is None


def test_refresh_on_legacy_child_promotes_event_to_project_root(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    root = isolated_store.put(_root())
    legacy = _child("shot-legacy", "segment-legacy")
    legacy["payload"]["matchBinding"] = {
        "source": "thesportsdb",
        "eventId": "event-spain-belgium",
        "fetchedAt": None,
    }
    legacy["payload"]["videoAsset"]["reconstruction"][
        "inputFingerprint"
    ] = reconstruction_input_fingerprint(legacy)
    isolated_store.put(legacy)
    isolated_store.put(_child("shot-sibling", "segment-sibling"))

    async def event_bundle(event_id: str):
        assert event_id == "event-spain-belgium"
        return _bundle()

    monkeypatch.setattr("app.main.sports_provider.event_bundle", event_bundle)
    monkeypatch.setattr("app.main.reconstruct_scene_by_id", lambda *_: None)
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("frame", 0.0)])

    response = _request(
        "POST",
        "/api/scenes/shot-legacy/match-binding/refresh",
    )

    assert response.status_code == 200, response.text
    assert response.json()["scene"]["id"] == "shot-legacy"
    for scene_id in (root["id"], "shot-legacy", "shot-sibling"):
        binding = isolated_store.get(scene_id)["payload"]["matchBinding"]
        assert binding["eventId"] == "event-spain-belgium"
        assert binding["projectSceneId"] == root["id"]
        assert binding["inherited"] is (scene_id != root["id"])


def test_put_many_rolls_back_every_scene_when_one_revision_is_stale(
    isolated_store: SceneStore,
) -> None:
    root = isolated_store.put(_root())
    child = isolated_store.put(_child("shot-1", "segment-1"))
    stale_child = deepcopy(child)

    current_child = isolated_store.get(child["id"])
    current_child["title"] = "Concurrent edit"
    isolated_store.put(current_child)
    root["title"] = "Must roll back"
    stale_child["title"] = "Stale edit"

    with pytest.raises(SceneRevisionConflict):
        isolated_store.put_many([root, stale_child])

    assert isolated_store.get(root["id"])["title"] == "Spain vs Belgium"
    assert isolated_store.get(child["id"])["title"] == "Concurrent edit"


def test_put_many_rolls_back_project_when_a_sibling_has_an_active_lease(
    isolated_store: SceneStore,
) -> None:
    root = isolated_store.put(_root())
    leased = isolated_store.put(_child("shot-leased", "segment-leased", status="queued"))
    reconstruction = leased["payload"]["videoAsset"]["reconstruction"]
    assert isolated_store.claim_reconstruction_run(
        leased["id"],
        reconstruction["runId"],
        reconstruction["inputFingerprint"],
        "worker-active",
        lease_ttl_seconds=60,
    )
    leased = isolated_store.get(leased["id"])
    root["title"] = "Must still roll back"
    leased["title"] = "Must not replace leased scene"

    with pytest.raises(SceneRevisionConflict):
        isolated_store.put_many([root, leased])

    assert isolated_store.get(root["id"])["title"] == "Spain vs Belgium"
    assert isolated_store.get(leased["id"])["title"] == "shot-leased"


def test_startup_migration_promotes_full_manual_child_and_rebuilds_stale_siblings(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    root = isolated_store.put(_root())
    manual = _child("shot-manual", "segment-manual")
    manual["payload"]["matchBinding"] = _manual_binding()
    manual["payload"]["teams"][0].update(
        {"name": "Spain", "externalTeamId": "spain"}
    )
    manual["payload"]["teams"][1].update(
        {"name": "Belgium", "externalTeamId": "belgium"}
    )
    manual["payload"]["videoAsset"]["reconstruction"][
        "inputFingerprint"
    ] = reconstruction_input_fingerprint(manual)
    isolated_store.put(manual)
    partial = _child("shot-partial", "segment-partial")
    partial["payload"]["matchBinding"] = {
        "source": "thesportsdb",
        "eventId": "legacy-provider-event",
        "players": [{"id": f"legacy-{index}"} for index in range(5)],
        "fetchedAt": None,
    }
    partial["payload"]["videoAsset"]["reconstruction"][
        "inputFingerprint"
    ] = reconstruction_input_fingerprint(partial)
    isolated_store.put(partial)
    isolated_store.put(_child("shot-unbound", "segment-unbound"))
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("frame", 0.0)])

    migrated = _migrate_legacy_project_match_bindings()

    assert migrated == [root["id"]]
    for scene_id in (root["id"], "shot-manual", "shot-partial", "shot-unbound"):
        scene = isolated_store.get(scene_id)
        binding = scene["payload"]["matchBinding"]
        assert binding["source"] == "manual"
        assert len(binding["players"]) == 52
        assert binding["scope"] == "project"
        assert binding["projectSceneId"] == root["id"]
        assert binding["inherited"] is (scene_id != root["id"])
    assert (
        isolated_store.get("shot-manual")["payload"]["videoAsset"]["reconstruction"][
            "status"
        ]
        == "ready"
    )
    assert (
        isolated_store.get("shot-partial")["payload"]["videoAsset"]["reconstruction"][
            "status"
        ]
        == "queued"
    )
    assert (
        isolated_store.get("shot-unbound")["payload"]["videoAsset"]["reconstruction"][
            "status"
        ]
        == "queued"
    )


def test_startup_repair_requeues_only_segments_and_restores_last_good_multi_pass(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    canonical = {
        **_manual_binding(),
        "scope": "project",
        "projectSceneId": "project-root",
        "inherited": False,
    }
    root = _root()
    root["payload"]["matchBinding"] = canonical
    isolated_store.put(root)

    segment = _child("shot-fingerprint-failed", "segment-failed", status="failed")
    segment["payload"]["matchBinding"] = {**canonical, "inherited": True}
    segment_reconstruction = segment["payload"]["videoAsset"]["reconstruction"]
    segment_reconstruction["error"] = RECONSTRUCTION_INPUT_CHANGED_ERROR
    segment_reconstruction["inputFingerprint"] = reconstruction_input_fingerprint(
        segment
    )
    old_segment_run_id = segment_reconstruction["runId"]
    isolated_store.put(segment)

    multi = _multi_pass_child("multi-fingerprint-failed", status="failed")
    multi["payload"]["matchBinding"] = {**canonical, "inherited": True}
    multi_reconstruction = multi["payload"]["videoAsset"]["reconstruction"]
    multi_reconstruction.update(
        {
            "error": RECONSTRUCTION_INPUT_CHANGED_ERROR,
            "progress": {
                "phase": "failed",
                "label": "Analysis failed",
                "detail": RECONSTRUCTION_INPUT_CHANGED_ERROR,
                "overallPercent": 0,
                "etaSeconds": 0.0,
            },
            "previousResult": {
                "completedAt": "2026-07-14T12:00:00Z",
                "trackCount": 1,
                "ballSamples": 0,
                "calibrationStatus": "ready",
            },
        }
    )
    multi_reconstruction["inputFingerprint"] = reconstruction_input_fingerprint(multi)
    isolated_store.put(multi)

    progress_residue = _multi_pass_child("multi-progress-residue", status="ready")
    progress_residue["payload"]["matchBinding"] = {
        **canonical,
        "inherited": True,
    }
    residue_reconstruction = progress_residue["payload"]["videoAsset"][
        "reconstruction"
    ]
    residue_reconstruction.update(
        {
            "error": None,
            "qualityVerdict": "review",
            "progress": {
                "phase": "failed",
                "label": "Analysis failed",
                "detail": RECONSTRUCTION_INPUT_CHANGED_ERROR,
                "overallPercent": 0,
                "etaSeconds": 0.0,
            },
        }
    )
    residue_reconstruction["inputFingerprint"] = reconstruction_input_fingerprint(
        progress_residue
    )
    isolated_store.put(progress_residue)
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("frame", 0.0)])

    migrated = _migrate_legacy_project_match_bindings()

    assert migrated == ["project-root"]
    repaired_segment = isolated_store.get(segment["id"])
    repaired_segment_reconstruction = repaired_segment["payload"]["videoAsset"][
        "reconstruction"
    ]
    assert repaired_segment_reconstruction["status"] == "queued"
    assert repaired_segment_reconstruction["runId"] != old_segment_run_id
    assert repaired_segment_reconstruction["inputFingerprint"] == (
        reconstruction_input_fingerprint(repaired_segment)
    )

    repaired_multi = isolated_store.get(multi["id"])
    repaired_multi_video = repaired_multi["payload"]["videoAsset"]
    assert repaired_multi_video["reconstruction"]["status"] == "ready"
    assert repaired_multi_video["reconstruction"]["qualityVerdict"] == "review"
    assert repaired_multi_video["reconstruction"]["error"] is None
    assert repaired_multi_video["processingState"] == "multi-pass-ready"
    assert repaired_multi_video["multiPass"]["matchBindingState"] == "stale"
    assert repaired_multi_video["multiPass"]["warnings"]
    repaired_progress = repaired_multi_video["reconstruction"]["progress"]
    assert repaired_progress["phase"] == "review"
    assert repaired_progress["label"] == "Multi-angle result needs refresh"
    assert "Project match data changed" in repaired_progress["detail"]
    assert repaired_progress["overallPercent"] == 100
    assert repaired_progress["etaSeconds"] == 0.0

    repaired_residue = isolated_store.get(progress_residue["id"])
    residue_video = repaired_residue["payload"]["videoAsset"]
    assert residue_video["reconstruction"]["status"] == "ready"
    assert residue_video["reconstruction"]["error"] is None
    assert residue_video["reconstruction"]["progress"]["phase"] == "review"
    assert residue_video["reconstruction"]["progress"]["overallPercent"] == 100
    assert residue_video["multiPass"]["matchBindingState"] == "stale"
    recoverable_ids = {
        scene_id
        for scene_id, _run_id, _fingerprint in (
            isolated_store.list_recoverable_reconstruction_runs()
        )
    }
    assert recoverable_ids == {segment["id"]}


def test_new_segment_and_multi_pass_inherit_project_binding(
    isolated_store: SceneStore,
) -> None:
    root = _root()
    root["payload"]["matchBinding"] = {
        **_manual_binding(),
        "scope": "project",
        "projectSceneId": root["id"],
        "inherited": False,
    }
    root["payload"]["teams"][0].update(
        {"name": "Spain", "externalTeamId": "spain"}
    )
    root["payload"]["teams"][1].update(
        {"name": "Belgium", "externalTeamId": "belgium"}
    )
    segments = [
        {"id": "segment-a", "label": "A", "start": 0.0, "end": 4.0, "duration": 4.0, "score": 0.9},
        {"id": "segment-b", "label": "B", "start": 4.0, "end": 8.0, "duration": 4.0, "score": 0.8},
    ]
    root["payload"]["videoAsset"]["segments"] = segments
    root = isolated_store.put(root)

    child = materialize_segment_scene(root, segments[0])
    multi = create_multi_pass_scene(root, segments)

    for scene in (child, multi):
        binding = scene["payload"]["matchBinding"]
        assert binding["scope"] == "project"
        assert binding["projectSceneId"] == root["id"]
        assert binding["inherited"] is True
        assert len(binding["players"]) == 52
        assert scene["payload"]["teams"][0]["name"] == "Spain"
        assert scene["payload"]["teams"][1]["name"] == "Belgium"
