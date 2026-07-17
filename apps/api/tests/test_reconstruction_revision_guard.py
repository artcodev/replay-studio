from __future__ import annotations

from copy import deepcopy
from threading import Event, Thread

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.store as store_module
from app.database import Base
from app.reconstruction import (
    IdentityCorrectionError,
    ReconstructionError,
    StaleReconstructionRun,
    queue_reconstruction,
    reconstruct_scene,
    reconstruct_scene_by_id,
)
from app.store import (
    SceneRevisionConflict,
    SceneStore,
    reconstruction_input_fingerprint,
)


def _scene(run_id: str = "run-old") -> dict:
    return {
        "id": "revision-scene",
        "title": "Revision guard",
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "selectedSegmentId": "segment-1",
                "sourceStart": 0.0,
                "sourceEnd": 4.0,
                "analysisFps": 10.0,
                "processingState": "tracks-ready",
                "reconstruction": {
                    "status": "queued",
                    "model": "yolo26m.pt",
                    "runId": run_id,
                    "frameAnnotations": [],
                },
            },
            "teams": [
                {"id": "home", "color": "#ff0000"},
                {"id": "away", "color": "#0000ff"},
            ],
            "tracks": [{"id": "last-good-track", "keyframes": []}],
            "ball": {"keyframes": [{"t": 1.0, "x": 0.0, "z": 0.0}]},
        },
    }


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


def _independent_sqlite_stores(tmp_path) -> tuple[SceneStore, SceneStore]:
    """Return stores backed by different engines/connections to one SQLite file."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'revision-cas.sqlite3'}"
    first_engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    second_engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    Base.metadata.create_all(first_engine)
    first_sessions = sessionmaker(bind=first_engine, expire_on_commit=False)
    second_sessions = sessionmaker(bind=second_engine, expire_on_commit=False)
    return SceneStore(first_sessions), SceneStore(second_sessions)


def test_queue_assigns_unique_run_revision_and_input_fingerprint(monkeypatch):
    scene = _scene()
    scene["payload"]["videoAsset"]["reconstruction"].update(
        {"status": "ready", "runRevision": 7}
    )
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("frame", 0.0)])
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: value)

    first = queue_reconstruction(scene, "yolo26s.pt")
    first_metadata = deepcopy(first["payload"]["videoAsset"]["reconstruction"])
    first["payload"]["videoAsset"]["reconstruction"]["status"] = "ready"
    second = queue_reconstruction(first, "yolo26l.pt")
    second_metadata = second["payload"]["videoAsset"]["reconstruction"]

    assert first_metadata["runId"] != second_metadata["runId"]
    assert first_metadata["runRevision"] == 8
    assert second_metadata["runRevision"] == 9
    assert first_metadata["inputFingerprint"].startswith("sha256:")
    assert first_metadata["inputFingerprint"] != second_metadata["inputFingerprint"]


def test_match_roster_change_invalidates_reconstruction_input() -> None:
    before = _scene()
    before["payload"]["matchBinding"] = {
        "source": "thesportsdb",
        "eventId": "match-a",
        "fetchedAt": "2026-07-17T00:00:00Z",
        "players": [{"id": "player-8", "name": "Player Eight", "number": "8"}],
    }
    after = deepcopy(before)
    after["payload"]["matchBinding"] = {
        "source": "thesportsdb",
        "eventId": "match-b",
        "fetchedAt": "2026-07-17T00:01:00Z",
        "players": [{"id": "player-9", "name": "Player Nine", "number": "9"}],
    }

    assert reconstruction_input_fingerprint(before) != reconstruction_input_fingerprint(after)


def test_identity_review_rejection_invalidates_reconstruction_input() -> None:
    before = _scene()
    after = deepcopy(before)
    after["payload"]["identityReviewDecisions"] = {
        "rosterRejections": [
            {
                "schema": "roster-candidate-rejection-v1",
                "canonicalPersonId": "canonical-1",
                "externalPlayerId": "player-8",
            }
        ]
    }

    assert reconstruction_input_fingerprint(before) != reconstruction_input_fingerprint(after)


def test_stale_publish_cannot_overwrite_newer_manual_correction(
    isolated_store: SceneStore,
):
    current = _scene()
    fingerprint = reconstruction_input_fingerprint(current)
    current["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"] = fingerprint
    isolated_store.put(current)
    stale_result = deepcopy(current)
    stale_result["payload"]["tracks"] = [{"id": "stale-generated-track"}]
    start_publish = Event()
    finished = Event()
    accepted: list[bool] = []

    def publish_old_worker() -> None:
        start_publish.wait(timeout=2)
        accepted.append(
            isolated_store.put_if_reconstruction_run(
                stale_result,
                "run-old",
                fingerprint,
            )
        )
        finished.set()

    thread = Thread(target=publish_old_worker)
    thread.start()
    corrected = isolated_store.get(current["id"])
    assert corrected is not None
    corrected["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] = [
        {"id": "manual-person-10", "action": "confirm"}
    ]
    isolated_store.put(corrected)
    start_publish.set()
    assert finished.wait(timeout=2)
    thread.join(timeout=2)

    saved = isolated_store.get(current["id"])
    assert accepted == [False]
    assert saved is not None
    assert saved["payload"]["tracks"] == current["payload"]["tracks"]
    assert saved["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] == [
        {"id": "manual-person-10", "action": "confirm"}
    ]


def test_two_connection_user_write_wins_before_stale_worker_publish(tmp_path):
    api_store, worker_store = _independent_sqlite_stores(tmp_path)
    queued = _scene()
    fingerprint = reconstruction_input_fingerprint(queued)
    queued["payload"]["videoAsset"]["reconstruction"][
        "inputFingerprint"
    ] = fingerprint
    api_store.put(queued)

    stale_worker = worker_store.get(queued["id"])
    user_edit = api_store.get(queued["id"])
    assert stale_worker is not None and user_edit is not None
    user_edit["title"] = "User correction survives"
    api_store.put(user_edit)

    stale_worker["payload"]["tracks"] = [{"id": "stale-worker-output"}]
    assert worker_store.put_if_reconstruction_run(
        stale_worker,
        "run-old",
        fingerprint,
    ) is False

    saved = worker_store.get(queued["id"])
    assert saved is not None
    assert saved["title"] == "User correction survives"
    assert saved["payload"]["tracks"] == queued["payload"]["tracks"]
    assert saved["revision"] == user_edit["revision"]


def test_two_connection_worker_publish_wins_before_stale_user_write(tmp_path):
    api_store, worker_store = _independent_sqlite_stores(tmp_path)
    queued = _scene()
    fingerprint = reconstruction_input_fingerprint(queued)
    queued["payload"]["videoAsset"]["reconstruction"][
        "inputFingerprint"
    ] = fingerprint
    api_store.put(queued)

    stale_user = api_store.get(queued["id"])
    worker_result = worker_store.get(queued["id"])
    assert stale_user is not None and worker_result is not None
    worker_result["payload"]["tracks"] = [{"id": "current-worker-output"}]
    assert worker_store.put_if_reconstruction_run(
        worker_result,
        "run-old",
        fingerprint,
    ) is True

    stale_user["title"] = "Must not replace worker output"
    with pytest.raises(SceneRevisionConflict):
        api_store.put(stale_user)

    saved = api_store.get(queued["id"])
    assert saved is not None
    assert saved["title"] == queued["title"]
    assert saved["payload"]["tracks"] == [{"id": "current-worker-output"}]
    assert saved["revision"] == worker_result["revision"]


def test_queue_compare_and_swap_rejects_a_stale_scene_snapshot(
    isolated_store: SceneStore,
    monkeypatch,
):
    original = _scene()
    original["payload"]["videoAsset"]["reconstruction"]["status"] = "ready"
    isolated_store.put(original)
    stale_snapshot = deepcopy(original)
    corrected = isolated_store.get(original["id"])
    assert corrected is not None
    corrected["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] = [
        {"id": "newer-manual-input", "action": "exclude"}
    ]
    isolated_store.put(corrected)
    monkeypatch.setattr("app.reconstruction.scene_store", isolated_store)
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("frame", 0.0)])

    with pytest.raises(StaleReconstructionRun):
        queue_reconstruction(stale_snapshot)

    saved = isolated_store.get(original["id"])
    assert saved is not None
    assert saved["payload"]["videoAsset"]["reconstruction"]["status"] == "ready"
    assert saved["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] == [
        {"id": "newer-manual-input", "action": "exclude"}
    ]


def test_stale_background_task_does_not_start_after_a_newer_run(monkeypatch):
    newer = _scene("run-new")
    newer_fingerprint = reconstruction_input_fingerprint(newer)
    newer["payload"]["videoAsset"]["reconstruction"][
        "inputFingerprint"
    ] = newer_fingerprint
    called = []
    monkeypatch.setattr("app.reconstruction.scene_store.get", lambda _: newer)
    monkeypatch.setattr(
        "app.reconstruction.reconstruct_scene",
        lambda *_args, **_kwargs: called.append(True),
    )

    reconstruct_scene_by_id("revision-scene", "run-old", "sha256:old")

    assert called == []
    assert newer["payload"]["videoAsset"]["reconstruction"]["runId"] == "run-new"


def test_failed_run_keeps_last_successful_tracks_and_ball(monkeypatch):
    scene = _scene()
    fingerprint = reconstruction_input_fingerprint(scene)
    scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"] = fingerprint
    persisted = deepcopy(scene)

    def guarded_put(value, run_id, input_fingerprint):
        nonlocal persisted
        assert run_id == "run-old"
        assert input_fingerprint == fingerprint
        persisted = deepcopy(value)
        return True

    monkeypatch.setattr(
        "app.reconstruction.scene_store.put_if_reconstruction_run",
        guarded_put,
    )
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [])

    with pytest.raises(ReconstructionError, match="No sampled frames"):
        reconstruct_scene(
            scene,
            expected_run_id="run-old",
            expected_input_fingerprint=fingerprint,
        )

    assert persisted["payload"]["tracks"] == [
        {"id": "last-good-track", "keyframes": []}
    ]
    assert persisted["payload"]["ball"] == {
        "keyframes": [{"t": 1.0, "x": 0.0, "z": 0.0}]
    }
    assert persisted["payload"]["videoAsset"]["reconstruction"]["status"] == "failed"


def test_worker_keeps_queued_ball_input_when_runtime_environment_changes(monkeypatch):
    scene = _scene()
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    queued_ball_input = {
        "schemaVersion": 1,
        "backend": "dedicated-ultralytics",
        "failurePolicy": "raise",
        "analysisFrameRate": 17.0,
        "maxCandidates": 5,
        "checkpoint": {"name": "queued-ball.pt", "size": 123},
    }
    reconstruction["ballBackend"] = "dedicated-ultralytics"
    reconstruction["ballDetectionInput"] = deepcopy(queued_ball_input)
    fingerprint = reconstruction_input_fingerprint(scene)
    reconstruction["inputFingerprint"] = fingerprint
    persisted = deepcopy(scene)

    def guarded_put(value, run_id, input_fingerprint):
        nonlocal persisted
        assert run_id == "run-old"
        assert input_fingerprint == fingerprint
        persisted = deepcopy(value)
        return True

    monkeypatch.setattr(
        "app.reconstruction.scene_store.put_if_reconstruction_run",
        guarded_put,
    )
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [])
    monkeypatch.setattr(
        "app.reconstruction._ball_detection_input",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("worker recalculated mutable ball input")
        ),
    )

    with pytest.raises(ReconstructionError, match="No sampled frames"):
        reconstruct_scene(
            scene,
            expected_run_id="run-old",
            expected_input_fingerprint=fingerprint,
        )

    saved_reconstruction = persisted["payload"]["videoAsset"]["reconstruction"]
    assert saved_reconstruction["status"] == "failed"
    assert saved_reconstruction["ballDetectionInput"] == queued_ball_input
    assert reconstruction_input_fingerprint(persisted) == fingerprint


def test_identity_remap_failure_publishes_structured_diagnostics_and_keeps_last_good(
    monkeypatch,
):
    scene = _scene()
    fingerprint = reconstruction_input_fingerprint(scene)
    scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"] = fingerprint
    persisted = deepcopy(scene)
    failure = IdentityCorrectionError(
        "Identity correction phantom is ambiguous between nearby trajectories",
        correction_id="phantom",
        action="exclude",
        status="ambiguous",
        reason="nearby-trajectories",
        source_track_id="auto-home-01",
        target_id="auto-home-01",
        candidates=[
            {"rawTrackId": 1, "medianDistanceMetres": 0.2},
            {"rawTrackId": 2, "medianDistanceMetres": 0.3},
        ],
    )

    def guarded_put(value, run_id, input_fingerprint):
        nonlocal persisted
        assert run_id == "run-old"
        assert input_fingerprint == fingerprint
        persisted = deepcopy(value)
        return True

    def fail_with_identity_diagnostic(_scene):
        raise failure

    monkeypatch.setattr(
        "app.reconstruction.scene_store.put_if_reconstruction_run",
        guarded_put,
    )
    monkeypatch.setattr("app.reconstruction._frame_paths", fail_with_identity_diagnostic)

    with pytest.raises(IdentityCorrectionError, match="ambiguous"):
        reconstruct_scene(
            scene,
            expected_run_id="run-old",
            expected_input_fingerprint=fingerprint,
        )

    assert persisted["payload"]["tracks"] == [
        {"id": "last-good-track", "keyframes": []}
    ]
    assert persisted["payload"]["ball"] == {
        "keyframes": [{"t": 1.0, "x": 0.0, "z": 0.0}]
    }
    reconstruction = persisted["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["status"] == "failed"
    assert reconstruction["processingStatus"] == "failed"
    diagnostics = reconstruction["identityCorrectionDiagnostics"]
    assert diagnostics == [failure.diagnostic]
    assert reconstruction["diagnostics"]["identityCorrections"] == diagnostics
    assert reconstruction["progress"]["identityCorrections"] == diagnostics


def test_failure_from_superseded_run_cannot_mark_newer_run_failed(monkeypatch):
    scene = _scene()
    fingerprint = reconstruction_input_fingerprint(scene)
    scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"] = fingerprint
    newer = _scene("run-new")
    newer["payload"]["videoAsset"]["reconstruction"]["status"] = "ready"
    writes = 0

    def guarded_put(_value, _run_id, _input_fingerprint):
        nonlocal writes
        writes += 1
        # Processing status and first progress update belong to the old run;
        # its failure arrives only after a newer run has become authoritative.
        return writes < 3

    monkeypatch.setattr(
        "app.reconstruction.scene_store.put_if_reconstruction_run",
        guarded_put,
    )
    monkeypatch.setattr("app.reconstruction.scene_store.get", lambda _: newer)
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [])

    with pytest.raises(StaleReconstructionRun):
        reconstruct_scene(
            scene,
            expected_run_id="run-old",
            expected_input_fingerprint=fingerprint,
        )

    assert newer["payload"]["videoAsset"]["reconstruction"]["status"] == "ready"
    assert newer["payload"]["videoAsset"]["reconstruction"]["runId"] == "run-new"
