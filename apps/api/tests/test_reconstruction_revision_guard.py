from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from threading import Event, Thread

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.scene_repository as scene_repository_module
from app.database import Base, ReconstructionJobRow, ReconstructionLeaseRow
from app.project_models import (
    AnalysisRunRow,
    MatchSnapshotRow,
    ProjectRow,
    ProjectSceneRow,
)
from app.project_match_repository import ProjectMatchRepository
from app.project_resource_repository import ProjectResourceRepository
from app.reconstruction import reconstruct_scene
from app.reconstruction_errors import ReconstructionError, StaleReconstructionRun
from app.reconstruction_errors import IdentityCorrectionError
from app.reconstruction_queue import queue_reconstruction
from app.reconstruction_run_repository import ReconstructionRunRepository
from app.reconstruction_worker import captured_match_snapshot, reconstruct_scene_by_id
from app.scene_document import (
    SceneRevisionConflict,
    reconstruction_input_fingerprint,
)
from app.scene_repository import SceneRepository


@dataclass(frozen=True)
class Persistence:
    documents: SceneRepository
    runs: ReconstructionRunRepository
    sessions: sessionmaker


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
def isolated_store(monkeypatch) -> Persistence:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(scene_repository_module, "SessionLocal", session_local)
    return Persistence(
        documents=SceneRepository(),
        runs=ReconstructionRunRepository(session_local),
        sessions=session_local,
    )


def _independent_sqlite_stores(tmp_path) -> tuple[Persistence, Persistence]:
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
    return (
        Persistence(
            documents=SceneRepository(first_sessions),
            runs=ReconstructionRunRepository(first_sessions),
            sessions=first_sessions,
        ),
        Persistence(
            documents=SceneRepository(second_sessions),
            runs=ReconstructionRunRepository(second_sessions),
            sessions=second_sessions,
        ),
    )


def _put_owned_scene(store: Persistence, scene: dict) -> None:
    """Persist ownership before exposing a queued physical job."""

    initial = deepcopy(scene)
    reconstruction = initial["payload"]["videoAsset"]["reconstruction"]
    was_queued = reconstruction.get("status") == "queued"
    if was_queued:
        initial["payload"]["videoAsset"]["processingState"] = "frames-ready"
        initial["payload"]["videoAsset"]["reconstruction"] = {
            "status": "not-started",
            "model": reconstruction.get("model"),
        }
    initial = store.documents.put(initial)
    project_id = f"project-{scene['id']}"
    with store.sessions() as session:
        session.add(ProjectRow(id=project_id, title=project_id))
        session.add(
            ProjectSceneRow(
                project_id=project_id,
                scene_id=scene["id"],
                role="segment",
            )
        )
        session.commit()
    scene["revision"] = initial["revision"]
    if was_queued:
        store.runs.enqueue_reconstruction(
            scene,
            expected_input_fingerprint=reconstruction_input_fingerprint(initial),
        )


def test_queue_assigns_unique_run_revision_and_input_fingerprint(monkeypatch):
    scene = _scene()
    scene["payload"]["videoAsset"]["reconstruction"].update(
        {"status": "ready", "runRevision": 7}
    )
    monkeypatch.setattr(
        "app.reconstruction_queue.frame_paths", lambda _: [("frame", 0.0)]
    )
    monkeypatch.setattr(
        "app.reconstruction_queue.reconstruction_runs.enqueue_reconstruction",
        lambda value, **_kwargs: value,
    )

    first = queue_reconstruction(scene, "yolo26s.pt", match_snapshot=None)
    first_metadata = deepcopy(first["payload"]["videoAsset"]["reconstruction"])
    first["payload"]["videoAsset"]["reconstruction"]["status"] = "ready"
    second = queue_reconstruction(first, "yolo26l.pt", match_snapshot=None)
    second_metadata = second["payload"]["videoAsset"]["reconstruction"]

    assert first_metadata["runId"] != second_metadata["runId"]
    assert first_metadata["runRevision"] == 8
    assert second_metadata["runRevision"] == 9
    assert first_metadata["inputFingerprint"].startswith("sha256:")
    assert first_metadata["inputFingerprint"] != second_metadata["inputFingerprint"]


def test_match_snapshot_reference_change_invalidates_reconstruction_input() -> None:
    before = _scene()
    before["payload"]["videoAsset"]["reconstruction"]["matchSnapshotRef"] = {
        "id": "snapshot-a",
        "contentHash": "sha256:a",
        "schemaVersion": 1,
    }
    after = deepcopy(before)
    after["payload"]["videoAsset"]["reconstruction"]["matchSnapshotRef"] = {
        "id": "snapshot-b",
        "contentHash": "sha256:b",
        "schemaVersion": 1,
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
    isolated_store: Persistence,
):
    current = _scene()
    fingerprint = reconstruction_input_fingerprint(current)
    current["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"] = fingerprint
    _put_owned_scene(isolated_store, current)
    stale_result = deepcopy(current)
    stale_result["payload"]["tracks"] = [{"id": "stale-generated-track"}]
    start_publish = Event()
    finished = Event()
    accepted: list[bool] = []

    def publish_old_worker() -> None:
        start_publish.wait(timeout=2)
        accepted.append(
            isolated_store.runs.put_if_reconstruction_run(
                stale_result,
                "run-old",
                fingerprint,
            )
        )
        finished.set()

    thread = Thread(target=publish_old_worker)
    thread.start()
    corrected = isolated_store.documents.get(current["id"])
    assert corrected is not None
    corrected["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] = [
        {"id": "manual-person-10", "action": "confirm"}
    ]
    isolated_store.documents.put(corrected)
    start_publish.set()
    assert finished.wait(timeout=2)
    thread.join(timeout=2)

    saved = isolated_store.documents.get(current["id"])
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
    _put_owned_scene(api_store, queued)

    stale_worker = worker_store.documents.get(queued["id"])
    user_edit = api_store.documents.get(queued["id"])
    assert stale_worker is not None and user_edit is not None
    user_edit["title"] = "User correction survives"
    user_edit = api_store.documents.put(user_edit)

    stale_worker["payload"]["tracks"] = [{"id": "stale-worker-output"}]
    assert worker_store.runs.put_if_reconstruction_run(
        stale_worker,
        "run-old",
        fingerprint,
    ) is False

    saved = worker_store.documents.get(queued["id"])
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
    _put_owned_scene(api_store, queued)

    stale_user = api_store.documents.get(queued["id"])
    assert worker_store.runs.claim_reconstruction_run(
        queued["id"],
        "run-old",
        fingerprint,
        "worker-current",
    )
    worker_result = worker_store.documents.get(queued["id"])
    assert stale_user is not None and worker_result is not None
    worker_result["payload"]["tracks"] = [{"id": "current-worker-output"}]
    worker_result["payload"]["videoAsset"]["reconstruction"].update(
        {"status": "ready", "processingStatus": "succeeded"}
    )
    assert worker_store.runs.put_if_reconstruction_run(
        worker_result,
        "run-old",
        fingerprint,
        "worker-current",
    ) is True

    stale_user["title"] = "Must not replace worker output"
    with pytest.raises(SceneRevisionConflict):
        api_store.documents.put(stale_user)

    saved = api_store.documents.get(queued["id"])
    assert saved is not None
    assert saved["title"] == queued["title"]
    assert saved["payload"]["tracks"] == [{"id": "current-worker-output"}]
    assert saved["revision"] == worker_result["revision"]


def test_queue_compare_and_swap_rejects_a_stale_scene_snapshot(
    isolated_store: Persistence,
    monkeypatch,
):
    original = _scene()
    original["payload"]["videoAsset"]["reconstruction"]["status"] = "ready"
    _put_owned_scene(isolated_store, original)
    stale_snapshot = deepcopy(original)
    corrected = isolated_store.documents.get(original["id"])
    assert corrected is not None
    corrected["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] = [
        {"id": "newer-manual-input", "action": "exclude"}
    ]
    isolated_store.documents.put(corrected)
    monkeypatch.setattr(
        "app.reconstruction_queue.reconstruction_runs",
        isolated_store.runs,
    )
    monkeypatch.setattr(
        "app.reconstruction_queue.frame_paths", lambda _: [("frame", 0.0)]
    )

    with pytest.raises(StaleReconstructionRun):
        queue_reconstruction(stale_snapshot, match_snapshot=None)

    saved = isolated_store.documents.get(original["id"])
    assert saved is not None
    assert saved["payload"]["videoAsset"]["reconstruction"]["status"] == "ready"
    assert saved["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] == [
        {"id": "newer-manual-input", "action": "exclude"}
    ]


def test_stale_background_task_does_not_start_after_a_newer_run(monkeypatch):
    called = []
    monkeypatch.setattr(
        "app.reconstruction_worker.reconstruction_runs.reconstruction_run_is_current",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "app.reconstruction.reconstruct_scene",
        lambda *_args, **_kwargs: called.append(True),
    )

    reconstruct_scene_by_id("revision-scene", "run-old", "sha256:old")

    assert called == []


@pytest.mark.parametrize("snapshot_state", ["missing", "hash-mismatch"])
def test_invalid_captured_match_snapshot_fails_run_and_releases_lease(
    isolated_store: Persistence,
    monkeypatch,
    snapshot_state: str,
) -> None:
    scene = _scene("run-invalid-snapshot")
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["matchSnapshotRef"] = {
        "id": "snapshot-input",
        "contentHash": "sha256:expected",
        "schemaVersion": 1,
    }
    fingerprint = reconstruction_input_fingerprint(scene)
    reconstruction["inputFingerprint"] = fingerprint
    _put_owned_scene(isolated_store, scene)
    if snapshot_state == "hash-mismatch":
        with isolated_store.sessions() as session:
            session.add(
                MatchSnapshotRow(
                    id="snapshot-input",
                    project_id="project-revision-scene",
                    provider="test",
                    schema_version=1,
                    content_hash="sha256:actual",
                    is_current=True,
                    payload={},
                )
            )
            session.commit()

    monkeypatch.setattr("app.reconstruction_worker.scenes", isolated_store.documents)
    monkeypatch.setattr(
        "app.reconstruction_worker.reconstruction_runs",
        isolated_store.runs,
    )
    monkeypatch.setattr(
        "app.reconstruction_worker.project_resources",
        ProjectResourceRepository(isolated_store.sessions),
    )
    monkeypatch.setattr(
        "app.reconstruction_worker.project_matches",
        ProjectMatchRepository(isolated_store.sessions),
    )

    assert reconstruct_scene_by_id(
        scene["id"],
        "run-invalid-snapshot",
        fingerprint,
    ) is True

    saved = isolated_store.documents.get(scene["id"])
    assert saved is not None
    saved_reconstruction = saved["payload"]["videoAsset"]["reconstruction"]
    assert saved_reconstruction["status"] == "failed"
    assert "match snapshot" in str(saved_reconstruction["error"]).lower()
    with isolated_store.sessions() as session:
        assert session.get(ReconstructionLeaseRow, scene["id"]) is None
        assert session.get(ReconstructionJobRow, scene["id"]).status == "failed"
        analysis = session.get(AnalysisRunRow, "run-invalid-snapshot")
        assert analysis is not None
        assert analysis.status == "failed"


def test_captured_match_snapshot_uses_compact_scene_owner_not_analysis_run(
    isolated_store: Persistence,
    monkeypatch,
) -> None:
    scene = _scene("run-snapshot-owner")
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["matchSnapshotRef"] = {
        "id": "snapshot-owner",
        "contentHash": "sha256:owner",
        "schemaVersion": 1,
    }
    reconstruction["inputFingerprint"] = reconstruction_input_fingerprint(scene)
    _put_owned_scene(isolated_store, scene)
    with isolated_store.sessions() as session:
        session.add(
            MatchSnapshotRow(
                id="snapshot-owner",
                project_id="project-revision-scene",
                provider="test",
                schema_version=1,
                content_hash="sha256:owner",
                is_current=True,
                payload={"teams": []},
            )
        )
        telemetry = session.get(AnalysisRunRow, "run-snapshot-owner")
        assert telemetry is not None
        session.delete(telemetry)
        session.commit()

    monkeypatch.setattr(
        "app.reconstruction_worker.project_resources",
        ProjectResourceRepository(isolated_store.sessions),
    )
    monkeypatch.setattr(
        "app.reconstruction_worker.project_matches",
        ProjectMatchRepository(isolated_store.sessions),
    )
    queued = isolated_store.documents.get(scene["id"])
    assert queued is not None
    snapshot = captured_match_snapshot(queued)
    assert snapshot is not None
    assert snapshot.id == "snapshot-owner"
    assert snapshot.project_id == "project-revision-scene"


def test_failed_run_keeps_last_successful_tracks_and_ball(monkeypatch):
    scene = _scene()
    fingerprint = reconstruction_input_fingerprint(scene)
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["inputFingerprint"] = fingerprint
    reconstruction["status"] = "processing"
    persisted = deepcopy(scene)

    def guarded_put(value, run_id, input_fingerprint, owner_id):
        nonlocal persisted
        assert run_id == "run-old"
        assert input_fingerprint == fingerprint
        assert owner_id == "worker-test"
        persisted = deepcopy(value)
        return True

    monkeypatch.setattr(
        "app.reconstruction._reconstruction_runs.put_if_reconstruction_run",
        guarded_put,
    )
    monkeypatch.setattr(
        "app.reconstruction._analysis_runtime.publish_reconstruction_progress",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.frame_paths",
        lambda _: [],
    )

    with pytest.raises(ReconstructionError, match="No sampled frames"):
        reconstruct_scene(
            scene,
            expected_run_id="run-old",
            expected_input_fingerprint=fingerprint,
            expected_lease_owner_id="worker-test",
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
    reconstruction["status"] = "processing"
    persisted = deepcopy(scene)

    def guarded_put(value, run_id, input_fingerprint, owner_id):
        nonlocal persisted
        assert run_id == "run-old"
        assert input_fingerprint == fingerprint
        assert owner_id == "worker-test"
        persisted = deepcopy(value)
        return True

    monkeypatch.setattr(
        "app.reconstruction._reconstruction_runs.put_if_reconstruction_run",
        guarded_put,
    )
    monkeypatch.setattr(
        "app.reconstruction._analysis_runtime.publish_reconstruction_progress",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.frame_paths",
        lambda _: [],
    )
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
            expected_lease_owner_id="worker-test",
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
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["inputFingerprint"] = fingerprint
    reconstruction["status"] = "processing"
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

    def guarded_put(value, run_id, input_fingerprint, owner_id):
        nonlocal persisted
        assert run_id == "run-old"
        assert input_fingerprint == fingerprint
        assert owner_id == "worker-test"
        persisted = deepcopy(value)
        return True

    def fail_with_identity_diagnostic(_scene):
        raise failure

    monkeypatch.setattr(
        "app.reconstruction._reconstruction_runs.put_if_reconstruction_run",
        guarded_put,
    )
    monkeypatch.setattr(
        "app.reconstruction._analysis_runtime.publish_reconstruction_progress",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.frame_paths",
        fail_with_identity_diagnostic,
    )

    with pytest.raises(IdentityCorrectionError, match="ambiguous"):
        reconstruct_scene(
            scene,
            expected_run_id="run-old",
            expected_input_fingerprint=fingerprint,
            expected_lease_owner_id="worker-test",
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
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["inputFingerprint"] = fingerprint
    reconstruction["status"] = "processing"
    newer = _scene("run-new")
    newer["payload"]["videoAsset"]["reconstruction"]["status"] = "ready"
    writes = 0

    def guarded_put(_value, _run_id, _input_fingerprint, _owner_id):
        nonlocal writes
        writes += 1
        # Claim owns the processing transition and compact progress never
        # writes Scene. The terminal failure arrives after a newer run became
        # authoritative, so its first dense publication is rejected.
        return False

    monkeypatch.setattr(
        "app.reconstruction._reconstruction_runs.put_if_reconstruction_run",
        guarded_put,
    )
    monkeypatch.setattr(
        "app.reconstruction._analysis_runtime.publish_reconstruction_progress",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.frame_paths",
        lambda _: [],
    )

    with pytest.raises(StaleReconstructionRun):
        reconstruct_scene(
            scene,
            expected_run_id="run-old",
            expected_input_fingerprint=fingerprint,
            expected_lease_owner_id="worker-test",
        )

    assert writes == 1
    assert newer["payload"]["videoAsset"]["reconstruction"]["status"] == "ready"
    assert newer["payload"]["videoAsset"]["reconstruction"]["runId"] == "run-new"
