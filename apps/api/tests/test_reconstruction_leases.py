from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from threading import Barrier, Event, Thread

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

import app.reconstruction as reconstruction_module
import app.reconstruction_recovery as recovery_module
import app.reconstruction_worker as reconstruction_worker_module
from app.database import Base, ReconstructionJobRow, ReconstructionLeaseRow, SceneRow
from app.project_models import AnalysisRunRow, ProjectRow, ProjectSceneRow
from app.reconstruction_errors import StaleReconstructionRun
from app.reconstruction_run_repository import ReconstructionRunRepository
from app.scene_document import SceneRevisionConflict, reconstruction_input_fingerprint
from app.scene_repository import SceneRepository


class MutableClock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


@dataclass(frozen=True)
class Repositories:
    documents: SceneRepository
    runs: ReconstructionRunRepository


def _scene(scene_id: str, status: str = "queued") -> dict:
    scene = {
        "id": scene_id,
        "title": f"Lease {scene_id}",
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "id": f"asset-{scene_id}",
                "selectedSegmentId": "segment-1",
                "sourceStart": 0.0,
                "sourceEnd": 4.0,
                "analysisFps": 10.0,
                "processingState": "reconstructing",
                "reconstruction": {
                    "status": status,
                    "processingStatus": status,
                    "model": "yolo26m.pt",
                    "runId": f"run-{scene_id}",
                    "runRevision": 3,
                    "frameAnnotations": [],
                },
            },
            "tracks": [],
            "ball": {"keyframes": []},
        },
    }
    scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"] = (
        reconstruction_input_fingerprint(scene)
    )
    return scene


def _independent_repositories(tmp_path, clock: MutableClock):
    url = f"sqlite+pysqlite:///{tmp_path / 'leases.sqlite3'}"
    first_engine = create_engine(
        url, connect_args={"check_same_thread": False, "timeout": 5}
    )
    second_engine = create_engine(
        url, connect_args={"check_same_thread": False, "timeout": 5}
    )
    Base.metadata.create_all(first_engine)
    first_sessions = sessionmaker(bind=first_engine, expire_on_commit=False)
    second_sessions = sessionmaker(bind=second_engine, expire_on_commit=False)
    return (
        Repositories(
            documents=SceneRepository(first_sessions, clock=clock),
            runs=ReconstructionRunRepository(
                first_sessions,
                clock=clock,
                reconstruction_lease_ttl_seconds=10,
            ),
        ),
        Repositories(
            documents=SceneRepository(second_sessions, clock=clock),
            runs=ReconstructionRunRepository(
                second_sessions,
                clock=clock,
                reconstruction_lease_ttl_seconds=10,
            ),
        ),
        first_sessions,
    )


def _tokens(scene: dict) -> tuple[str, str]:
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    return reconstruction["runId"], reconstruction["inputFingerprint"]


def _put_owned(repositories: Repositories, sessions, scene: dict) -> None:
    """Create the editor document, ownership, then its atomic queued run."""

    initial = deepcopy(scene)
    video = initial["payload"]["videoAsset"]
    video["processingState"] = "frames-ready"
    video["reconstruction"] = {
        "status": "not-started",
        "model": video["reconstruction"].get("model"),
    }
    initial = repositories.documents.put(initial)
    project_id = f"project-{scene['id']}"
    with sessions.begin() as session:
        session.add(ProjectRow(id=project_id, title=project_id))
        session.add(
            ProjectSceneRow(
                project_id=project_id,
                scene_id=scene["id"],
                role="segment",
            )
        )
    scene["revision"] = initial["revision"]
    repositories.runs.enqueue_reconstruction(
        scene,
        expected_input_fingerprint=reconstruction_input_fingerprint(initial),
    )


def test_reconstruct_scene_requires_complete_fence_without_mutating_database(
    tmp_path,
) -> None:
    clock = MutableClock()
    store, _other, sessions = _independent_repositories(tmp_path, clock)
    queued = _scene("complete-fence")
    _put_owned(store, sessions, queued)
    with sessions() as session:
        before_scene = deepcopy(session.get(SceneRow, queued["id"]).payload)
        before_job = deepcopy(session.get(ReconstructionJobRow, queued["id"]).__dict__)
        before_run = deepcopy(session.get(AnalysisRunRow, "run-complete-fence").__dict__)

    with pytest.raises(TypeError, match="expected_run_id"):
        reconstruction_module.reconstruct_scene(queued)

    with sessions() as session:
        assert session.get(SceneRow, queued["id"]).payload == before_scene
        job = session.get(ReconstructionJobRow, queued["id"])
        run = session.get(AnalysisRunRow, "run-complete-fence")
        assert {
            key: value
            for key, value in job.__dict__.items()
            if not key.startswith("_sa_")
        } == {
            key: value
            for key, value in before_job.items()
            if not key.startswith("_sa_")
        }
        assert {
            key: value
            for key, value in run.__dict__.items()
            if not key.startswith("_sa_")
        } == {
            key: value
            for key, value in before_run.items()
            if not key.startswith("_sa_")
        }


def test_stale_lease_cannot_publish_progress_or_terminal(
    tmp_path,
    monkeypatch,
) -> None:
    clock = MutableClock()
    store, _other, sessions = _independent_repositories(tmp_path, clock)
    queued = _scene("stale-execution")
    _put_owned(store, sessions, queued)
    run_id, fingerprint = _tokens(queued)
    assert store.runs.claim_reconstruction_run(
        queued["id"],
        run_id,
        fingerprint,
        "expired-owner",
    )
    claimed = store.documents.get(queued["id"])
    assert claimed is not None
    with sessions() as session:
        before_scene = deepcopy(session.get(SceneRow, queued["id"]).payload)
        before_progress = deepcopy(
            session.get(AnalysisRunRow, run_id).progress
        )

    clock.value += 11
    frame_reads: list[bool] = []
    monkeypatch.setattr(reconstruction_module, "_reconstruction_runs", store.runs)
    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.frame_paths",
        lambda *_args: frame_reads.append(True) or [],
    )
    with pytest.raises(StaleReconstructionRun):
        reconstruction_module.reconstruct_scene(
            claimed,
            expected_run_id=run_id,
            expected_input_fingerprint=fingerprint,
            expected_lease_owner_id="expired-owner",
        )

    assert frame_reads == []
    with sessions() as session:
        assert session.get(SceneRow, queued["id"]).payload == before_scene
        assert session.get(ReconstructionJobRow, queued["id"]).status == "processing"
        lease = session.get(ReconstructionLeaseRow, queued["id"])
        assert lease is not None
        assert lease.owner_id == "expired-owner"
        analysis = session.get(AnalysisRunRow, run_id)
        assert analysis.status == "running"
        assert analysis.progress == before_progress


def test_two_stores_claim_queued_run_exactly_once(tmp_path):
    clock = MutableClock()
    first, second, sessions = _independent_repositories(tmp_path, clock)
    scene = _scene("queued-race")
    _put_owned(first, sessions, scene)
    run_id, fingerprint = _tokens(scene)
    barrier = Barrier(2)
    results: list[tuple[str, bool]] = []

    def claim(store: Repositories, owner: str) -> None:
        barrier.wait(timeout=2)
        results.append(
            (
                owner,
                store.runs.claim_reconstruction_run(
                    scene["id"], run_id, fingerprint, owner
                ),
            )
        )

    workers = [
        Thread(target=claim, args=(first, "owner-a")),
        Thread(target=claim, args=(second, "owner-b")),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)

    assert sorted(result for _owner, result in results) == [False, True]
    winner = next(owner for owner, result in results if result)
    saved = first.documents.get(scene["id"])
    assert saved is not None
    reconstruction = saved["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["status"] == "processing"
    with sessions() as session:
        raw = session.get(SceneRow, scene["id"])
        lease = session.get(ReconstructionLeaseRow, scene["id"])
        assert raw is not None
        assert lease is not None
        assert lease.owner_id == winner
        assert "lease" not in raw.payload["payload"]["videoAsset"]["reconstruction"]


def test_active_lease_is_not_listed_or_stolen_and_heartbeat_extends_it(tmp_path):
    clock = MutableClock()
    first, second, _sessions = _independent_repositories(tmp_path, clock)
    scene = _scene("active")
    _put_owned(first, _sessions, scene)
    run_id, fingerprint = _tokens(scene)
    assert first.runs.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner-a"
    )

    claimed_revision = first.documents.get(scene["id"])["revision"]
    clock.value += 5
    assert second.runs.list_recoverable_reconstruction_runs() == []
    assert not second.runs.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner-b"
    )
    assert not second.runs.heartbeat_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner-b"
    )
    assert first.runs.heartbeat_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner-a"
    )
    # Heartbeats are deliberately outside the revisioned scene document.
    assert first.documents.get(scene["id"])["revision"] == claimed_revision

    clock.value += 6
    assert second.runs.list_recoverable_reconstruction_runs() == []
    assert not second.runs.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner-b"
    )


def test_idle_scan_liveness_and_heartbeat_never_select_scene_payload(tmp_path):
    clock = MutableClock()
    store, _other, sessions = _independent_repositories(tmp_path, clock)
    scene = _scene("compact-hot-path")
    # Make accidental dense reads visible without making the test expensive.
    scene["payload"]["tracks"] = [{"samples": [0] * 50_000}]
    _put_owned(store, sessions, scene)
    run_id, fingerprint = _tokens(scene)
    assert store.runs.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "compact-owner"
    )

    statements: list[str] = []
    engine = sessions.kw["bind"]

    @event.listens_for(engine, "before_cursor_execute")
    def record_sql(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.lower())

    try:
        assert store.runs.list_recoverable_reconstruction_runs() == []
        assert store.runs.reconstruction_run_is_current(
            scene["id"], run_id, fingerprint
        )
        assert store.runs.heartbeat_reconstruction_run(
            scene["id"], run_id, fingerprint, "compact-owner"
        )
    finally:
        event.remove(engine, "before_cursor_execute", record_sql)

    assert statements
    assert not any(" from scenes" in statement for statement in statements)
    assert not any("scenes.payload" in statement for statement in statements)


@pytest.mark.parametrize("write_many", [False, True])
def test_generic_document_write_cannot_bypass_active_lease(tmp_path, write_many):
    clock = MutableClock()
    store, _other, sessions = _independent_repositories(tmp_path, clock)
    scene = _scene("generic-write")
    _put_owned(store, sessions, scene)
    run_id, fingerprint = _tokens(scene)
    assert store.runs.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner"
    )
    current = store.documents.get(scene["id"])
    assert current is not None
    current["title"] = "must not overwrite worker"
    statements: list[str] = []
    engine = sessions.kw["bind"]

    @event.listens_for(engine, "before_cursor_execute")
    def record_sql(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.lower())

    try:
        with pytest.raises(SceneRevisionConflict):
            if write_many:
                store.documents.put_many([current])
            else:
                store.documents.put(current)
    finally:
        event.remove(engine, "before_cursor_execute", record_sql)

    def select_index(table: str) -> int:
        return next(
            index
            for index, statement in enumerate(statements)
            if statement.lstrip().startswith("select") and table in statement
        )

    assert (
        select_index("reconstruction_jobs")
        < select_index("reconstruction_leases")
        < select_index("scenes.payload")
    )


def test_expired_lease_has_single_reclaimer_and_old_worker_is_fenced(tmp_path):
    clock = MutableClock()
    first, second, _sessions = _independent_repositories(tmp_path, clock)
    scene = _scene("reclaim-race")
    _put_owned(first, _sessions, scene)
    run_id, fingerprint = _tokens(scene)
    assert first.runs.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "crashed-owner"
    )
    old_snapshot = first.documents.get(scene["id"])

    clock.value += 11
    assert first.runs.list_recoverable_reconstruction_runs() == [
        (scene["id"], run_id, fingerprint)
    ]
    barrier = Barrier(2)
    results: list[tuple[str, bool]] = []

    def reclaim(store: Repositories, owner: str) -> None:
        barrier.wait(timeout=2)
        results.append(
            (
                owner,
                store.runs.claim_reconstruction_run(
                    scene["id"], run_id, fingerprint, owner
                ),
            )
        )

    workers = [
        Thread(target=reclaim, args=(first, "replacement-a")),
        Thread(target=reclaim, args=(second, "replacement-b")),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)

    assert sorted(result for _owner, result in results) == [False, True]
    winner = next(owner for owner, result in results if result)
    assert old_snapshot is not None
    old_snapshot["payload"]["tracks"] = [{"id": "stale-output"}]
    assert not first.runs.put_if_reconstruction_run(
        old_snapshot,
        run_id,
        fingerprint,
        "crashed-owner",
    )

    current = second.documents.get(scene["id"])
    assert current is not None
    current["payload"]["tracks"] = [{"id": "winner-output"}]
    current["payload"]["videoAsset"]["reconstruction"]["status"] = "ready"
    assert second.runs.put_if_reconstruction_run(
        current,
        run_id,
        fingerprint,
        winner,
    )
    saved = first.documents.get(scene["id"])
    assert saved is not None
    assert saved["payload"]["tracks"] == [{"id": "winner-output"}]
    assert "lease" not in saved["payload"]["videoAsset"]["reconstruction"]
    assert not first.runs.heartbeat_reconstruction_run(
        scene["id"], run_id, fingerprint, winner
    )


def test_failed_terminal_publish_also_clears_lease(tmp_path):
    clock = MutableClock()
    store, _other, sessions = _independent_repositories(tmp_path, clock)
    scene = _scene("failed-cleanup")
    _put_owned(store, sessions, scene)
    run_id, fingerprint = _tokens(scene)
    assert store.runs.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner"
    )
    current = store.documents.get(scene["id"])
    assert current is not None
    current["payload"]["videoAsset"]["reconstruction"].update(
        {"status": "failed", "processingStatus": "failed", "error": "boom"}
    )
    assert store.runs.put_if_reconstruction_run(
        current, run_id, fingerprint, "owner"
    )
    with sessions() as session:
        assert session.get(ReconstructionLeaseRow, scene["id"]) is None


def test_processing_document_without_explicit_job_tokens_is_not_a_queue(tmp_path):
    clock = MutableClock()
    store, _other, sessions = _independent_repositories(tmp_path, clock)
    scene = _scene("unowned", status="processing")
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction.pop("runId")
    reconstruction.pop("runRevision")
    reconstruction.pop("inputFingerprint")
    store.documents.put(scene)

    assert store.runs.list_recoverable_reconstruction_runs() == []
    with sessions() as session:
        assert session.get(ReconstructionJobRow, scene["id"]) is None


def test_corrupt_generic_scene_state_never_enters_scheduler(tmp_path):
    clock = MutableClock()
    store, _other, sessions = _independent_repositories(tmp_path, clock)
    corrupt = _scene("corrupt", status="processing")
    corrupt["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"] = (
        "sha256:stale"
    )
    store.documents.put(corrupt)
    assert store.runs.list_recoverable_reconstruction_runs() == []
    with sessions() as session:
        assert session.get(ReconstructionJobRow, corrupt["id"]) is None


def test_delayed_duplicate_runner_child_does_not_re_run_terminal_same_run(
    tmp_path, monkeypatch
):
    clock = MutableClock()
    store, _other, _sessions = _independent_repositories(tmp_path, clock)
    ready = _scene("terminal", status="ready")
    store.documents.put(ready)
    run_id, fingerprint = _tokens(ready)
    calls: list[bool] = []
    monkeypatch.setattr(reconstruction_worker_module, "scenes", store.documents)
    monkeypatch.setattr(
        reconstruction_worker_module,
        "reconstruction_runs",
        store.runs,
    )
    monkeypatch.setattr(
        reconstruction_module,
        "reconstruct_scene",
        lambda *_args, **_kwargs: calls.append(True),
    )

    assert not reconstruction_worker_module.reconstruct_scene_by_id(
        ready["id"], run_id, fingerprint
    )
    assert calls == []


def test_by_id_claims_and_propagates_owner_to_terminal_publish(tmp_path, monkeypatch):
    clock = MutableClock()
    store, _other, _sessions = _independent_repositories(tmp_path, clock)
    queued = _scene("by-id")
    _put_owned(store, _sessions, queued)
    run_id, fingerprint = _tokens(queued)
    observed_owners: list[str] = []
    monkeypatch.setattr(reconstruction_worker_module, "scenes", store.documents)
    monkeypatch.setattr(
        reconstruction_worker_module,
        "reconstruction_runs",
        store.runs,
    )

    def reconstruct(
        scene: dict,
        *,
        expected_run_id: str,
        expected_input_fingerprint: str,
        expected_lease_owner_id: str,
        match_snapshot=None,
    ) -> dict:
        assert match_snapshot is None
        observed_owners.append(expected_lease_owner_id)
        scene["payload"]["videoAsset"]["reconstruction"]["status"] = "ready"
        assert store.runs.put_if_reconstruction_run(
            scene,
            expected_run_id,
            expected_input_fingerprint,
            expected_lease_owner_id,
        )
        return scene

    monkeypatch.setattr(reconstruction_module, "reconstruct_scene", reconstruct)
    assert reconstruction_worker_module.reconstruct_scene_by_id(
        queued["id"], run_id, fingerprint
    )
    assert len(observed_owners) == 1
    saved = store.documents.get(queued["id"])
    assert saved is not None
    assert saved["payload"]["videoAsset"]["reconstruction"]["status"] == "ready"
    assert "lease" not in saved["payload"]["videoAsset"]["reconstruction"]


def test_unexpected_by_id_crash_persists_failure_and_clears_lease(
    tmp_path, monkeypatch
):
    clock = MutableClock()
    store, _other, _sessions = _independent_repositories(tmp_path, clock)
    queued = _scene("wrapper-crash")
    _put_owned(store, _sessions, queued)
    run_id, fingerprint = _tokens(queued)
    monkeypatch.setattr(reconstruction_worker_module, "scenes", store.documents)
    monkeypatch.setattr(
        reconstruction_worker_module,
        "reconstruction_runs",
        store.runs,
    )
    monkeypatch.setattr(
        reconstruction_module,
        "reconstruct_scene",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("crash")),
    )

    assert reconstruction_worker_module.reconstruct_scene_by_id(
        queued["id"], run_id, fingerprint
    )
    saved = store.documents.get(queued["id"])
    assert saved is not None
    reconstruction = saved["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["status"] == "failed"
    assert reconstruction["error"] == "Reconstruction worker crashed: crash"
    assert "lease" not in reconstruction


def test_dedicated_runner_scans_repeatedly_not_only_at_startup(monkeypatch):
    calls: list[int] = []
    scanned_twice = Event()

    class EmptyStore:
        def list_recoverable_reconstruction_runs(self):
            calls.append(len(calls) + 1)
            if len(calls) >= 2:
                scanned_twice.set()
            return []

    monkeypatch.setattr(recovery_module, "reconstruction_runs", EmptyStore())
    monitor = recovery_module.DedicatedReconstructionRecoveryMonitor(
        poll_seconds=0.02
    ).start()
    assert scanned_twice.wait(timeout=1)
    monitor.stop()
    assert len(calls) >= 2
    assert not monitor.is_alive()
