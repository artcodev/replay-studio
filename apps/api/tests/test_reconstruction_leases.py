from __future__ import annotations

from threading import Barrier, Event, Thread

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import app.reconstruction as reconstruction_module
import app.reconstruction_recovery as recovery_module
from app.database import Base, ReconstructionLeaseRow, SceneRow
from app.store import (
    SceneRevisionConflict,
    SceneStore,
    reconstruction_input_fingerprint,
)


class MutableClock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


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


def _independent_stores(tmp_path, clock: MutableClock):
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
        SceneStore(
            first_sessions,
            clock=clock,
            reconstruction_lease_ttl_seconds=10,
        ),
        SceneStore(
            second_sessions,
            clock=clock,
            reconstruction_lease_ttl_seconds=10,
        ),
        first_sessions,
    )


def _tokens(scene: dict) -> tuple[str, str]:
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    return reconstruction["runId"], reconstruction["inputFingerprint"]


def test_two_stores_claim_queued_run_exactly_once(tmp_path):
    clock = MutableClock()
    first, second, sessions = _independent_stores(tmp_path, clock)
    scene = _scene("queued-race")
    first.put(scene)
    run_id, fingerprint = _tokens(scene)
    barrier = Barrier(2)
    results: list[tuple[str, bool]] = []

    def claim(store: SceneStore, owner: str) -> None:
        barrier.wait(timeout=2)
        results.append(
            (
                owner,
                store.claim_reconstruction_run(
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
    saved = first.get(scene["id"])
    assert saved is not None
    reconstruction = saved["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["status"] == "processing"
    assert reconstruction["lease"]["ownerId"] == winner
    with sessions() as session:
        raw = session.get(SceneRow, scene["id"])
        assert raw is not None
        assert "lease" not in raw.payload["payload"]["videoAsset"]["reconstruction"]


def test_active_lease_is_not_listed_or_stolen_and_heartbeat_extends_it(tmp_path):
    clock = MutableClock()
    first, second, _sessions = _independent_stores(tmp_path, clock)
    scene = _scene("active")
    first.put(scene)
    run_id, fingerprint = _tokens(scene)
    assert first.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner-a"
    )

    claimed_revision = first.get(scene["id"])["revision"]
    clock.value += 5
    assert second.list_recoverable_reconstruction_runs() == []
    assert not second.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner-b"
    )
    assert not second.heartbeat_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner-b"
    )
    assert first.heartbeat_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner-a"
    )
    # Heartbeats are deliberately outside the revisioned scene document.
    assert first.get(scene["id"])["revision"] == claimed_revision

    clock.value += 6
    assert second.list_recoverable_reconstruction_runs() == []
    assert not second.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner-b"
    )


def test_generic_document_write_cannot_bypass_active_lease(tmp_path):
    clock = MutableClock()
    store, _other, _sessions = _independent_stores(tmp_path, clock)
    scene = _scene("generic-write")
    store.put(scene)
    run_id, fingerprint = _tokens(scene)
    assert store.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner"
    )
    current = store.get(scene["id"])
    assert current is not None
    current["title"] = "must not overwrite worker"
    try:
        store.put(current)
    except SceneRevisionConflict:
        pass
    else:  # pragma: no cover - explicit assertion gives a clearer failure
        raise AssertionError("active lease accepted a generic document write")


def test_expired_lease_has_single_reclaimer_and_old_worker_is_fenced(tmp_path):
    clock = MutableClock()
    first, second, _sessions = _independent_stores(tmp_path, clock)
    scene = _scene("reclaim-race")
    first.put(scene)
    run_id, fingerprint = _tokens(scene)
    assert first.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "crashed-owner"
    )
    old_snapshot = first.get(scene["id"])

    clock.value += 11
    assert first.list_recoverable_reconstruction_runs() == [
        (scene["id"], run_id, fingerprint)
    ]
    barrier = Barrier(2)
    results: list[tuple[str, bool]] = []

    def reclaim(store: SceneStore, owner: str) -> None:
        barrier.wait(timeout=2)
        results.append(
            (
                owner,
                store.claim_reconstruction_run(
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
    assert not first.put_if_reconstruction_run(
        old_snapshot,
        run_id,
        fingerprint,
        "crashed-owner",
    )

    current = second.get(scene["id"])
    assert current is not None
    current["payload"]["tracks"] = [{"id": "winner-output"}]
    current["payload"]["videoAsset"]["reconstruction"]["status"] = "ready"
    assert second.put_if_reconstruction_run(
        current,
        run_id,
        fingerprint,
        winner,
    )
    saved = first.get(scene["id"])
    assert saved is not None
    assert saved["payload"]["tracks"] == [{"id": "winner-output"}]
    assert "lease" not in saved["payload"]["videoAsset"]["reconstruction"]
    assert not first.heartbeat_reconstruction_run(
        scene["id"], run_id, fingerprint, winner
    )


def test_failed_terminal_publish_also_clears_lease(tmp_path):
    clock = MutableClock()
    store, _other, sessions = _independent_stores(tmp_path, clock)
    scene = _scene("failed-cleanup")
    store.put(scene)
    run_id, fingerprint = _tokens(scene)
    assert store.claim_reconstruction_run(
        scene["id"], run_id, fingerprint, "owner"
    )
    current = store.get(scene["id"])
    assert current is not None
    current["payload"]["videoAsset"]["reconstruction"].update(
        {"status": "failed", "processingStatus": "failed", "error": "boom"}
    )
    assert store.put_if_reconstruction_run(
        current, run_id, fingerprint, "owner"
    )
    with sessions() as session:
        assert session.get(ReconstructionLeaseRow, scene["id"]) is None


def test_legacy_processing_without_tokens_or_lease_is_upgraded_and_reclaimed(tmp_path):
    clock = MutableClock()
    store, _other, _sessions = _independent_stores(tmp_path, clock)
    legacy = _scene("legacy", status="processing")
    reconstruction = legacy["payload"]["videoAsset"]["reconstruction"]
    reconstruction.pop("runId")
    reconstruction.pop("runRevision")
    reconstruction.pop("inputFingerprint")
    store.put(legacy)

    candidates = store.list_recoverable_reconstruction_runs()
    assert len(candidates) == 1
    scene_id, run_id, fingerprint = candidates[0]
    assert run_id.startswith("legacy-")
    assert fingerprint.startswith("sha256:")
    assert store.claim_reconstruction_run(
        scene_id, run_id, fingerprint, "legacy-recovery"
    )
    saved = store.get(scene_id)
    assert saved is not None
    saved_reconstruction = saved["payload"]["videoAsset"]["reconstruction"]
    assert saved_reconstruction["runId"] == run_id
    assert saved_reconstruction["runRevision"] == 1
    assert saved_reconstruction["inputFingerprint"] == fingerprint
    assert saved_reconstruction["lease"]["ownerId"] == "legacy-recovery"


def test_orphaned_corrupt_input_run_is_failed_instead_of_blocking_forever(tmp_path):
    clock = MutableClock()
    store, _other, _sessions = _independent_stores(tmp_path, clock)
    corrupt = _scene("corrupt", status="processing")
    corrupt["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"] = (
        "sha256:stale"
    )
    store.put(corrupt)

    assert store.fail_unrecoverable_reconstruction_runs() == 1
    saved = store.get(corrupt["id"])
    assert saved is not None
    reconstruction = saved["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["status"] == "failed"
    assert "start a fresh reconstruction" in reconstruction["error"]
    assert saved["payload"]["videoAsset"]["processingState"] == "frames-ready"


def test_delayed_duplicate_task_does_not_re_run_terminal_same_run(tmp_path, monkeypatch):
    clock = MutableClock()
    store, _other, _sessions = _independent_stores(tmp_path, clock)
    ready = _scene("terminal", status="ready")
    store.put(ready)
    run_id, fingerprint = _tokens(ready)
    calls: list[bool] = []
    monkeypatch.setattr(reconstruction_module, "scene_store", store)
    monkeypatch.setattr(
        reconstruction_module,
        "reconstruct_scene",
        lambda *_args, **_kwargs: calls.append(True),
    )

    assert not reconstruction_module.reconstruct_scene_by_id(
        ready["id"], run_id, fingerprint
    )
    assert calls == []


def test_by_id_claims_and_propagates_owner_to_terminal_publish(tmp_path, monkeypatch):
    clock = MutableClock()
    store, _other, _sessions = _independent_stores(tmp_path, clock)
    queued = _scene("by-id")
    store.put(queued)
    run_id, fingerprint = _tokens(queued)
    observed_owners: list[str] = []
    monkeypatch.setattr(reconstruction_module, "scene_store", store)

    def reconstruct(
        scene: dict,
        *,
        expected_run_id: str,
        expected_input_fingerprint: str,
        expected_lease_owner_id: str,
    ) -> dict:
        observed_owners.append(expected_lease_owner_id)
        assert scene["payload"]["videoAsset"]["reconstruction"]["lease"][
            "ownerId"
        ] == expected_lease_owner_id
        scene["payload"]["videoAsset"]["reconstruction"]["status"] = "ready"
        assert store.put_if_reconstruction_run(
            scene,
            expected_run_id,
            expected_input_fingerprint,
            expected_lease_owner_id,
        )
        return scene

    monkeypatch.setattr(reconstruction_module, "reconstruct_scene", reconstruct)
    assert reconstruction_module.reconstruct_scene_by_id(
        queued["id"], run_id, fingerprint
    )
    assert len(observed_owners) == 1
    saved = store.get(queued["id"])
    assert saved is not None
    assert saved["payload"]["videoAsset"]["reconstruction"]["status"] == "ready"
    assert "lease" not in saved["payload"]["videoAsset"]["reconstruction"]


def test_unexpected_by_id_crash_persists_failure_and_clears_lease(
    tmp_path, monkeypatch
):
    clock = MutableClock()
    store, _other, _sessions = _independent_stores(tmp_path, clock)
    queued = _scene("wrapper-crash")
    store.put(queued)
    run_id, fingerprint = _tokens(queued)
    monkeypatch.setattr(reconstruction_module, "scene_store", store)
    monkeypatch.setattr(
        reconstruction_module,
        "reconstruct_scene",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("crash")),
    )

    assert reconstruction_module.reconstruct_scene_by_id(
        queued["id"], run_id, fingerprint
    )
    saved = store.get(queued["id"])
    assert saved is not None
    reconstruction = saved["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["status"] == "failed"
    assert reconstruction["error"] == "Reconstruction worker crashed: crash"
    assert "lease" not in reconstruction


def test_recovery_monitor_scans_repeatedly_not_only_at_startup(monkeypatch):
    calls: list[int] = []
    scanned_twice = Event()

    class EmptyStore:
        def fail_unrecoverable_reconstruction_runs(self):
            return 0

        def list_recoverable_reconstruction_runs(self):
            calls.append(len(calls) + 1)
            if len(calls) >= 2:
                scanned_twice.set()
            return []

    monkeypatch.setattr(recovery_module, "scene_store", EmptyStore())
    monitor = recovery_module.ReconstructionRecoveryMonitor(poll_seconds=0.02).start()
    assert scanned_twice.wait(timeout=1)
    monitor.stop()
    assert len(calls) >= 2
    assert not monitor.is_alive()


def test_seed_is_atomic_across_independent_sqlite_engines(tmp_path):
    clock = MutableClock()
    first, second, sessions = _independent_stores(tmp_path, clock)
    barrier = Barrier(2)
    errors: list[Exception] = []

    def seed(store: SceneStore) -> None:
        try:
            barrier.wait(timeout=2)
            store.seed()
        except Exception as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    workers = [Thread(target=seed, args=(first,)), Thread(target=seed, args=(second,))]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)

    assert errors == []
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(SceneRow)) == 1
        assert session.scalar(select(SceneRow.id)) == "moment-01"
