from __future__ import annotations

import asyncio
from threading import Barrier, Thread

import app.main as main_module
import app.reconstruction_recovery as recovery_module
import app.store as store_module
from app.database import Base
from app.store import SceneStore, reconstruction_input_fingerprint
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _scene(scene_id: str, status: str = "queued") -> dict:
    scene = {
        "id": scene_id,
        "title": f"Recovery {scene_id}",
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
                    "runRevision": 7,
                    "frameAnnotations": [],
                },
            },
            "teams": [],
            "tracks": [],
            "ball": {"keyframes": []},
        },
    }
    scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"] = (
        reconstruction_input_fingerprint(scene)
    )
    return scene


def _isolated_store(tmp_path, monkeypatch) -> SceneStore:
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'recovery.sqlite3'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(store_module, "SessionLocal", session_local)
    return SceneStore()


def test_claim_preserves_run_tokens_and_skips_processing(tmp_path, monkeypatch):
    store = _isolated_store(tmp_path, monkeypatch)
    queued = _scene("queued")
    processing = _scene("processing", status="processing")
    store.put(queued)
    store.put(processing)

    expected = (
        "queued",
        "run-queued",
        queued["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"],
    )
    assert store.list_queued_reconstruction_runs() == [expected]
    assert store.claim_queued_reconstruction_run(*expected) is True
    assert store.claim_queued_reconstruction_run(*expected) is False

    saved = store.get("queued")
    assert saved is not None
    reconstruction = saved["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["status"] == "processing"
    assert reconstruction["runId"] == "run-queued"
    assert reconstruction["runRevision"] == 7
    assert reconstruction["inputFingerprint"] == expected[2]
    assert store.list_queued_reconstruction_runs() == []


def test_recovery_executes_queued_and_missing_lease_processing_runs(tmp_path, monkeypatch):
    store = _isolated_store(tmp_path, monkeypatch)
    queued = _scene("queued-job")
    processing = _scene("active-job", status="processing")
    store.put(queued)
    store.put(processing)
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(recovery_module, "scene_store", store)
    def reconstruct(scene_id: str, run_id: str, fingerprint: str) -> bool:
        calls.append((scene_id, run_id, fingerprint))
        return True

    monkeypatch.setattr(recovery_module, "reconstruct_scene_by_id", reconstruct)

    assert recovery_module.recover_queued_reconstruction_jobs() == 2

    queued_reconstruction = queued["payload"]["videoAsset"]["reconstruction"]
    assert calls == [
        (
            "active-job",
            "run-active-job",
            processing["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"],
        ),
        (
            "queued-job",
            queued_reconstruction["runId"],
            queued_reconstruction["inputFingerprint"],
        )
    ]
    active = store.get("active-job")
    assert active is not None
    assert active["payload"]["videoAsset"]["reconstruction"]["status"] == "processing"


def test_stale_fingerprint_is_not_recovered(tmp_path, monkeypatch):
    store = _isolated_store(tmp_path, monkeypatch)
    stale = _scene("stale")
    stale["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"] = (
        "sha256:stale"
    )
    store.put(stale)

    assert store.list_queued_reconstruction_runs() == []
    assert store.claim_queued_reconstruction_run(
        "stale",
        "run-stale",
        "sha256:stale",
    ) is False
    saved = store.get("stale")
    assert saved is not None
    assert saved["payload"]["videoAsset"]["reconstruction"]["status"] == "queued"


def test_unusual_recovery_wrapper_crash_does_not_block_later_jobs(
    tmp_path,
    monkeypatch,
):
    store = _isolated_store(tmp_path, monkeypatch)
    first = _scene("first")
    second = _scene("second")
    store.put(first)
    store.put(second)
    calls: list[str] = []
    monkeypatch.setattr(recovery_module, "scene_store", store)

    def reconstruct(scene_id: str, _run_id: str, _fingerprint: str) -> bool:
        calls.append(scene_id)
        if scene_id == "first":
            raise RuntimeError("worker thread crashed")
        return True

    monkeypatch.setattr(recovery_module, "reconstruct_scene_by_id", reconstruct)

    assert recovery_module.recover_queued_reconstruction_jobs() == 1
    assert calls == ["first", "second"]
    failed = store.get("first")
    assert failed is not None
    failed_reconstruction = failed["payload"]["videoAsset"]["reconstruction"]
    # The fake crashed before the atomic claim. The durable queued job remains
    # eligible for the next monitor pass instead of being lost.
    assert failed_reconstruction["status"] == "queued"
    assert failed_reconstruction["runId"] == "run-first"
    assert failed_reconstruction["runRevision"] == 7


def test_atomic_claim_allows_only_one_recovery_worker(tmp_path, monkeypatch):
    first_store = _isolated_store(tmp_path, monkeypatch)
    second_store = SceneStore()
    queued = _scene("race")
    first_store.put(queued)
    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]
    arguments = (
        "race",
        reconstruction["runId"],
        reconstruction["inputFingerprint"],
    )
    barrier = Barrier(2)
    results: list[bool] = []

    def claim(store: SceneStore) -> None:
        barrier.wait(timeout=2)
        results.append(store.claim_queued_reconstruction_run(*arguments))

    workers = [Thread(target=claim, args=(store,)) for store in (first_store, second_store)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)

    assert sorted(results) == [False, True]


def test_fastapi_lifespan_starts_reconstruction_recovery(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(main_module, "init_database", lambda: calls.append("database"))
    monkeypatch.setattr(main_module.scene_store, "seed", lambda: calls.append("seed"))
    monkeypatch.setattr(
        main_module,
        "start_queued_reconstruction_recovery",
        lambda: calls.append("recovery"),
    )

    async def enter_lifespan() -> None:
        async with main_module.lifespan(main_module.app):
            calls.append("serving")

    asyncio.run(enter_lifespan())

    assert calls == ["database", "seed", "recovery", "serving"]
