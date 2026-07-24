from __future__ import annotations

import asyncio
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from threading import Barrier, Event, Thread
from types import SimpleNamespace

import app.main as main_module
import app.reconstruction_recovery as recovery_module
import app.project_resource_access as resource_access
import app.scene_analysis_routes as scene_analysis_routes
from app.database import Base
from app.project_models import ProjectRow, ProjectSceneRow
from app.reconstruction_run_repository import ReconstructionRunRepository
from app.scene_document import reconstruction_input_fingerprint
from app.scene_repository import SceneRepository
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


@dataclass(frozen=True)
class Persistence:
    documents: SceneRepository
    runs: ReconstructionRunRepository
    sessions: sessionmaker


def _isolated_store(tmp_path, monkeypatch) -> Persistence:
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'recovery.sqlite3'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    return Persistence(
        documents=SceneRepository(session_local),
        runs=ReconstructionRunRepository(session_local),
        sessions=session_local,
    )


def _put_owned_queue(store: Persistence, scene: dict) -> None:
    initial = deepcopy(scene)
    video = initial["payload"]["videoAsset"]
    video["processingState"] = "frames-ready"
    video["reconstruction"] = {
        "status": "not-started",
        "model": video["reconstruction"].get("model"),
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
    store.runs.enqueue_reconstruction(
        scene,
        expected_input_fingerprint=reconstruction_input_fingerprint(initial),
    )


def test_claim_preserves_run_tokens_and_skips_processing(tmp_path, monkeypatch):
    store = _isolated_store(tmp_path, monkeypatch)
    queued = _scene("queued")
    processing = _scene("processing", status="processing")
    _put_owned_queue(store, queued)
    store.documents.put(processing)

    expected = (
        "queued",
        "run-queued",
        queued["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"],
    )
    assert store.runs.list_recoverable_reconstruction_runs(
        include_processing=False
    ) == [expected]
    assert store.runs.claim_reconstruction_run(*expected, "owner-a") is True
    assert store.runs.claim_reconstruction_run(*expected, "owner-b") is False

    saved = store.documents.get("queued")
    assert saved is not None
    reconstruction = saved["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["status"] == "processing"
    assert reconstruction["runId"] == "run-queued"
    assert reconstruction["runRevision"] == 7
    assert reconstruction["inputFingerprint"] == expected[2]
    assert store.runs.list_recoverable_reconstruction_runs(
        include_processing=False
    ) == []


def test_stale_fingerprint_is_not_recovered(tmp_path, monkeypatch):
    store = _isolated_store(tmp_path, monkeypatch)
    stale = _scene("stale")
    stale["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"] = (
        "sha256:stale"
    )
    store.documents.put(stale)
    assert store.runs.list_recoverable_reconstruction_runs() == []


def test_atomic_claim_allows_only_one_recovery_worker(tmp_path, monkeypatch):
    first_store = _isolated_store(tmp_path, monkeypatch)
    second_store = Persistence(
        documents=SceneRepository(first_store.sessions),
        runs=ReconstructionRunRepository(first_store.sessions),
        sessions=first_store.sessions,
    )
    queued = _scene("race")
    _put_owned_queue(first_store, queued)
    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]
    arguments = (
        "race",
        reconstruction["runId"],
        reconstruction["inputFingerprint"],
    )
    barrier = Barrier(2)
    results: list[bool] = []

    def claim(store: Persistence, owner: str) -> None:
        barrier.wait(timeout=2)
        results.append(store.runs.claim_reconstruction_run(*arguments, owner))

    workers = [
        Thread(target=claim, args=(store, owner))
        for store, owner in (
            (first_store, "owner-a"),
            (second_store, "owner-b"),
        )
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)

    assert sorted(results) == [False, True]


def test_fastapi_lifespan_initializes_schema_without_starting_a_runner(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(main_module, "init_database", lambda: calls.append("database"))

    async def enter_lifespan() -> None:
        async with main_module.lifespan(main_module.app):
            calls.append("serving")

    asyncio.run(enter_lifespan())

    assert calls == ["database", "serving"]
    assert not hasattr(main_module, "start_queued_reconstruction_recovery")


def test_reconstruction_endpoint_only_persists_the_queued_job(monkeypatch):
    source = _scene("api-queue", status="ready")
    queued = _scene("api-queue", status="queued")
    calls: list[tuple[str, str | None, str | None]] = []
    monkeypatch.setattr(
        resource_access,
        "project_scene_or_404",
        lambda project_id, scene_id: source,
    )

    def persist_queue(
        scene: dict,
        model: str | None,
        *,
        ball_backend: str | None,
        ball_detection_profile: str | None = None,
        jersey_ocr_profile: str | None = None,
            contact_point_profile: str | None = None,
            mode: str | None = None,
            sampling_frame_rate: float | None = None,
            direct_calibration_max_gap_seconds: float | None = None,
            match_snapshot=None,
        ):
        calls.append(
            (scene["id"], model, ball_backend, ball_detection_profile, jersey_ocr_profile)
        )
        return queued

    monkeypatch.setattr(scene_analysis_routes, "queue_reconstruction", persist_queue)
    monkeypatch.setattr(
        scene_analysis_routes.project_matches,
        "current_snapshot",
        lambda _project_id: None,
    )

    assert scene_analysis_routes.reconstruct_video_scene("project-api", "api-queue") is queued
    assert calls == [("api-queue", None, None, None, None)]
    assert not hasattr(main_module, "reconstruct_scene_by_id")


def test_dedicated_runner_terminates_cancelled_blocking_job_before_retry(
    monkeypatch,
):
    """A fenced native inference cannot continue occupying the only slot."""

    old = _scene("blocking", status="processing")
    old_reconstruction = old["payload"]["videoAsset"]["reconstruction"]
    old_run_id = old_reconstruction["runId"]
    old_fingerprint = old_reconstruction["inputFingerprint"]
    current = {"scene": old}
    events: list[str] = []
    old_started = Event()
    retry_started = Event()

    class FakeStore:
        def reconstruction_run_is_current(
            self,
            scene_id,
            run_id,
            input_fingerprint,
            *,
            statuses,
        ):
            reconstruction = current["scene"]["payload"]["videoAsset"][
                "reconstruction"
            ]
            return (
                current["scene"]["id"] == scene_id
                and reconstruction["runId"] == run_id
                and reconstruction["inputFingerprint"] == input_fingerprint
                and reconstruction["status"] in statuses
            )

        def list_recoverable_reconstruction_runs(self):
            reconstruction = current["scene"]["payload"]["videoAsset"][
                "reconstruction"
            ]
            if reconstruction["status"] != "queued":
                return []
            return [
                (
                    current["scene"]["id"],
                    reconstruction["runId"],
                    reconstruction["inputFingerprint"],
                )
            ]

    class BlockingProcess:
        next_pid = 9000

        def __init__(self, run_id):
            self.run_id = run_id
            self.pid = BlockingProcess.next_pid
            BlockingProcess.next_pid += 1
            self.returncode = None

        def poll(self):
            return self.returncode

    processes: dict[str, BlockingProcess] = {}

    def spawn(_scene_id, run_id, _input_fingerprint):
        process = BlockingProcess(run_id)
        processes[run_id] = process
        events.append(f"start:{run_id}")
        if run_id == old_run_id:
            old_started.set()
        else:
            retry_started.set()
        return process

    def terminate(process, _grace_seconds):
        events.append(f"terminate:{process.run_id}")
        process.returncode = -15

    monkeypatch.setattr(recovery_module, "reconstruction_runs", FakeStore())
    monkeypatch.setattr(recovery_module, "_spawn_reconstruction_process", spawn)
    monkeypatch.setattr(recovery_module, "_terminate_process_tree", terminate)

    # Expose the already-running old job on the first scan. It deliberately
    # never returns on its own, matching a blocking native YOLO invocation.
    old_reconstruction["status"] = "queued"
    monitor = recovery_module.DedicatedReconstructionRecoveryMonitor(
        poll_seconds=0.01,
        max_workers=1,
        termination_grace_seconds=0,
    ).start()
    assert old_started.wait(timeout=1)
    old_reconstruction["status"] = "processing"

    # Atomic cancellation and immediate retry replace the durable run token.
    # The old process is still alive, so a thread-backed max_workers=1 monitor
    # would remain stuck here forever.
    retry = _scene("blocking")
    retry_reconstruction = retry["payload"]["videoAsset"]["reconstruction"]
    retry_reconstruction["runId"] = "run-retry"
    retry_reconstruction["inputFingerprint"] = old_fingerprint
    current["scene"] = retry

    assert retry_started.wait(timeout=1)
    assert events[:3] == [
        f"start:{old_run_id}",
        f"terminate:{old_run_id}",
        "start:run-retry",
    ]
    assert processes[old_run_id].returncode == -15

    # Mark the replacement complete so monitor shutdown has nothing left to
    # terminate and the ordering assertion stays focused on cancellation.
    processes["run-retry"].returncode = 0
    monitor.stop()
    assert not monitor.is_alive()


def test_same_run_terminal_publication_is_allowed_to_finish(monkeypatch):
    """Do not kill a child between scene and AnalysisRun terminal writes."""

    scene = _scene("published", status="ready")
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    monkeypatch.setattr(
        recovery_module,
        "reconstruction_runs",
        SimpleNamespace(
            reconstruction_run_is_current=lambda *_args, **_kwargs: True
        ),
    )

    assert recovery_module.DedicatedReconstructionRecoveryMonitor._run_is_current(
        scene["id"],
        reconstruction["runId"],
        reconstruction["inputFingerprint"],
    )


def test_cancelled_same_run_is_not_considered_current(monkeypatch):
    scene = _scene("cancelled", status="cancelled")
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    monkeypatch.setattr(
        recovery_module,
        "reconstruction_runs",
        SimpleNamespace(
            reconstruction_run_is_current=lambda *_args, **_kwargs: False
        ),
    )

    assert not recovery_module.DedicatedReconstructionRecoveryMonitor._run_is_current(
        scene["id"],
        reconstruction["runId"],
        reconstruction["inputFingerprint"],
    )


def test_process_tree_termination_releases_native_blocking_process():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        recovery_module._terminate_process_tree(process, grace_seconds=0.05)
        assert process.poll() is not None
    finally:
        if process.poll() is None:  # pragma: no cover - assertion cleanup
            process.kill()
            process.wait(timeout=1)
