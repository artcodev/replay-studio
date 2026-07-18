from __future__ import annotations

from copy import deepcopy
from threading import Event, Thread

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.analysis_cancellation import AnalysisCancellationService
from app.analysis_run_repository import AnalysisRunRepository
from app.database import Base
from app.project_lifecycle_contract import ProjectCreate
from app.project_resource_repository import ProjectResourceRepository
from app.project_store import ProjectStore
from app.reconstruction_run_repository import ReconstructionRunRepository
from app.scene_document import reconstruction_input_fingerprint
from app.scene_repository import SceneRepository


def _scene() -> dict:
    scene = {
        "id": "scene-race",
        "title": "Cancellation race",
        "revision": 0,
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "id": "asset-race",
                "selectedSegmentId": "segment-race",
                "sourceStart": 0.0,
                "sourceEnd": 4.0,
                "analysisFps": 10.0,
                "processingState": "reconstructing",
                "reconstruction": {
                    "status": "queued",
                    "processingStatus": "queued",
                    "runId": "run-race",
                    "runRevision": 1,
                    "model": "yolo26m.pt",
                    "progress": {"phase": "preparing", "overallPercent": 0},
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


def test_cancel_reserves_sqlite_write_before_reading_run_and_scene(tmp_path) -> None:
    """A real second connection cannot publish between the two cancel reads."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'cancel-race.sqlite3'}"
    cancel_engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    with cancel_engine.connect() as connection:
        # WAL lets the test expose a deferred-reader race: without the early
        # BEGIN IMMEDIATE, the worker can otherwise commit while cancellation
        # is paused immediately after reading AnalysisRun.
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
    Base.metadata.create_all(cancel_engine)
    worker_engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    cancel_sessions = sessionmaker(bind=cancel_engine, expire_on_commit=False)
    worker_sessions = sessionmaker(bind=worker_engine, expire_on_commit=False)
    project_store = ProjectStore(cancel_sessions)
    project_resources = ProjectResourceRepository(cancel_sessions)
    analysis_runs = AnalysisRunRepository(cancel_sessions)
    cancellation = AnalysisCancellationService(cancel_sessions)
    scene_documents = SceneRepository(worker_sessions)
    reconstruction_runs = ReconstructionRunRepository(worker_sessions)
    scene = _scene()
    fingerprint = scene["payload"]["videoAsset"]["reconstruction"][
        "inputFingerprint"
    ]
    initial = deepcopy(scene)
    initial_video = initial["payload"]["videoAsset"]
    initial_video["processingState"] = "frames-ready"
    initial_video["reconstruction"] = {
        "status": "not-started",
        "model": "yolo26m.pt",
    }
    persisted_initial = scene_documents.put(initial)
    project_store.create_project(ProjectCreate(id="project-race", title="Race"))
    project_resources.link_scene("project-race", "scene-race", role="segment")
    scene["revision"] = persisted_initial["revision"]
    reconstruction_runs.enqueue_reconstruction(
        scene,
        expected_input_fingerprint=reconstruction_input_fingerprint(initial),
    )
    assert reconstruction_runs.claim_reconstruction_run(
        "scene-race",
        "run-race",
        fingerprint,
        "worker-race",
    )
    accepted = scene_documents.get("scene-race")
    assert accepted is not None
    accepted_video = accepted["payload"]["videoAsset"]
    accepted_video["processingState"] = "tracks-ready"
    accepted_video["reconstruction"].update(
        {
            "status": "ready",
            "processingStatus": "succeeded",
            "progress": {"phase": "complete", "overallPercent": 100},
        }
    )

    analysis_read = Event()
    worker_begin = Event()
    worker_done = Event()
    read_hook_used = Event()
    worker_completed_while_cancel_was_paused: list[bool] = []
    errors: list[BaseException] = []
    cancel_results = []
    worker_results: list[bool] = []
    cancel_statements: list[str] = []

    @event.listens_for(cancel_engine, "after_cursor_execute")
    def pause_after_analysis_read(
        _connection, _cursor, statement, _parameters, _context, _many
    ) -> None:
        cancel_statements.append(statement.lower())
        if "FROM analysis_runs" not in statement or read_hook_used.is_set():
            return
        read_hook_used.set()
        analysis_read.set()
        if not worker_begin.wait(timeout=2):
            raise AssertionError("worker did not attempt its independent write")
        worker_completed_while_cancel_was_paused.append(
            worker_done.wait(timeout=0.2)
        )

    @event.listens_for(worker_engine, "before_cursor_execute")
    def observe_worker_begin(
        _connection, _cursor, statement, _parameters, _context, _many
    ) -> None:
        if statement.strip().upper().startswith("BEGIN IMMEDIATE"):
            worker_begin.set()

    def cancel() -> None:
        try:
            cancel_results.append(
                cancellation.cancel("run-race")
            )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    def publish() -> None:
        try:
            if not analysis_read.wait(timeout=2):
                raise AssertionError("cancellation never read the AnalysisRun")
            worker_results.append(
                reconstruction_runs.put_if_reconstruction_run(
                    accepted,
                    "run-race",
                    fingerprint,
                    "worker-race",
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)
        finally:
            worker_done.set()

    worker_thread = Thread(target=publish)
    cancel_thread = Thread(target=cancel)
    worker_thread.start()
    cancel_thread.start()
    cancel_thread.join(timeout=5)
    worker_thread.join(timeout=5)

    assert not cancel_thread.is_alive()
    assert not worker_thread.is_alive()
    assert errors == []
    assert worker_completed_while_cancel_was_paused == [False]
    assert [result.status for result in cancel_results] == ["cancelled"]
    assert worker_results == [False]
    persisted = scene_documents.get("scene-race")
    assert persisted is not None
    assert persisted["payload"]["videoAsset"]["reconstruction"]["status"] == "cancelled"
    assert analysis_runs.get("run-race").status == "cancelled"

    analysis_reads = [
        index
        for index, statement in enumerate(cancel_statements)
        if statement.lstrip().startswith("select") and "analysis_runs" in statement
    ]
    job_read = next(
        index
        for index, statement in enumerate(cancel_statements)
        if statement.lstrip().startswith("select") and "reconstruction_jobs" in statement
    )
    lease_read = next(
        index
        for index, statement in enumerate(cancel_statements)
        if statement.lstrip().startswith("select") and "reconstruction_leases" in statement
    )
    scene_read = next(
        index
        for index, statement in enumerate(cancel_statements)
        if statement.lstrip().startswith("select") and "scenes.payload" in statement
    )
    assert len(analysis_reads) >= 2
    # The first AnalysisRun query is an unlocked compact key lookup. Physical
    # locks then follow Job -> Lease -> Scene -> AnalysisRun on every backend.
    assert analysis_reads[0] < job_read < lease_read < scene_read < analysis_reads[-1]

    Base.metadata.drop_all(cancel_engine)
    worker_engine.dispose()
    cancel_engine.dispose()
