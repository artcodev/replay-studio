from __future__ import annotations

from copy import deepcopy

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.analysis_cancellation import AnalysisCancellationService
from app.analysis_run_repository import AnalysisRunRepository
from app.analysis_runtime import (
    AnalysisCancellationRequested,
    publish_reconstruction_progress,
    publish_reconstruction_terminal,
)
from app.database import (
    Base,
    ReconstructionJobRow,
    ReconstructionLeaseRow,
    SceneRow,
    VideoAssetRow,
)
from app.project_models import AnalysisRunRow, ProjectPersonMembershipRow
from app.project_identity_repository import ProjectIdentityRepository
from app.project_match_repository import ProjectMatchRepository
from app.project_resource_repository import (
    ProjectResourceConflict,
    ProjectResourceRepository,
)
from app.analysis_run_contract import AnalysisRunCreate
from app.project_lifecycle_contract import ProjectCreate
from app.project_segment_contract import SegmentUpsert
from app.project_store import ProjectStore
from app.reconstruction_run_repository import ReconstructionRunRepository
from app.scene_document import reconstruction_input_fingerprint
from app.scene_repository import SceneRepository


def _scene(scene_id: str = "scene-1") -> dict:
    return {
        "id": scene_id,
        "title": "1-A",
        "version": 1,
        "revision": 0,
        "duration": 8.0,
        "payload": {
            "videoAsset": {
                "id": f"asset-{scene_id}",
                "selectedSegmentId": "shot-1",
                "processingState": "frames-ready",
                "reconstruction": {"status": "not-started", "model": "yolo26m.pt"},
            },
            "teams": [{"id": "home"}, {"id": "away"}],
            "tracks": [],
            "ball": {"keyframes": []},
        },
    }


def _queue(scene: dict, run_id: str = "run-1") -> dict:
    queued = deepcopy(scene)
    video = queued["payload"]["videoAsset"]
    video["processingState"] = "reconstructing"
    video["reconstruction"].update(
        {
            "status": "queued",
            "processingStatus": "queued",
            "runId": run_id,
            "runRevision": 1,
            "progress": {
                "phase": "preparing",
                "overallPercent": 0,
                "phases": [{"dense": "scene-only"}],
            },
        }
    )
    video["reconstruction"]["inputFingerprint"] = reconstruction_input_fingerprint(
        queued
    )
    return queued


def _analysis_runs(persistence) -> AnalysisRunRepository:
    return AnalysisRunRepository(persistence[4])


def _analysis_cancellation(persistence) -> AnalysisCancellationService:
    return AnalysisCancellationService(persistence[4])


def _project_identities(persistence) -> ProjectIdentityRepository:
    return ProjectIdentityRepository(persistence[4])


def _project_matches(persistence) -> ProjectMatchRepository:
    return ProjectMatchRepository(persistence[4])


def _project_resources(persistence) -> ProjectResourceRepository:
    return ProjectResourceRepository(persistence[4])


@pytest.fixture
def persistence():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    scene_documents = SceneRepository(sessions, clock=lambda: 1_000.0)
    reconstruction_runs = ReconstructionRunRepository(
        sessions,
        clock=lambda: 1_000.0,
    )
    project_store = ProjectStore(sessions)
    project_resources = ProjectResourceRepository(sessions)
    scene = _scene()
    scene = scene_documents.put(scene)
    with sessions.begin() as session:
        session.add(
            VideoAssetRow(
                id="asset-scene-1",
                filename="source.mp4",
                original_name="source.mp4",
                content_type="video/mp4",
                status="ready",
                stage="ready",
                progress=100,
                frame_count=80,
                scene_id="scene-1",
            )
        )
    project_store.create_project(ProjectCreate(id="project-1", title="Match"))
    project_resources.link_scene("project-1", "scene-1", role="segment")
    project_resources.link_video_asset("project-1", "asset-scene-1")
    project_resources.upsert_segment(
        "project-1",
        SegmentUpsert(
            id="segment-1",
            video_asset_id="asset-scene-1",
            scene_id="scene-1",
            source_segment_id="shot-1",
            start_seconds=0,
            end_seconds=8,
        ),
    )
    queued = _queue(scene)
    reconstruction_runs.enqueue_reconstruction(
        queued,
        expected_input_fingerprint=reconstruction_input_fingerprint(scene),
    )
    yield (
        project_store,
        scene_documents,
        reconstruction_runs,
        queued,
        sessions,
        engine,
    )
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_queue_creates_scene_job_and_analysis_atomically(persistence) -> None:
    project_store, _scene_documents, _reconstruction_runs, scene, sessions, _engine = persistence
    fingerprint = scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]
    with sessions() as session:
        persisted = session.get(SceneRow, "scene-1")
        job = session.get(ReconstructionJobRow, "scene-1")
        analysis = session.get(AnalysisRunRow, "run-1")
    assert persisted is not None
    assert persisted.payload["payload"]["videoAsset"]["reconstruction"]["status"] == "queued"
    assert job is not None
    assert (job.run_id, job.input_fingerprint, job.status) == (
        "run-1",
        fingerprint,
        "queued",
    )
    assert analysis is not None
    assert analysis.project_id == "project-1"
    assert analysis.segment_id == "segment-1"
    assert analysis.status == "queued"
    assert analysis.progress == {"phase": "preparing", "overallPercent": 0}
    assert _analysis_runs(persistence).get("run-1") is not None


@pytest.mark.parametrize("write_many", [False, True])
def test_generic_scene_writes_never_rewrite_existing_job_or_telemetry(
    persistence,
    write_many: bool,
) -> None:
    _project_store, scene_documents, _reconstruction_runs, _scene, sessions, _engine = persistence
    edited = scene_documents.get("scene-1")
    assert edited is not None
    edited["title"] = "Generic editor update"
    edited["payload"]["videoAsset"]["reconstruction"].update(
        {
            "status": "failed",
            "processingStatus": "failed",
            "runId": "client-invented-run",
            "inputFingerprint": "sha256:client-invented",
        }
    )
    if write_many:
        scene_documents.put_many([edited])
    else:
        scene_documents.put(edited)

    with sessions() as session:
        job = session.get(ReconstructionJobRow, "scene-1")
        original = session.get(AnalysisRunRow, "run-1")
        invented = session.get(AnalysisRunRow, "client-invented-run")
        assert (job.run_id, job.status) == ("run-1", "queued")
        assert original.status == "queued"
        assert invented is None


def test_scene_repository_has_no_dense_project_scan_helper() -> None:
    assert not hasattr(SceneRepository, "project_scenes")


def test_superseding_queue_cancels_old_telemetry_and_creates_new_pair(persistence) -> None:
    project_store, scene_documents, reconstruction_runs, scene, sessions, engine = persistence
    fingerprint = scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]
    replacement = scene_documents.get("scene-1")
    replacement_reconstruction = replacement["payload"]["videoAsset"]["reconstruction"]
    replacement_reconstruction.update(
        {
            "status": "queued",
            "processingStatus": "queued",
            "runId": "run-2",
            "runRevision": 2,
            "inputFingerprint": fingerprint,
            "progress": {"phase": "preparing", "overallPercent": 0},
        }
    )
    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.lower())

    try:
        reconstruction_runs.enqueue_reconstruction(
            replacement,
            expected_input_fingerprint=fingerprint,
        )
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    with sessions() as session:
        job = session.get(ReconstructionJobRow, "scene-1")
        old = session.get(AnalysisRunRow, "run-1")
        new = session.get(AnalysisRunRow, "run-2")
        assert (job.run_id, job.status) == ("run-2", "queued")
        assert old.status == "cancelled"
        assert new.status == "queued"
        assert new.project_id == "project-1"
        assert new.segment_id == "segment-1"
    assert _analysis_runs(persistence).get("run-2") is not None
    assert any("project_scenes.project_id" in statement for statement in statements)
    assert any("segments.id" in statement for statement in statements)
    assert not any(" from projects" in statement for statement in statements)


def test_generic_put_does_not_create_scheduler_rows_for_unowned_scene(persistence) -> None:
    _project_store, scene_documents, _reconstruction_runs, _queued_scene, sessions, _engine = persistence
    unowned = _queue(_scene("unowned"), "run-unowned")
    scene_documents.put(unowned)
    with sessions() as session:
        assert session.get(SceneRow, "unowned") is not None
        assert session.get(ReconstructionJobRow, "unowned") is None
        assert session.get(AnalysisRunRow, "run-unowned") is None


def test_explicit_queue_requires_compact_project_ownership(persistence) -> None:
    _project_store, scene_documents, reconstruction_runs, _queued_scene, sessions, _engine = persistence
    initial = scene_documents.put(_scene("unowned-explicit"))
    queued = _queue(initial, "run-unowned-explicit")
    with pytest.raises(ProjectResourceConflict):
        reconstruction_runs.enqueue_reconstruction(
            queued,
            expected_input_fingerprint=reconstruction_input_fingerprint(initial),
        )
    with sessions() as session:
        row = session.get(SceneRow, "unowned-explicit")
        assert row.payload["payload"]["videoAsset"]["reconstruction"]["status"] == "not-started"
        assert session.get(ReconstructionJobRow, "unowned-explicit") is None
        assert session.get(AnalysisRunRow, "run-unowned-explicit") is None


def test_claim_atomically_sets_job_lease_scene_and_analysis_running(persistence) -> None:
    _project_store, _scene_documents, reconstruction_runs, scene, sessions, _engine = persistence
    fingerprint = scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]
    assert reconstruction_runs.claim_reconstruction_run(
        "scene-1", "run-1", fingerprint, "worker-1"
    )
    with sessions() as session:
        job = session.get(ReconstructionJobRow, "scene-1")
        lease = session.get(ReconstructionLeaseRow, "scene-1")
        persisted = session.get(SceneRow, "scene-1")
        analysis = session.get(AnalysisRunRow, "run-1")
    assert job.status == "processing"
    assert lease.owner_id == "worker-1"
    assert persisted.payload["payload"]["videoAsset"]["reconstruction"]["status"] == "processing"
    assert analysis.status == "running"
    assert analysis.started_at is not None


@pytest.mark.parametrize("damage", ["missing", "drift"])
def test_analysis_telemetry_damage_never_blocks_fenced_execution(
    persistence,
    damage: str,
) -> None:
    _project_store, scene_documents, reconstruction_runs, scene, sessions, _engine = persistence
    fingerprint = scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]

    def damage_run() -> None:
        with sessions.begin() as session:
            row = session.get(AnalysisRunRow, "run-1")
            if damage == "missing":
                if row is not None:
                    session.delete(row)
                return
            assert row is not None
            row.scene_id = None
            row.segment_id = None
            row.kind = "wrong-kind"
            row.status = "cancelled"
            row.source_run_id = "wrong-run"
            row.input_fingerprint = "sha256:wrong"

    damage_run()
    assert reconstruction_runs.claim_reconstruction_run(
        "scene-1",
        "run-1",
        fingerprint,
        "worker-repair",
    )
    with sessions() as session:
        repaired = session.get(AnalysisRunRow, "run-1")
        assert (
            repaired.project_id,
            repaired.scene_id,
            repaired.segment_id,
            repaired.kind,
            repaired.status,
            repaired.input_fingerprint,
        ) == (
            "project-1",
            "scene-1",
            "segment-1",
            "reconstruction",
            "running",
            fingerprint,
        )

    damage_run()
    assert reconstruction_runs.publish_reconstruction_progress(
        "scene-1",
        "run-1",
        fingerprint,
        "worker-repair",
        {"phase": "tracking", "overallPercent": 45},
    ) == "published"
    with sessions() as session:
        repaired = session.get(AnalysisRunRow, "run-1")
        assert repaired.status == "running"
        assert repaired.progress == {"phase": "tracking", "overallPercent": 45}

    result = scene_documents.get("scene-1")
    assert result is not None
    result["payload"]["videoAsset"]["reconstruction"].update(
        {
            "status": "ready",
            "processingStatus": "succeeded",
            "progress": {"phase": "complete", "overallPercent": 100},
        }
    )
    damage_run()
    assert reconstruction_runs.put_if_reconstruction_run(
        result,
        "run-1",
        fingerprint,
        "worker-repair",
    )
    with sessions() as session:
        repaired = session.get(AnalysisRunRow, "run-1")
        assert repaired.status == "succeeded"
        assert repaired.scene_id == "scene-1"
        assert repaired.input_fingerprint == fingerprint


def test_invalid_claim_terminalizes_compact_telemetry_once(persistence) -> None:
    _project_store, _scene_documents, reconstruction_runs, scene, sessions, _engine = persistence
    fingerprint = scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]
    with sessions.begin() as session:
        row = session.get(SceneRow, "scene-1")
        broken = deepcopy(row.payload)
        broken["payload"]["videoAsset"]["reconstruction"]["runId"] = "other-run"
        row.payload = broken
    assert not reconstruction_runs.claim_reconstruction_run(
        "scene-1", "run-1", fingerprint, "worker-invalid"
    )
    with sessions() as session:
        assert session.get(ReconstructionJobRow, "scene-1").status == "invalid"
        analysis = session.get(AnalysisRunRow, "run-1")
        assert analysis.status == "failed"
        assert "does not match" in analysis.error
        assert analysis.completed_at is not None
        assert session.get(ReconstructionLeaseRow, "scene-1") is None
        preserved = session.get(SceneRow, "scene-1").payload
        preserved_reconstruction = preserved["payload"]["videoAsset"][
            "reconstruction"
        ]
        assert preserved_reconstruction["runId"] == "other-run"
        assert preserved_reconstruction["status"] == "queued"


def test_fenced_progress_is_compact_and_never_selects_scene_payload(persistence) -> None:
    project_store, _scene_documents, reconstruction_runs, scene, sessions, engine = persistence
    fingerprint = scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]
    assert reconstruction_runs.claim_reconstruction_run(
        "scene-1", "run-1", fingerprint, "worker-progress"
    )
    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.lower())

    try:
        assert publish_reconstruction_progress(
            scene,
            {
                "phase": "tracking",
                "label": "Tracking players",
                "overallPercent": 42,
                "completed": 21,
                "total": 50,
                "phases": [{"large": "scene-only"}],
            },
            expected_run_id="run-1",
            expected_input_fingerprint=fingerprint,
            expected_lease_owner_id="worker-progress",
            run_repository=reconstruction_runs,
        )
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    run = _analysis_runs(persistence).get("run-1")
    assert run.status == "running"
    assert run.progress["overallPercent"] == 42
    assert "phases" not in run.progress
    assert statements
    assert not any(" from scenes" in statement for statement in statements)
    assert not any("scenes.payload" in statement for statement in statements)


def test_terminal_publish_atomically_closes_scene_job_lease_and_analysis(persistence) -> None:
    project_store, scene_documents, reconstruction_runs, scene, sessions, _engine = persistence
    fingerprint = scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]
    assert reconstruction_runs.claim_reconstruction_run(
        "scene-1", "run-1", fingerprint, "worker-final"
    )
    result = scene_documents.get("scene-1")
    result["payload"]["tracks"] = [{"id": "accepted-track"}]
    result["payload"]["videoAsset"]["processingState"] = "tracks-ready"
    result["payload"]["videoAsset"]["reconstruction"].update(
        {
            "status": "ready",
            "processingStatus": "succeeded",
            "progress": {"phase": "complete", "overallPercent": 100},
            "error": None,
        }
    )
    assert reconstruction_runs.put_if_reconstruction_run(
        result, "run-1", fingerprint, "worker-final"
    )
    with sessions() as session:
        assert session.get(SceneRow, "scene-1").payload["payload"]["tracks"] == [
            {"id": "accepted-track"}
        ]
        assert session.get(ReconstructionJobRow, "scene-1").status == "ready"
        assert session.get(ReconstructionLeaseRow, "scene-1") is None
        analysis = session.get(AnalysisRunRow, "run-1")
        assert analysis.status == "succeeded"
        assert analysis.progress["overallPercent"] == 100
        assert analysis.completed_at is not None
    assert _analysis_runs(persistence).get("run-1").status == "succeeded"


def test_cancel_atomically_fences_worker_and_progress_reports_cancellation(persistence) -> None:
    project_store, _scene_documents, reconstruction_runs, scene, sessions, _engine = persistence
    fingerprint = scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]
    assert reconstruction_runs.claim_reconstruction_run(
        "scene-1", "run-1", fingerprint, "worker-cancel"
    )
    assert _analysis_cancellation(persistence).cancel("run-1").status == "cancelled"
    with sessions() as session:
        assert session.get(ReconstructionJobRow, "scene-1").status == "cancelled"
        assert session.get(ReconstructionLeaseRow, "scene-1") is None
        assert session.get(AnalysisRunRow, "run-1").status == "cancelled"
        reconstruction = session.get(SceneRow, "scene-1").payload["payload"]["videoAsset"]["reconstruction"]
        assert reconstruction["status"] == "cancelled"
    with pytest.raises(AnalysisCancellationRequested):
        publish_reconstruction_progress(
            scene,
            {"phase": "tracking", "overallPercent": 43},
            expected_run_id="run-1",
            expected_input_fingerprint=fingerprint,
            expected_lease_owner_id="worker-cancel",
            run_repository=reconstruction_runs,
        )


def test_orphan_analysis_cancel_never_reads_dense_scene(persistence) -> None:
    project_store, _scene_documents, _reconstruction_runs, _scene, sessions, engine = persistence
    _analysis_runs(persistence).create(
        "project-1",
        AnalysisRunCreate(
            id="orphan-run",
            scene_id="scene-1",
            kind="reconstruction",
            status="queued",
            source_run_id="orphan-run",
            input_fingerprint="sha256:orphan",
            progress={"phase": "preparing", "overallPercent": 0},
        ),
    )
    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.lower())

    try:
        cancelled = _analysis_cancellation(persistence).cancel("orphan-run")
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert cancelled.status == "cancelled"
    assert not any("scenes.payload" in statement for statement in statements)
    with sessions() as session:
        assert session.get(ReconstructionJobRow, "scene-1").run_id == "run-1"
        assert session.get(ReconstructionJobRow, "scene-1").status == "queued"


def test_terminal_hook_only_adds_identity_sync_diagnostics(persistence) -> None:
    project_store, scene_documents, reconstruction_runs, scene, _sessions, _engine = persistence
    fingerprint = scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]
    assert reconstruction_runs.claim_reconstruction_run(
        "scene-1", "run-1", fingerprint, "worker-identities"
    )
    result = scene_documents.get("scene-1")
    result["payload"]["canonicalPeople"] = [
        {
            "canonicalPersonId": "canonical-home-8",
            "displayName": "Home Eight",
            "teamId": "home",
            "role": "player",
            "identityStatus": "provisional",
            "observations": [{"frameIndex": 1}],
        }
    ]
    result["payload"]["videoAsset"]["reconstruction"].update(
        {
            "status": "ready",
            "processingStatus": "succeeded",
            "progress": {"phase": "complete", "overallPercent": 100},
        }
    )
    assert reconstruction_runs.put_if_reconstruction_run(
        result, "run-1", fingerprint, "worker-identities"
    )
    runs = _analysis_runs(persistence)
    before = runs.get("run-1")
    assert before.status == "succeeded"
    assert publish_reconstruction_terminal(
        result,
        "ready",
        projects=project_store,
        resources=_project_resources(persistence),
        runs=runs,
        matches=_project_matches(persistence),
        identities=_project_identities(persistence),
    )
    after = runs.get("run-1")
    assert after.status == "succeeded"
    assert after.diagnostics["identitySync"]["status"] == "succeeded"


def test_startup_sweep_repairs_a_crashed_identity_sync_epilogue(
    persistence, monkeypatch
) -> None:
    from app.analysis_runtime import recover_missed_identity_sync

    project_store, scene_documents, reconstruction_runs, scene, _sessions, _engine = (
        persistence
    )
    fingerprint = scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]
    assert reconstruction_runs.claim_reconstruction_run(
        "scene-1", "run-1", fingerprint, "worker-crash"
    )
    result = scene_documents.get("scene-1")
    assert result is not None
    result["payload"]["canonicalPeople"] = [
        {
            "canonicalPersonId": "canonical-home-9",
            "displayName": "Home Nine",
            "teamId": "home",
            "role": "player",
            "identityStatus": "provisional",
            "observations": [{"frameIndex": 1}],
        }
    ]
    result["payload"]["videoAsset"]["reconstruction"].update(
        {
            "status": "ready",
            "processingStatus": "succeeded",
            "progress": {"phase": "complete", "overallPercent": 100},
        }
    )
    assert reconstruction_runs.put_if_reconstruction_run(
        result, "run-1", fingerprint, "worker-crash"
    )
    # The worker died here: the fenced terminal commit exists, but the
    # identity-sync epilogue never ran and telemetry has no identitySync.
    runs = _analysis_runs(persistence)
    assert "identitySync" not in dict(runs.get("run-1").diagnostics or {})

    def sweep() -> int:
        return recover_missed_identity_sync(
            scenes=scene_documents,
            runs=runs,
            projects=project_store,
            resources=_project_resources(persistence),
            matches=_project_matches(persistence),
            identities=_project_identities(persistence),
        )

    repaired = sweep()

    assert repaired == 1
    after = runs.get("run-1")
    assert after.diagnostics["identitySync"]["status"] == "succeeded"
    # A second sweep finds nothing left to repair.
    assert sweep() == 0


def test_terminal_identity_sync_uses_scene_owner_when_telemetry_is_missing(
    persistence,
) -> None:
    project_store, scene_documents, reconstruction_runs, scene, sessions, _engine = persistence
    fingerprint = scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]
    assert reconstruction_runs.claim_reconstruction_run(
        "scene-1", "run-1", fingerprint, "worker-no-telemetry"
    )
    result = scene_documents.get("scene-1")
    assert result is not None
    result["payload"]["canonicalPeople"] = [
        {
            "canonicalPersonId": "canonical-home-9",
            "displayName": "Home Nine",
            "teamId": "home",
            "role": "player",
            "identityStatus": "provisional",
            "observations": [{"frameIndex": 1}],
        }
    ]
    result["payload"]["videoAsset"]["reconstruction"].update(
        {
            "status": "ready",
            "processingStatus": "succeeded",
            "progress": {"phase": "complete", "overallPercent": 100},
        }
    )
    assert reconstruction_runs.put_if_reconstruction_run(
        result,
        "run-1",
        fingerprint,
        "worker-no-telemetry",
    )
    with sessions.begin() as session:
        run = session.get(AnalysisRunRow, "run-1")
        assert run is not None
        session.delete(run)

    assert publish_reconstruction_terminal(
        result,
        "ready",
        projects=project_store,
        resources=_project_resources(persistence),
        runs=_analysis_runs(persistence),
        matches=_project_matches(persistence),
        identities=_project_identities(persistence),
    )
    with sessions() as session:
        assert session.get(AnalysisRunRow, "run-1") is None
        memberships = session.scalars(
            select(ProjectPersonMembershipRow).where(
                ProjectPersonMembershipRow.project_id == "project-1",
                ProjectPersonMembershipRow.scene_id == "scene-1",
            )
        ).all()
        assert len(memberships) == 1
        assert memberships[0].scene_person_id == "canonical-home-9"
