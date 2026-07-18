from __future__ import annotations

from datetime import UTC, datetime
from threading import Barrier, Thread

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.database import Base, PipelineJobRow, VideoAssetRow
from app.model_comparison_pipeline_service import ModelComparisonPipelineService
from app.multi_pass_pipeline_service import MultiPassPipelineService
from app.pipeline_domain import PipelineJob, PipelineJobConflict
from app.pipeline_store import PipelineStore
from app.pipeline_terminal_service import PipelineTerminalService
from app.project_models import (
    AnalysisRunRow,
    ProjectRow,
    ProjectSceneRow,
    ProjectVideoAssetRow,
    SegmentRow,
)
from app.sample import make_video_scene
from app.scene_repository import SceneRepository
from app.video_pipeline import VideoPipelineService


class MutableClock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _stores(tmp_path, clock: MutableClock):
    url = f"sqlite+pysqlite:///{tmp_path / 'pipeline.sqlite3'}"
    first_engine = create_engine(
        url, connect_args={"check_same_thread": False, "timeout": 5}
    )
    second_engine = create_engine(
        url, connect_args={"check_same_thread": False, "timeout": 5}
    )
    Base.metadata.create_all(first_engine)
    first_sessions = sessionmaker(bind=first_engine, expire_on_commit=False)
    second_sessions = sessionmaker(bind=second_engine, expire_on_commit=False)
    with first_sessions.begin() as session:
        session.add(ProjectRow(id="project-1", title="Pipeline", status="active"))
    return (
        PipelineStore(first_sessions, clock=clock),
        PipelineStore(second_sessions, clock=clock),
        first_engine,
    )


def _terminals(store: PipelineStore) -> PipelineTerminalService:
    return PipelineTerminalService(store._session_factory, clock=store._clock)


def test_pipeline_store_has_no_resource_publication_authority() -> None:
    moved_methods = {
        "cancel",
        "complete",
        "enqueue_with_analysis",
        "enqueue_model_comparison",
        "enqueue_multi_pass_scene",
        "enqueue_video_upload",
        "fail",
        "publish_model_comparison_result",
        "publish_scene_terminal",
        "publish_video_result",
        "reconstruction_statuses",
        "update_video_progress",
    }
    assert moved_methods.isdisjoint(vars(PipelineStore))


def _seed_job(
    store: PipelineStore,
    *,
    job_id: str,
    project_id: str,
    kind: str,
    subject_id: str,
    state: dict | None = None,
    scene_id: str | None = None,
) -> None:
    """Create scheduler state directly for repository-only lease tests."""

    now = float(store._clock())
    with store._session_factory.begin() as session:
        session.add_all(
            [
                PipelineJobRow(
                    id=job_id,
                    project_id=project_id,
                    kind=kind,
                    subject_id=subject_id,
                    status="queued",
                    state=dict(state or {}),
                    parameters={},
                    available_at=now,
                    attempts=0,
                    requested_at=now,
                    updated_at=now,
                ),
                AnalysisRunRow(
                    id=job_id,
                    project_id=project_id,
                    scene_id=scene_id,
                    kind=kind,
                    status="queued",
                    progress={},
                    diagnostics={},
                    requested_at=datetime.fromtimestamp(now, UTC),
                ),
            ]
        )


def test_persisted_job_survives_api_store_restart_and_has_one_claim(tmp_path):
    clock = MutableClock()
    first, restarted, _engine = _stores(tmp_path, clock)
    _seed_job(
        first,
        job_id="video-run-1",
        project_id="project-1",
        kind="video-processing",
        subject_id="asset-1",
        state={"phase": "ingest"},
    )

    # A fresh store represents a new API/runner process. No in-memory task or
    # AnalysisRun scan is required to recover the durable job.
    assert restarted.list_recoverable() == ["video-run-1"]
    barrier = Barrier(2)
    claims = []

    def claim(store: PipelineStore, owner: str) -> None:
        barrier.wait(timeout=2)
        claims.append(store.claim("video-run-1", owner, ttl_seconds=10))

    workers = [
        Thread(target=claim, args=(first, "runner-a")),
        Thread(target=claim, args=(restarted, "runner-b")),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert winners[0].lease_token


def test_idle_discovery_and_liveness_never_select_scene_payload(tmp_path):
    clock = MutableClock()
    store, _other, engine = _stores(tmp_path, clock)
    _seed_job(
        store,
        job_id="multi-run-1",
        project_id="project-1",
        kind="multi-pass",
        subject_id="scene-composite",
        state={"phase": "prepare"},
    )
    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(" ".join(statement.lower().split()))

    assert store.list_recoverable() == ["multi-run-1"]
    claim = store.claim("multi-run-1", "runner", ttl_seconds=10)
    assert claim is not None and claim.lease_token is not None
    assert store.is_claim_current("multi-run-1", claim.lease_token)
    assert store.heartbeat(
        "multi-run-1", claim.lease_token, ttl_seconds=10
    )

    assert statements
    assert all(" from scenes" not in statement for statement in statements)
    assert all("scenes.payload" not in statement for statement in statements)


def test_cancel_and_expired_reclaim_fence_stale_pipeline_worker(tmp_path):
    clock = MutableClock()
    first, second, _engine = _stores(tmp_path, clock)
    _seed_job(
        first,
        job_id="video-run-fence",
        project_id="project-1",
        kind="video-processing",
        subject_id="asset-fence",
    )
    original = first.claim("video-run-fence", "runner-a", ttl_seconds=10)
    assert original is not None and original.lease_token

    assert _terminals(second).cancel("video-run-fence").status == "cancelled"
    assert not first.heartbeat(
        "video-run-fence", original.lease_token, ttl_seconds=10
    )
    assert not _terminals(first).fail(
        "video-run-fence", original.lease_token, "stale"
    )
    assert second.list_recoverable() == []

    _seed_job(
        first,
        job_id="multi-run-reclaim",
        project_id="project-1",
        kind="multi-pass",
        subject_id="scene-reclaim",
    )
    stale = first.claim("multi-run-reclaim", "runner-a", ttl_seconds=10)
    assert stale is not None and stale.lease_token
    clock.value += 11
    recovered = second.claim("multi-run-reclaim", "runner-b", ttl_seconds=10)
    assert recovered is not None and recovered.lease_token != stale.lease_token
    assert not _terminals(first).fail(
        "multi-run-reclaim", stale.lease_token, "stale"
    )
    assert _terminals(second).fail(
        "multi-run-reclaim", recovered.lease_token, "synthetic"
    )


def test_video_upload_asset_owner_job_and_telemetry_commit_as_one_unit(tmp_path):
    clock = MutableClock()
    store, _other, engine = _stores(tmp_path, clock)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    video = VideoPipelineService(store._session_factory, clock=clock)

    created = video.enqueue_upload(
        job_id="video-upload-run",
        project_id="project-1",
        asset_id="asset-upload",
        filename="source.mp4",
        original_name="highlight.mp4",
        content_type="video/mp4",
        title="Highlight",
    )
    assert created.status == "queued"
    with sessions() as session:
        asset = session.get(VideoAssetRow, "asset-upload")
        owner = session.get(ProjectVideoAssetRow, ("project-1", "asset-upload"))
        job = session.get(PipelineJobRow, "video-upload-run")
        run = session.get(AnalysisRunRow, "video-upload-run")
        assert asset is not None and owner is not None and job is not None and run is not None
        assert asset.status == job.status == run.status == "queued"
        assert job.parameters == {"title": "Highlight"}

    with pytest.raises(PipelineJobConflict, match="was not found"):
        video.enqueue_upload(
            job_id="orphan-upload-run",
            project_id="missing-project",
            asset_id="asset-orphan",
            filename="source.mp4",
            original_name="orphan.mp4",
            content_type="video/mp4",
        )
    with sessions() as session:
        assert session.get(VideoAssetRow, "asset-orphan") is None
        assert session.get(PipelineJobRow, "orphan-upload-run") is None
        assert session.get(AnalysisRunRow, "orphan-upload-run") is None


def _queued_multi_pass_scene(scene_id: str, run_id: str) -> dict:
    return make_video_scene(
        scene_id=scene_id,
        title="Two angles",
        duration=8.0,
        video_asset={
            "id": "asset-source",
            "filename": "source.mp4",
            "generationKey": "generation-source",
            "multiPass": {
                "status": "queued",
                "sourcePasses": [
                    {"segmentId": "a", "sceneId": "source-a"},
                    {"segmentId": "b", "sceneId": "source-b"},
                ],
            },
            "reconstruction": {
                "status": "queued",
                "processingStatus": "queued",
                "runId": run_id,
                "model": "model.pt",
                "progress": {"phase": "angle-1", "overallPercent": 0},
            },
        },
    )


def test_multi_pass_scene_link_job_and_telemetry_commit_as_one_unit(tmp_path):
    clock = MutableClock()
    store, _other, engine = _stores(tmp_path, clock)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    multi_pass_pipeline = MultiPassPipelineService(
        store._session_factory,
        clock=clock,
    )
    scenes = SceneRepository(sessions)
    for source_id in ("source-a", "source-b"):
        scenes.put(
            make_video_scene(
                scene_id=source_id,
                title=source_id,
                duration=8.0,
                video_asset={"id": "asset-source", "filename": "source.mp4"},
            )
        )
    with sessions.begin() as session:
        for source_id in ("source-a", "source-b"):
            session.add(
                ProjectSceneRow(
                    project_id="project-1",
                    scene_id=source_id,
                    role="segment",
                )
            )

    prepared = _queued_multi_pass_scene("multi-atomic", "multi-atomic-run")
    published = multi_pass_pipeline.enqueue(
        project_id="project-1",
        scene=prepared,
        source_scene_ids=["source-a", "source-b"],
    )
    assert published["revision"] == 1
    with sessions() as session:
        assert session.get(ProjectSceneRow, ("project-1", "multi-atomic")) is not None
        assert session.get(PipelineJobRow, "multi-atomic-run").status == "queued"
        assert session.get(AnalysisRunRow, "multi-atomic-run").status == "queued"
        assert session.get(AnalysisRunRow, "multi-atomic-run").scene_id == "multi-atomic"

    rejected = _queued_multi_pass_scene("multi-orphan", "multi-orphan-run")
    with pytest.raises(PipelineJobConflict, match="missing or owned"):
        multi_pass_pipeline.enqueue(
            project_id="project-1",
            scene=rejected,
            source_scene_ids=["source-a", "missing-source"],
        )
    with sessions() as session:
        assert session.get(ProjectSceneRow, ("project-1", "multi-orphan")) is None
        assert session.get(PipelineJobRow, "multi-orphan-run") is None
        assert session.get(AnalysisRunRow, "multi-orphan-run") is None


def _multi_scene(scene_id: str) -> dict:
    return {
        "id": scene_id,
        "title": "Composite",
        "duration": 5.0,
        "revision": 1,
        "payload": {
            "videoAsset": {
                "multiPass": {"status": "queued", "sourcePasses": []},
                "reconstruction": {
                    "status": "queued",
                    "runId": f"run-{scene_id}",
                    "progress": {"phase": "queued", "overallPercent": 0},
                },
            }
        },
    }


def test_multi_pass_terminal_scene_and_job_obey_cancel_order(tmp_path):
    clock = MutableClock()
    store, _other, engine = _stores(tmp_path, clock)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    scenes = SceneRepository(sessions)
    multi_pass_pipeline = MultiPassPipelineService(
        store._session_factory,
        clock=clock,
    )

    cancel_first = scenes.put(_multi_scene("cancel-first"))
    _seed_job(
        store,
        job_id="run-cancel-first",
        project_id="project-1",
        kind="multi-pass",
        subject_id="cancel-first",
        scene_id="cancel-first",
    )
    claim = store.claim("run-cancel-first", "runner", ttl_seconds=10)
    assert claim is not None and claim.lease_token
    terminal = dict(cancel_first)
    terminal["payload"] = {
        **cancel_first["payload"],
        "videoAsset": {
            **cancel_first["payload"]["videoAsset"],
            "reconstruction": {
                **cancel_first["payload"]["videoAsset"]["reconstruction"],
                "status": "ready",
            },
        },
    }
    assert _terminals(store).cancel("run-cancel-first").status == "cancelled"
    assert not multi_pass_pipeline.publish(
        "run-cancel-first",
        claim.lease_token,
        scene=terminal,
        status="succeeded",
    )
    cancelled_scene = scenes.get("cancel-first")
    assert cancelled_scene["payload"]["videoAsset"]["reconstruction"][
        "status"
    ] == "cancelled"
    assert cancelled_scene["payload"]["videoAsset"]["multiPass"][
        "status"
    ] == "cancelled"
    with sessions() as session:
        assert session.get(AnalysisRunRow, "run-cancel-first").status == "cancelled"

    final_first = scenes.put(_multi_scene("final-first"))
    _seed_job(
        store,
        job_id="run-final-first",
        project_id="project-1",
        kind="multi-pass",
        subject_id="final-first",
        scene_id="final-first",
    )
    claim = store.claim("run-final-first", "runner", ttl_seconds=10)
    assert claim is not None and claim.lease_token
    final_first["payload"]["videoAsset"]["reconstruction"]["status"] = "ready"
    assert multi_pass_pipeline.publish(
        "run-final-first",
        claim.lease_token,
        scene=final_first,
        status="succeeded",
    )
    assert _terminals(store).cancel("run-final-first").status == "succeeded"
    assert scenes.get("final-first")["payload"]["videoAsset"]["reconstruction"][
        "status"
    ] == "ready"
    with sessions() as session:
        assert session.get(AnalysisRunRow, "run-final-first").status == "succeeded"


def _prepared_video_graph(asset_id: str, generation_key: str):
    root = make_video_scene(
        scene_id=f"root-{asset_id}",
        title="Processed video",
        duration=8.0,
        video_asset={
            "id": asset_id,
            "filename": "highlight.mp4",
            "generationKey": generation_key,
            "analysisFps": 10.0,
            "frameCount": 80,
            "segments": [],
        },
    )
    segment = {
        "id": "shot-01",
        "label": "Shot 01",
        "start": 0.0,
        "end": 8.0,
        "duration": 8.0,
        "sceneId": f"child-{asset_id}",
    }
    child = make_video_scene(
        scene_id=segment["sceneId"],
        title="Processed video · Shot 01",
        duration=8.0,
        video_asset={
            **root["payload"]["videoAsset"],
            "parentSceneId": root["id"],
            "selectedSegmentId": segment["id"],
            "segments": [],
        },
    )
    root["payload"]["videoAsset"]["segments"] = [segment]
    return root, child, segment


def test_video_generation_pointer_graph_and_terminal_state_publish_together(tmp_path):
    clock = MutableClock()
    store, _other, engine = _stores(tmp_path, clock)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    video = VideoPipelineService(store._session_factory, clock=clock)
    with sessions.begin() as session:
        for asset_id in ("asset-cancel", "asset-stale", "asset-final"):
            session.add(
                VideoAssetRow(
                    id=asset_id,
                    filename="source.mp4",
                    original_name="highlight.mp4",
                    content_type="video/mp4",
                    status="queued",
                    stage="Queued",
                    progress=2,
                    frame_count=0,
                )
            )
            session.add(
                ProjectVideoAssetRow(
                    project_id="project-1",
                    video_asset_id=asset_id,
                    role="source",
                )
            )

    _seed_job(
        store,
        job_id="video-cancel",
        project_id="project-1",
        kind="video-processing",
        subject_id="asset-cancel",
    )
    claim = store.claim("video-cancel", "runner", ttl_seconds=10)
    assert claim is not None and claim.lease_token
    root, child, segment = _prepared_video_graph("asset-cancel", "generation-a")
    assert _terminals(store).cancel("video-cancel").status == "cancelled"
    assert not video.update_progress(
        "video-cancel",
        claim.lease_token,
        {"stage": "Stale worker write", "progress": 77},
    )
    assert not video.publish_result(
        "video-cancel",
        claim.lease_token,
        root_scene=root,
        child_scenes=[child],
        segments=[segment],
        frame_count=80,
        generation_key="generation-a",
        stage="Ready",
    )
    with sessions() as session:
        asset = session.get(VideoAssetRow, "asset-cancel")
        assert asset is not None and asset.generation_key is None
        assert (asset.status, asset.stage, asset.progress) == (
            "cancelled",
            "Processing cancelled",
            100,
        )
        assert session.get(AnalysisRunRow, "video-cancel").status == "cancelled"
        assert session.get(ProjectSceneRow, ("project-1", root["id"])) is None

    _seed_job(
        store,
        job_id="video-stale",
        project_id="project-1",
        kind="video-processing",
        subject_id="asset-stale",
    )
    stale = store.claim("video-stale", "runner-a", ttl_seconds=10)
    assert stale is not None and stale.lease_token
    clock.value += 11
    replacement = store.claim("video-stale", "runner-b", ttl_seconds=10)
    assert replacement is not None and replacement.lease_token != stale.lease_token
    root, child, segment = _prepared_video_graph("asset-stale", "generation-stale")
    assert not video.update_progress(
        "video-stale",
        stale.lease_token,
        {"stage": "Stale worker write", "progress": 77},
    )
    assert not video.publish_result(
        "video-stale",
        stale.lease_token,
        root_scene=root,
        child_scenes=[child],
        segments=[segment],
        frame_count=80,
        generation_key="generation-stale",
        stage="Ready",
    )
    with sessions() as session:
        asset = session.get(VideoAssetRow, "asset-stale")
        assert asset is not None
        assert (asset.status, asset.stage, asset.progress) == (
            "queued",
            "Queued",
            2,
        )
        assert asset.generation_key is None and asset.scene_id is None
        assert session.get(ProjectSceneRow, ("project-1", root["id"])) is None
    assert _terminals(store).cancel("video-stale").status == "cancelled"

    _seed_job(
        store,
        job_id="video-final",
        project_id="project-1",
        kind="video-processing",
        subject_id="asset-final",
    )
    claim = store.claim("video-final", "runner", ttl_seconds=10)
    assert claim is not None and claim.lease_token
    root, child, segment = _prepared_video_graph("asset-final", "generation-b")
    with pytest.raises(ValueError, match="empty or duplicate Segment ids"):
        video.publish_result(
            "video-final",
            claim.lease_token,
            root_scene=root,
            child_scenes=[child],
            segments=[{**segment, "id": ""}],
            frame_count=80,
            generation_key="generation-b",
            stage="Ready",
        )
    with pytest.raises(ValueError, match="empty or duplicate Segment ids"):
        video.publish_result(
            "video-final",
            claim.lease_token,
            root_scene=root,
            child_scenes=[child],
            segments=[segment, dict(segment)],
            frame_count=80,
            generation_key="generation-b",
            stage="Ready",
        )
    assert video.publish_result(
        "video-final",
        claim.lease_token,
        root_scene=root,
        child_scenes=[child],
        segments=[segment],
        frame_count=80,
        generation_key="generation-b",
        stage="Ready",
        state={"phase": "complete"},
    )
    with sessions() as session:
        asset = session.get(VideoAssetRow, "asset-final")
        assert asset is not None
        assert (asset.status, asset.generation_key, asset.scene_id) == (
            "ready",
            "generation-b",
            root["id"],
        )
        assert session.get(AnalysisRunRow, "video-final").status == "succeeded"
        assert session.get(PipelineJobRow, "video-final").status == "succeeded"
        assert session.get(ProjectSceneRow, ("project-1", root["id"])) is not None
        assert session.scalar(
            select(SegmentRow).where(SegmentRow.project_id == "project-1")
        ).scene_id == child["id"]


def _model_comparison_scene(scene_id: str) -> dict:
    return make_video_scene(
        scene_id=scene_id,
        title="Detection benchmark",
        duration=5.0,
        video_asset={
            "id": "asset-comparison",
            "filename": "comparison.mp4",
            "generationKey": "generation-comparison",
            "selectedSegmentId": "shot-1",
            "sourceStart": 0.0,
            "sourceEnd": 5.0,
            "reconstruction": {"status": "ready", "model": "yolo26m.pt"},
        },
    )


def test_model_comparison_is_durable_deduplicated_and_published_atomically(tmp_path):
    clock = MutableClock()
    store, _other, engine = _stores(tmp_path, clock)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    scenes = SceneRepository(sessions)
    model_comparison_pipeline = ModelComparisonPipelineService(
        store._session_factory,
        clock=clock,
    )
    original = scenes.put(_model_comparison_scene("comparison-scene"))
    with sessions.begin() as session:
        session.add(
            ProjectSceneRow(
                project_id="project-1",
                scene_id="comparison-scene",
                role="segment",
            )
        )

    queued = model_comparison_pipeline.enqueue(
        job_id="comparison-run",
        project_id="project-1",
        scene_id="comparison-scene",
        baseline_model="yolo26n.pt",
        candidate_model="yolo26m.pt",
    )
    duplicate = model_comparison_pipeline.enqueue(
        job_id="comparison-duplicate",
        project_id="project-1",
        scene_id="comparison-scene",
        baseline_model="yolo26n.pt",
        candidate_model="yolo26m.pt",
    )
    assert duplicate.id == queued.id == "comparison-run"

    claim = store.claim("comparison-run", "pipeline-runner", ttl_seconds=10)
    assert claim is not None and claim.lease_token
    report = {
        "sceneId": "comparison-scene",
        "frameCount": 50,
        "comparison": {"verdict": "candidate"},
    }
    assert model_comparison_pipeline.publish(
        "comparison-run",
        claim.lease_token,
        report=report,
    )

    published = scenes.get("comparison-scene")
    assert published["revision"] == original["revision"] + 1
    assert published["payload"]["videoAsset"]["reconstruction"][
        "modelComparison"
    ] == report
    with sessions() as session:
        assert session.get(PipelineJobRow, "comparison-run").status == "succeeded"
        assert session.get(AnalysisRunRow, "comparison-run").status == "succeeded"


def test_model_comparison_rejects_a_report_for_a_newer_scene_revision(tmp_path):
    clock = MutableClock()
    store, _other, engine = _stores(tmp_path, clock)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    scenes = SceneRepository(sessions)
    model_comparison_pipeline = ModelComparisonPipelineService(
        store._session_factory,
        clock=clock,
    )
    scenes.put(_model_comparison_scene("comparison-stale"))
    with sessions.begin() as session:
        session.add(
            ProjectSceneRow(
                project_id="project-1",
                scene_id="comparison-stale",
                role="segment",
            )
        )
    model_comparison_pipeline.enqueue(
        job_id="comparison-stale-run",
        project_id="project-1",
        scene_id="comparison-stale",
        baseline_model="yolo26n.pt",
        candidate_model="yolo26m.pt",
    )
    claim = store.claim(
        "comparison-stale-run", "pipeline-runner", ttl_seconds=10
    )
    assert claim is not None and claim.lease_token

    changed = scenes.get("comparison-stale")
    changed["title"] = "Edited while comparison was running"
    scenes.put(changed)
    assert not model_comparison_pipeline.publish(
        "comparison-stale-run",
        claim.lease_token,
        report={
            "sceneId": "comparison-stale",
            "comparison": {"verdict": "candidate"},
        },
    )
    assert "modelComparison" not in (
        scenes.get("comparison-stale")["payload"]["videoAsset"]["reconstruction"]
    )
    with sessions() as session:
        job = session.get(PipelineJobRow, "comparison-stale-run")
        run = session.get(AnalysisRunRow, "comparison-stale-run")
        assert job.status == run.status == "failed"
        assert "Scene changed" in str(job.error)


def test_model_comparison_can_be_queued_again_after_a_terminal_run(tmp_path):
    clock = MutableClock()
    store, _other, engine = _stores(tmp_path, clock)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    scenes = SceneRepository(sessions)
    model_comparison_pipeline = ModelComparisonPipelineService(
        store._session_factory,
        clock=clock,
    )
    scenes.put(_model_comparison_scene("comparison-rerun"))
    with sessions.begin() as session:
        session.add(
            ProjectSceneRow(
                project_id="project-1",
                scene_id="comparison-rerun",
                role="segment",
            )
        )

    first = model_comparison_pipeline.enqueue(
        job_id="comparison-first",
        project_id="project-1",
        scene_id="comparison-rerun",
        baseline_model="yolo26n.pt",
        candidate_model="yolo26m.pt",
    )
    first_claim = store.claim(first.id, "pipeline-runner", ttl_seconds=10)
    assert first_claim is not None and first_claim.lease_token
    assert _terminals(store).fail(
        first.id,
        first_claim.lease_token,
        "synthetic failure",
    )

    second = model_comparison_pipeline.enqueue(
        job_id="comparison-second",
        project_id="project-1",
        scene_id="comparison-rerun",
        baseline_model="yolo26n.pt",
        candidate_model="yolo26m.pt",
    )

    assert second.id == "comparison-second"
    assert second.status == "queued"
    with sessions() as session:
        assert session.get(PipelineJobRow, "comparison-first") is None
        assert session.get(AnalysisRunRow, "comparison-first").status == "failed"
        assert session.get(PipelineJobRow, "comparison-second").status == "queued"
        assert session.get(AnalysisRunRow, "comparison-second").status == "queued"


def _pipeline_job(state: dict) -> PipelineJob:
    return PipelineJob(
        id="multi-run",
        project_id="project-1",
        kind="multi-pass",
        subject_id="composite",
        status="processing",
        state=state,
        parameters={},
        available_at=0.0,
        attempts=1,
        error=None,
        requested_at=0.0,
        updated_at=0.0,
        lease_token="token",
    )


def test_multi_pass_retry_does_not_duplicate_child_reconstruction_jobs(monkeypatch):
    import app.multi_pass_job as multi_pass_job

    segments = [
        {"id": "a", "segmentId": "a", "sceneId": "child-a", "label": "A"},
        {"id": "b", "segmentId": "b", "sceneId": "child-b", "label": "B"},
    ]
    composite = {
        "id": "composite",
        "payload": {
            "videoAsset": {
                "multiPass": {"sourcePasses": segments, "status": "queued"},
                "reconstruction": {"status": "queued", "runId": "multi-run"},
            }
        },
    }
    children = {
        scene_id: {
            "id": scene_id,
            "payload": {"videoAsset": {"reconstruction": {}}},
        }
        for scene_id in ("child-a", "child-b")
    }
    status_by_scene: dict[str, str] = {}
    queued: list[str] = []

    class FakeSceneRepository:
        def get(self, scene_id):
            if scene_id == "composite":
                return composite
            return children.get(scene_id)

        def put(self, scene):
            return scene

    class FakeReconstructionJobs:
        def statuses(self, scene_ids):
            return {
                scene_id: status_by_scene[scene_id]
                for scene_id in scene_ids
                if scene_id in status_by_scene
            }

    def queue(child, *, match_snapshot=None):
        assert match_snapshot is None
        queued.append(child["id"])
        status_by_scene[child["id"]] = "queued"
        return child

    monkeypatch.setattr(multi_pass_job, "scenes", FakeSceneRepository())
    monkeypatch.setattr(
        multi_pass_job,
        "reconstruction_jobs",
        FakeReconstructionJobs(),
    )
    monkeypatch.setattr(multi_pass_job, "queue_reconstruction", queue)

    first = multi_pass_job.advance_multi_pass_pipeline_job(
        _pipeline_job({"phase": "prepare"})
    )
    # Simulate a crash before the parent persisted its waiting state. The child
    # jobs are already durable, so replaying prepare observes them and queues
    # nothing a second time.
    second = multi_pass_job.advance_multi_pass_pipeline_job(
        _pipeline_job({"phase": "prepare"})
    )

    assert first["status"] == second["status"] == "waiting"
    assert queued == ["child-a", "child-b"]


def test_multi_pass_waiting_poll_reads_only_compact_dependency_rows(monkeypatch):
    import app.multi_pass_job as multi_pass_job

    sources = [
        {"id": "a", "segmentId": "a", "sceneId": "child-a"},
        {"id": "b", "segmentId": "b", "sceneId": "child-b"},
    ]

    class NoSceneReads:
        def get(self, _scene_id):
            raise AssertionError("waiting dependency poll must not load Scene payload")

    class CompactDependencies:
        def statuses(self, scene_ids):
            assert scene_ids == ["child-a", "child-b"]
            return {"child-a": "processing", "child-b": "queued"}

    monkeypatch.setattr(multi_pass_job, "scenes", NoSceneReads())
    monkeypatch.setattr(
        multi_pass_job, "reconstruction_jobs", CompactDependencies()
    )
    outcome = multi_pass_job.advance_multi_pass_pipeline_job(
        _pipeline_job({"phase": "dependencies", "sources": sources})
    )

    assert outcome["status"] == "waiting"
    assert outcome["state"]["dependencyStatuses"] == {
        "child-a": "processing",
        "child-b": "queued",
    }
