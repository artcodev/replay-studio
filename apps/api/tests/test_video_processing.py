from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.video_ingest_preparation as video_preparation
from app.analysis_run_repository import AnalysisRunRepository
from app.database import Base, VideoAssetRow
from app.analysis_run_contract import AnalysisRunCreate
from app.project_lifecycle_contract import ProjectCreate
from app.project_resource_repository import ProjectResourceRepository
from app.project_store import ProjectStore
from app.video_ingest_preparation import prepare_video_generation
from app.video_processing_contract import (
    PreparedVideoGeneration,
    VideoProcessingCancelled,
    VideoProcessingError,
    video_processing_run_id,
)
from app.video_segment_planning import rank_reconstruction_shots


def _redirect_asset_paths(monkeypatch, directory: Path) -> None:
    monkeypatch.setattr(video_preparation, "asset_directory", lambda _asset_id: directory)
    monkeypatch.setattr(
        video_preparation,
        "video_generation_directory",
        lambda _asset_id, generation_key: (
            directory / ".pipeline-runs" / generation_key
        ),
    )


def _stub_media_generation(monkeypatch) -> None:
    monkeypatch.setattr(video_preparation, "require_ffmpeg", lambda: None)

    def create_proxy(_source: Path, destination: Path) -> None:
        destination.write_bytes(b"proxy")

    def create_poster(
        _source: Path,
        destination: Path,
        *,
        at_seconds: float,
    ) -> None:
        assert at_seconds >= 0
        destination.write_bytes(b"poster")

    def sample_frames(
        _source: Path,
        destination_pattern: Path,
        *,
        fps: float,
    ) -> None:
        assert fps > 0
        (destination_pattern.parent / "frame_00001.jpg").write_bytes(b"frame")

    monkeypatch.setattr(video_preparation, "create_browser_proxy", create_proxy)
    monkeypatch.setattr(video_preparation, "create_poster", create_poster)
    monkeypatch.setattr(video_preparation, "sample_detector_frames", sample_frames)


def test_rank_reconstruction_shots_marks_top_eligible_segments():
    segments = [
        {"id": "short", "duration": 3.9, "score": 1.0},
        {"id": "steady", "duration": 6.0, "score": 0.8},
        {"id": "best", "duration": 5.0, "score": 0.95},
        {"id": "third", "duration": 7.0, "score": 0.7},
    ]

    ranked = rank_reconstruction_shots(segments, limit=2)

    assert [segment["id"] for segment in ranked] == ["best", "steady"]
    assert [segment["recommended"] for segment in segments] == [False, True, True, False]


def test_ingest_stops_after_preparing_recommended_scenes(monkeypatch, tmp_path):
    asset_id = "asset-ingest-only"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    asset = {
        "id": asset_id,
        "filename": source.name,
        "original_name": "match-highlight.mp4",
    }
    updates: list[dict] = []

    _redirect_asset_paths(monkeypatch, tmp_path)
    _stub_media_generation(monkeypatch)
    monkeypatch.setattr(
        video_preparation,
        "get_settings",
        lambda: SimpleNamespace(
            max_video_duration=60.0,
            analysis_frame_rate=10.0,
        ),
    )
    monkeypatch.setattr(
        video_preparation,
        "probe_video",
        lambda _source: {"duration": 8.0, "width": 1920, "height": 1080, "fps": 25.0},
    )
    segments = [
        {"id": "shot-01", "label": "Shot 01", "start": 0.0, "end": 8.0, "duration": 8.0, "score": 1.0}
    ]
    monkeypatch.setattr(video_preparation, "detect_shots", lambda *_args: segments)
    monkeypatch.setattr(
        video_preparation,
        "propose_segment_layout",
        lambda *_args: {"groups": [], "warnings": []},
    )

    monkeypatch.setattr(video_preparation.video_store, "get", lambda _asset_id: asset)
    monkeypatch.setattr(
        video_preparation.video_store,
        "update",
        lambda *_args, **_values: pytest.fail(
            "claimed processing must not write VideoAsset directly"
        ),
    )

    def write_progress(values: dict) -> bool:
        updates.append(values)
        return True

    result = prepare_video_generation(
        asset_id,
        "Prepared highlight",
        claim_check=lambda: True,
        progress_writer=write_progress,
        staging_key="lease-ingest-only",
    )

    assert isinstance(result, PreparedVideoGeneration)
    assert result.root_scene["payload"]["tracks"] == []
    assert len(result.child_scenes) == 1
    assert result.child_scenes[0]["payload"]["tracks"] == []
    assert (
        result.root_scene["payload"]["videoAsset"]["generationKey"]
        == "lease-ingest-only"
    )
    result.validate()
    assert all(update.get("stage") != "Reconstructing players and ball" for update in updates)
    assert updates[-1] == {
        "stage": "Ready for reconstruction · 1 recommended moment",
        "progress": 99,
        "frame_count": 1,
    }


@pytest.fixture
def normalized_video_processing(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    store = ProjectStore(sessions)
    resources = ProjectResourceRepository(sessions)
    runs = AnalysisRunRepository(sessions)
    asset_id = "asset-processing-run"
    with sessions.begin() as session:
        session.add(
            VideoAssetRow(
                id=asset_id,
                filename="source.mp4",
                original_name="highlight.mp4",
                content_type="video/mp4",
                status="queued",
                stage="Waiting for FFmpeg",
                progress=2,
            )
        )
    store.create_project(ProjectCreate(id="project-processing", title="Processing"))
    resources.link_video_asset("project-processing", asset_id)
    run_id = video_processing_run_id("project-processing", asset_id)
    runs.create(
        "project-processing",
        AnalysisRunCreate(
            id=run_id,
            kind="video-processing",
            status="queued",
            source_run_id=asset_id,
            progress={
                "phase": "upload-complete",
                "label": "Waiting to process video",
                "completed": 0,
                "total": 100,
                "overallPercent": 0,
            },
        ),
    )
    state = {
        "id": asset_id,
        "filename": "source.mp4",
        "original_name": "highlight.mp4",
        "status": "queued",
        "stage": "Waiting for FFmpeg",
        "progress": 2,
    }

    def get_asset(requested_id: str):
        return dict(state) if requested_id == asset_id else None

    monkeypatch.setattr(video_preparation.video_store, "get", get_asset)
    monkeypatch.setattr(
        video_preparation.video_store,
        "update",
        lambda *_args, **_values: pytest.fail(
            "claimed processing must not write VideoAsset directly"
        ),
    )
    yield runs, state, asset_id, run_id
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_video_processing_refuses_an_unfenced_execution_contract(
    normalized_video_processing,
) -> None:
    runs, state, asset_id, run_id = normalized_video_processing

    with pytest.raises(TypeError):
        prepare_video_generation(asset_id)
    with pytest.raises(VideoProcessingError, match="claim check"):
        prepare_video_generation(
            asset_id,
            claim_check=None,  # type: ignore[arg-type]
            progress_writer=lambda _values: True,
            staging_key="lease-invalid-claim",
        )
    with pytest.raises(
        VideoProcessingError,
        match="progress writer",
    ):
        prepare_video_generation(
            asset_id,
            claim_check=lambda: True,
            progress_writer=None,  # type: ignore[arg-type]
            staging_key="lease-invalid-writer",
        )
    with pytest.raises(VideoProcessingError, match="staging key"):
        prepare_video_generation(
            asset_id,
            claim_check=lambda: True,
            progress_writer=lambda _values: True,
            staging_key="direct",
        )

    queued = runs.get(run_id)
    assert queued is not None and queued.status == "queued"
    assert state["status"] == "queued"


def test_video_processing_refuses_a_reused_staging_key(
    normalized_video_processing,
    monkeypatch,
    tmp_path,
) -> None:
    runs, state, asset_id, run_id = normalized_video_processing
    used = tmp_path / ".pipeline-runs" / "lease-already-used"
    used.mkdir(parents=True)
    progress_calls: list[dict] = []
    _redirect_asset_paths(monkeypatch, tmp_path)

    with pytest.raises(
        VideoProcessingError,
        match="already been used",
    ):
        prepare_video_generation(
            asset_id,
            claim_check=lambda: True,
            progress_writer=lambda values: progress_calls.append(values) or True,
            staging_key="lease-already-used",
        )

    assert progress_calls == []
    assert state["status"] == "queued"
    assert runs.get(run_id).status == "queued"


def test_fenced_queued_video_processing_claim_prevents_work(
    normalized_video_processing,
    monkeypatch,
    tmp_path,
) -> None:
    runs, state, asset_id, run_id = normalized_video_processing
    _redirect_asset_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        video_preparation,
        "probe_video",
        lambda _source: pytest.fail("cancelled queued work must not probe the video"),
    )
    with pytest.raises(VideoProcessingCancelled):
        prepare_video_generation(
            asset_id,
            claim_check=lambda: False,
            progress_writer=lambda _values: pytest.fail(
                "a fenced claim must not write progress"
            ),
            staging_key="lease-fenced-before-start",
        )

    assert state["status"] == "queued"
    assert runs.get(run_id).status == "queued"


def test_running_video_processing_acknowledges_fenced_claim_at_next_phase(
    normalized_video_processing,
    monkeypatch,
    tmp_path,
) -> None:
    runs, state, asset_id, run_id = normalized_video_processing
    _redirect_asset_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(video_preparation, "require_ffmpeg", lambda: None)

    claim_is_current = True

    def probe_and_fence(_source: Path) -> dict:
        nonlocal claim_is_current
        assert runs.get(run_id).status == "queued"
        claim_is_current = False
        return {"duration": 8.0, "width": 1920, "height": 1080, "fps": 25.0}

    monkeypatch.setattr(video_preparation, "probe_video", probe_and_fence)
    monkeypatch.setattr(
        video_preparation,
        "create_browser_proxy",
        lambda *_args: pytest.fail("cancellation must stop before FFmpeg proxy work"),
    )

    def write_progress(values: dict) -> bool:
        state.update(values)
        state["status"] = "processing"
        return True

    with pytest.raises(VideoProcessingCancelled):
        prepare_video_generation(
            asset_id,
            claim_check=lambda: claim_is_current,
            progress_writer=write_progress,
            staging_key="lease-fenced-after-probe",
        )

    assert state["status"] == "processing"


def test_video_processing_claim_fence_wins_status_update_race(
    normalized_video_processing,
    monkeypatch,
    tmp_path,
) -> None:
    runs, state, asset_id, run_id = normalized_video_processing
    _redirect_asset_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(video_preparation, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(
        video_preparation,
        "probe_video",
        lambda _source: pytest.fail("the raced cancellation must stop before probing"),
    )
    raced = False
    claim_is_current = True

    def fence_after_progress(_values):
        nonlocal raced, claim_is_current
        if not raced:
            raced = True
            claim_is_current = False
        return False

    with pytest.raises(VideoProcessingCancelled):
        prepare_video_generation(
            asset_id,
            claim_check=lambda: claim_is_current,
            progress_writer=fence_after_progress,
            staging_key="lease-progress-race",
        )

    assert raced is True
    assert state["status"] == "queued"
    assert runs.get(run_id).status == "queued"


def test_claim_fenced_after_layout_publishes_no_scene_or_project_graph(
    normalized_video_processing,
    monkeypatch,
    tmp_path,
) -> None:
    _runs, state, asset_id, _run_id = normalized_video_processing
    claim_is_current = True
    _redirect_asset_paths(monkeypatch, tmp_path)
    _stub_media_generation(monkeypatch)
    monkeypatch.setattr(
        video_preparation,
        "probe_video",
        lambda _source: {
            "duration": 8.0,
            "width": 1920,
            "height": 1080,
            "fps": 25.0,
        },
    )
    segments = [
        {
            "id": "shot-01",
            "label": "Shot 01",
            "start": 0.0,
            "end": 8.0,
            "duration": 8.0,
            "score": 1.0,
        }
    ]
    monkeypatch.setattr(video_preparation, "detect_shots", lambda *_args: segments)

    def layout_then_fence(*_args):
        nonlocal claim_is_current
        claim_is_current = False
        return {"groups": []}

    monkeypatch.setattr(video_preparation, "propose_segment_layout", layout_then_fence)

    def write_progress(values: dict) -> bool:
        state.update(values)
        state["status"] = "processing"
        return True

    with pytest.raises(VideoProcessingCancelled):
        prepare_video_generation(
            asset_id,
            claim_check=lambda: claim_is_current,
            progress_writer=write_progress,
            staging_key="lease-fenced-after-layout",
        )

    assert state["status"] == "processing"
    assert state.get("scene_id") is None
