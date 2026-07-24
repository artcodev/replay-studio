from __future__ import annotations

"""Execute one already-claimed generic pipeline job."""

import argparse
from threading import Event, Thread

from .analysis_frame_generation import prepare_analysis_frame_generation
from .analysis_frame_generation_pipeline import analysis_frame_generation_pipeline
from .config import get_settings
from .model_comparison import compare_scene_models
from .model_comparison_pipeline_service import ModelComparisonPipelineService
from .multi_pass_job import advance_multi_pass_pipeline_job
from .multi_pass_pipeline_service import MultiPassPipelineService
from .pipeline_domain import PipelineJob
from .pipeline_store import pipeline_store
from .pipeline_terminal_service import pipeline_terminals
from .scene_document import scene_revision
from .scene_repository import scenes
from .video_ingest_preparation import prepare_video_generation
from .video_processing_contract import VideoProcessingCancelled
from .video_pipeline import video_pipeline


model_comparison_pipeline = ModelComparisonPipelineService()
multi_pass_pipeline = MultiPassPipelineService()


class PipelineClaimLost(RuntimeError):
    pass


class _PipelineHeartbeat:
    def __init__(self, job_id: str, token: str) -> None:
        settings = get_settings()
        ttl = max(1.0, float(settings.pipeline_lease_ttl_seconds))
        self.interval = min(
            max(0.05, float(settings.pipeline_lease_heartbeat_seconds)),
            max(0.05, ttl / 3.0),
        )
        self.job_id = job_id
        self.token = token
        self._stop = Event()
        self._lost = Event()
        self._thread = Thread(
            target=self._run,
            name=f"pipeline-heartbeat-{job_id}",
            daemon=True,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                current = pipeline_store.heartbeat(self.job_id, self.token)
            except Exception:
                # A transient database fault is retried. Fenced publication and
                # supervisor expiry still prevent a stale result from winning.
                continue
            if not current:
                self._lost.set()
                return

    def current(self) -> bool:
        return not self._lost.is_set() and pipeline_store.is_claim_current(
            self.job_id, self.token
        )

    def __enter__(self) -> "_PipelineHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join(timeout=max(0.1, self.interval * 2.0))


def _execute_video(job: PipelineJob, token: str, heartbeat: _PipelineHeartbeat) -> None:
    prepared = prepare_video_generation(
        job.subject_id,
        str(job.parameters.get("title") or "") or None,
        claim_check=heartbeat.current,
        progress_writer=lambda values: video_pipeline.update_progress(
            job.id, token, values
        ),
        staging_key=token,
    )
    if not heartbeat.current():
        raise PipelineClaimLost("Video-processing claim was fenced")
    if prepared is None:
        message = "Video asset was not found"
        pipeline_terminals.fail(job.id, token, message)
        return
    prepared.validate()
    if not heartbeat.current():
        raise PipelineClaimLost("Video-processing claim was fenced")
    published = video_pipeline.publish_result(
        job.id,
        token,
        root_scene=prepared.root_scene,
        child_scenes=prepared.child_scenes,
        segments=prepared.segments,
        frame_count=int(prepared.asset.get("frame_count") or 0),
        generation_key=prepared.generation_key,
        stage=prepared.stage,
        state={"phase": "complete"},
    )
    if not published:
        raise PipelineClaimLost("Video-processing claim was fenced")


def _execute_analysis_frame_generation(
    job: PipelineJob,
    token: str,
    heartbeat: _PipelineHeartbeat,
) -> None:
    prepared = prepare_analysis_frame_generation(
        job.subject_id,
        staging_key=token,
        claim_check=heartbeat.current,
        progress_writer=lambda progress: pipeline_store.update_progress(
            job.id,
            token,
            progress,
        ),
    )
    if not heartbeat.current():
        raise PipelineClaimLost("Analysis-frame generation claim was fenced")
    published = analysis_frame_generation_pipeline.publish(job.id, token, prepared)
    if not published:
        raise PipelineClaimLost("Analysis-frame generation claim was fenced")


def _execute_multi_pass(job: PipelineJob, token: str) -> None:
    outcome = advance_multi_pass_pipeline_job(job)
    status = str(outcome.get("status") or "failed")
    if status == "waiting":
        pipeline_store.yield_waiting(
            job.id,
            token,
            state=dict(outcome.get("state") or {}),
            delay_seconds=get_settings().pipeline_dependency_poll_seconds,
            progress=dict(outcome.get("progress") or {}),
        )
        return
    if status == "succeeded":
        scene = outcome.get("scene")
        if not isinstance(scene, dict):
            raise RuntimeError("Multi-pass completion produced no Scene")
        multi_pass_pipeline.publish(
            job.id,
            token,
            scene=scene,
            status="succeeded",
            state=dict(outcome.get("state") or {}),
        )
        return
    message = str(outcome.get("error") or "Multi-pass analysis failed")
    failed_scene = outcome.get("scene")
    if isinstance(failed_scene, dict):
        multi_pass_pipeline.publish(
            job.id,
            token,
            scene=failed_scene,
            status="failed",
            state=dict(outcome.get("state") or job.state or {}),
            error=message,
        )
    else:
        pipeline_terminals.fail(job.id, token, message)


def _execute_model_comparison(
    job: PipelineJob,
    token: str,
    heartbeat: _PipelineHeartbeat,
) -> None:
    scene = scenes.get(job.subject_id)
    if scene is None:
        pipeline_terminals.fail(job.id, token, "Scene was not found")
        return
    expected_revision = int(job.parameters.get("sceneRevision", -1))
    if scene_revision(scene) != expected_revision:
        pipeline_terminals.fail(
            job.id,
            token,
            "Scene changed before detection model comparison started; run it again",
        )
        return

    def publish_progress(phase: str, completed: int, total: int) -> None:
        if not heartbeat.current():
            raise PipelineClaimLost("Model-comparison claim was fenced")
        labels = {
            "baseline": "Running baseline detector",
            "candidate": "Running candidate detector",
        }
        pipeline_store.update_progress(
            job.id,
            token,
            {
                "phase": phase,
                "phaseIndex": completed + 1,
                "phaseCount": total,
                "label": labels.get(phase, "Comparing detection models"),
                "completed": completed,
                "total": total,
                "phasePercent": 0,
                "overallPercent": round(completed / max(1, total) * 100),
                "etaSeconds": None,
            },
        )

    report = compare_scene_models(scene, on_progress=publish_progress)
    if not heartbeat.current():
        raise PipelineClaimLost("Model-comparison claim was fenced")
    published = model_comparison_pipeline.publish(
        job.id,
        token,
        report=report,
    )
    if not published:
        # A stale Scene revision is terminalized as failed by the publisher;
        # a lost lease is simply fenced. In either case no API process ran
        # inference and no stale report reached the Scene.
        return


def execute_claimed_pipeline_job(job_id: str, token: str) -> bool:
    job = pipeline_store.get(job_id)
    if (
        job is None
        or job.status != "processing"
        or job.lease_token != token
        or not pipeline_store.is_claim_current(job_id, token)
    ):
        return False
    try:
        with _PipelineHeartbeat(job_id, token) as heartbeat:
            if job.kind == "video-processing":
                _execute_video(job, token, heartbeat)
            elif job.kind == "analysis-frame-generation":
                _execute_analysis_frame_generation(job, token, heartbeat)
            elif job.kind == "multi-pass":
                _execute_multi_pass(job, token)
            elif job.kind == "model-comparison":
                _execute_model_comparison(job, token, heartbeat)
            else:  # schema/store validation should make this unreachable
                raise RuntimeError(f"Unsupported pipeline job kind: {job.kind}")
    except (PipelineClaimLost, VideoProcessingCancelled):
        return False
    except Exception as exc:
        pipeline_terminals.fail(job.id, token, str(exc))
        return True
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("lease_token")
    arguments = parser.parse_args(argv)
    execute_claimed_pipeline_job(arguments.job_id, arguments.lease_token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
