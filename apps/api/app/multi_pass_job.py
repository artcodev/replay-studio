from __future__ import annotations

"""Idempotent state machine for one durable multi-angle pipeline job."""

from datetime import UTC, datetime

from .multi_pass_domain import mark_multi_pass_failed, source_segments
from .multi_pass_finalization import finalize_multi_pass
from .multi_pass_metrics import pass_summary
from .multi_pass_progress import dependency_progress
from .pipeline_domain import PipelineJob
from .project_match import (
    reconstruction_match_snapshot_reference,
    snapshot_matches_reference,
)
from .project_match_repository import project_matches
from .reconstruction_artifact_hydration import hydrate_scene_reconstruction
from .reconstruction_errors import ReconstructionError
from .reconstruction_job_queries import reconstruction_jobs
from .reconstruction_queue import queue_reconstruction
from .scene_repository import scenes


def _failed(scene: dict, message: str, passes: list[dict] | None = None) -> dict:
    mark_multi_pass_failed(scene, message, passes)
    return {"status": "failed", "error": message, "scene": scene}


def _prepare(job: PipelineJob) -> dict:
    scene = scenes.get(job.subject_id)
    if scene is None:
        return {"status": "failed", "error": "Multi-pass scene was not found"}
    segments = source_segments(scene)
    if len(segments) < 2:
        return _failed(scene, "At least two explicit source camera angles are required")

    video = scene["payload"]["videoAsset"]
    video["multiPass"].update(
        {"status": "processing", "currentPass": 0, "passes": []}
    )
    reconstruction = video.get("reconstruction") or {}
    reconstruction.update(
        {
            "status": "processing",
            "processingStatus": "processing",
            "startedAt": reconstruction.get("startedAt")
            or datetime.now(UTC).isoformat(),
            "error": None,
        }
    )
    video["reconstruction"] = reconstruction

    snapshot = None
    snapshot_ref = reconstruction_match_snapshot_reference(scene)
    if snapshot_ref is not None:
        snapshot = project_matches.get_snapshot(
            job.project_id,
            str(snapshot_ref["id"]),
        )
        if snapshot is None or not snapshot_matches_reference(snapshot, snapshot_ref):
            return _failed(scene, "The captured match snapshot is unavailable or invalid")

    scene_ids = [str(segment["sceneId"]) for segment in segments]
    statuses = reconstruction_jobs.statuses(scene_ids)
    for segment in segments:
        child_scene_id = str(segment["sceneId"])
        if child_scene_id in statuses:
            continue
        child = scenes.get(child_scene_id)
        if child is None:
            return _failed(scene, f"Source camera scene {child_scene_id} was not found")
        try:
            queue_reconstruction(child, match_snapshot=snapshot)
        except ReconstructionError as exc:
            return _failed(scene, str(exc))

    statuses = reconstruction_jobs.statuses(scene_ids)
    missing = [scene_id for scene_id in scene_ids if scene_id not in statuses]
    if missing:
        return _failed(
            scene,
            f"Child reconstruction jobs were not persisted: {', '.join(missing)}",
        )
    return {
        "status": "waiting",
        "state": {
            "phase": "dependencies",
            "sources": segments,
            "dependencyStatuses": statuses,
        },
        "progress": dependency_progress(segments, statuses),
    }


def _collect_terminal_results(
    scene: dict,
    segments: list[dict],
    statuses: dict[str, str],
) -> tuple[list[dict], list[tuple[dict, dict, dict]]]:
    pass_summaries: list[dict] = []
    ready_scenes: list[tuple[dict, dict, dict]] = []
    for segment in segments:
        child_scene_id = str(segment["sceneId"])
        child = scenes.get(child_scene_id)
        if child is None:
            pass_summaries.append(
                {
                    "sceneId": child_scene_id,
                    "segmentId": segment.get("segmentId"),
                    "status": "failed",
                    "quality": 0.0,
                    "error": "Source scene was not found",
                }
            )
            continue
        if statuses[child_scene_id] != "ready":
            pass_summaries.append(
                pass_summary(
                    child,
                    segment,
                    status="failed",
                    error=f"Child reconstruction ended as {statuses[child_scene_id]}",
                )
            )
            continue
        try:
            hydrate_scene_reconstruction(child)
            summary = pass_summary(child, segment)
        except Exception as exc:
            summary = pass_summary(
                child,
                segment,
                status="failed",
                error=f"Artifact hydration failed: {exc}",
            )
        if summary.get("status") == "ready" and summary.get("qualityVerdict") in {
            "pass",
            "review",
        }:
            ready_scenes.append((child, summary, segment))
        elif summary.get("status") == "ready":
            summary["error"] = "Reconstruction completed without usable quality evidence"
        pass_summaries.append(summary)
    return pass_summaries, ready_scenes


def _poll_dependencies(job: PipelineJob, state: dict) -> dict:
    segments = [
        dict(item)
        for item in state.get("sources") or []
        if isinstance(item, dict)
    ]
    scene_ids = [str(item.get("sceneId") or "") for item in segments]
    statuses = reconstruction_jobs.statuses(scene_ids)
    missing = [scene_id for scene_id in scene_ids if scene_id not in statuses]
    if missing:
        return {
            "status": "failed",
            "error": (
                "Child reconstruction control rows disappeared: "
                + ", ".join(missing)
            ),
        }
    if any(statuses.get(scene_id) in {"queued", "processing"} for scene_id in scene_ids):
        return {
            "status": "waiting",
            "state": {**state, "dependencyStatuses": statuses},
            "progress": dependency_progress(segments, statuses),
        }

    scene = scenes.get(job.subject_id)
    if scene is None:
        return {"status": "failed", "error": "Multi-pass scene was not found"}
    pass_summaries, ready_scenes = _collect_terminal_results(
        scene,
        segments,
        statuses,
    )
    if not ready_scenes:
        return _failed(
            scene,
            "None of the selected camera angles produced a usable reconstruction",
            pass_summaries,
        )
    progress = finalize_multi_pass(scene, segments, ready_scenes, pass_summaries)
    return {
        "status": "succeeded",
        "state": {**state, "phase": "complete", "dependencyStatuses": statuses},
        "progress": progress,
        "scene": scene,
    }


def advance_multi_pass_pipeline_job(job: PipelineJob) -> dict:
    """Advance one state-machine transition without running child work inline."""

    state = dict(job.state or {})
    phase = str(state.get("phase") or "prepare")
    if phase == "prepare":
        return _prepare(job)
    if phase == "dependencies":
        return _poll_dependencies(job, state)
    return {"status": "failed", "error": f"Unknown multi-pass phase: {phase}"}
