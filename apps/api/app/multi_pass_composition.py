from __future__ import annotations

"""Creation and durable enqueue of a multi-angle composition."""

from copy import deepcopy
from datetime import UTC, datetime
from uuid import uuid4

from .config import get_settings
from .multi_pass_domain import MultiPassError
from .multi_pass_pipeline_service import MultiPassPipelineService
from .multi_pass_progress import multi_pass_phases
from .pipeline_domain import PipelineJobConflict
from .project_match import match_snapshot_reference
from .project_match_repository import project_matches
from .project_resource_repository import (
    ProjectResourceNotFound,
    project_resources,
)
from .sample import make_video_scene
from .scene_document import reconstruction_input_fingerprint
from .scene_repository import scenes


multi_pass_pipeline = MultiPassPipelineService()


def create_project_multi_pass_scene(
    project_scene_id: str,
    source_passes: list[dict],
    *,
    project_id: str,
    title: str | None = None,
    manual_alignment_anchors: list[dict] | None = None,
    source_scenes: dict[str, dict] | None = None,
) -> dict:
    """Create a composition from explicit, project-owned scene dependencies."""

    normalized_project_id = str(project_id or "").strip()
    if not normalized_project_id:
        raise MultiPassError("Project id is required for multi-pass composition")
    if len(source_passes) < 2:
        raise MultiPassError("Choose at least two camera angles")
    scene_ids = [str(item.get("sceneId") or "") for item in source_passes]
    if any(not value for value in scene_ids) or len(set(scene_ids)) < 2:
        raise MultiPassError("Each camera angle must reference a different scene")
    required_scene_ids = list(dict.fromkeys([str(project_scene_id or ""), *scene_ids]))
    if any(not scene_id for scene_id in required_scene_ids):
        raise MultiPassError("Every multi-pass source must have a scene id")
    try:
        owned_scene_ids = {
            link.scene_id
            for link in project_resources.list_scene_links(normalized_project_id)
        }
    except ProjectResourceNotFound as exc:
        raise MultiPassError(f"Project {normalized_project_id} was not found") from exc
    invalid_scene_ids = [
        scene_id for scene_id in required_scene_ids if scene_id not in owned_scene_ids
    ]
    if invalid_scene_ids:
        raise MultiPassError(
            "Multi-pass source scenes are missing or owned by another project: "
            + ", ".join(invalid_scene_ids)
        )
    resolved_source_scenes = source_scenes or {
        scene_id: scenes.get(scene_id) for scene_id in scene_ids
    }
    missing = [
        scene_id
        for scene_id in scene_ids
        if resolved_source_scenes.get(scene_id) is None
    ]
    if missing:
        raise MultiPassError(f"Source scenes not found: {', '.join(missing)}")

    reference_pass = max(
        source_passes,
        key=lambda item: (
            float(item.get("score") or 0.0),
            float(item.get("duration") or 0.0),
        ),
    )
    reference_child = resolved_source_scenes[str(reference_pass["sceneId"])]
    assert reference_child is not None
    group_id = f"angles-{uuid4().hex[:8]}"
    run_id = uuid4().hex
    video = deepcopy(reference_child["payload"]["videoAsset"])
    video["processingState"] = "multi-pass-queued"
    video["multiPass"] = {
        "id": group_id,
        "status": "queued",
        "parentSceneId": project_scene_id,
        "selectedSegmentIds": [str(item["segmentId"]) for item in source_passes],
        "sourcePasses": deepcopy(source_passes),
        "referenceSceneId": None,
        "currentPass": 0,
        "passes": [],
        "consensus": None,
        "manualAlignmentAnchors": deepcopy(manual_alignment_anchors or []),
        "warnings": [],
    }
    video["reconstruction"] = {
        "status": "queued",
        "processingStatus": "queued",
        "model": get_settings().reconstruction_model,
        "runId": run_id,
        "runRevision": 1,
        "error": None,
        "progress": {
            "phase": "angle-1",
            "phaseIndex": 1,
            "phaseCount": len(source_passes) + 2,
            "label": "Waiting to analyze camera angles",
            "detail": f"Queued {len(source_passes)} selected views.",
            "completed": 0,
            "total": len(source_passes),
            "phasePercent": 0,
            "overallPercent": 0,
            "elapsedSeconds": 0.0,
            "etaSeconds": None,
            "updatedAt": datetime.now(UTC).isoformat(),
            "phases": multi_pass_phases(source_passes, 1),
        },
    }
    scene = make_video_scene(
        scene_id=f"multi-{uuid4().hex[:8]}",
        title=title or f"Multi-angle ({len(source_passes)} passes)",
        duration=reference_child["duration"],
        video_asset=video,
    )
    snapshot_ref = match_snapshot_reference(
        project_matches.current_summary(normalized_project_id)
    )
    composite_reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    if snapshot_ref is not None:
        composite_reconstruction["matchSnapshotRef"] = snapshot_ref
    composite_reconstruction["inputFingerprint"] = reconstruction_input_fingerprint(scene)
    try:
        return multi_pass_pipeline.enqueue(
            project_id=normalized_project_id,
            scene=scene,
            source_scene_ids=required_scene_ids,
        )
    except (PipelineJobConflict, ValueError) as exc:
        raise MultiPassError(str(exc)) from exc
