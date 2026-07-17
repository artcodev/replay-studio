from __future__ import annotations

from copy import deepcopy


PROJECT_BINDING_METADATA_FIELDS = {
    "scope",
    "projectSceneId",
    "inherited",
}

MULTI_PASS_MATCH_BINDING_STALE_WARNING = (
    "Project match data changed after this multi-angle result was fused. "
    "The existing composite remains available for review; rerun multi-angle "
    "analysis after its source shots are ready to refresh identity fusion."
)


def project_parent_scene_id(scene: dict) -> str | None:
    """Return the direct project owner advertised by a derived scene."""

    video = scene.get("payload", {}).get("videoAsset") or {}
    multi_pass = video.get("multiPass") or {}
    parent_id = multi_pass.get("parentSceneId") or video.get("parentSceneId")
    normalized = str(parent_id or "").strip()
    return normalized or None


def is_multi_pass_scene(scene: dict) -> bool:
    video = scene.get("payload", {}).get("videoAsset") or {}
    return bool(video.get("multiPass"))


def is_single_pass_reconstruction_scene(scene: dict) -> bool:
    video = scene.get("payload", {}).get("videoAsset") or {}
    return bool(video.get("selectedSegmentId")) and not bool(video.get("multiPass"))


def mark_multi_pass_match_binding_stale(scene: dict) -> None:
    """Keep a composite visible while declaring its identity fusion stale."""

    video = scene.get("payload", {}).get("videoAsset") or {}
    multi_pass = video.get("multiPass") or {}
    if not multi_pass:
        return
    multi_pass["matchBindingState"] = "stale"
    warnings = list(multi_pass.get("warnings") or [])
    if MULTI_PASS_MATCH_BINDING_STALE_WARNING not in warnings:
        warnings.append(MULTI_PASS_MATCH_BINDING_STALE_WARNING)
    multi_pass["warnings"] = warnings

    reconstruction = video.get("reconstruction") or {}
    reconstruction["qualityVerdict"] = "review"
    reconstruction_warnings = list(reconstruction.get("warnings") or [])
    if MULTI_PASS_MATCH_BINDING_STALE_WARNING not in reconstruction_warnings:
        reconstruction_warnings.append(MULTI_PASS_MATCH_BINDING_STALE_WARNING)
    reconstruction["warnings"] = reconstruction_warnings


def project_match_binding(
    binding: dict | None,
    project_scene_id: str,
    *,
    inherited: bool,
) -> dict | None:
    """Decorate one canonical snapshot for storage on a project member.

    The binding contents stay identical on every scene.  Only the ownership
    metadata differs, making the root the authoritative copy while keeping
    derived scenes self-contained for reconstruction workers and old clients.
    """

    if not isinstance(binding, dict) or not binding:
        return None
    result = deepcopy(binding)
    result.update(
        {
            "scope": "project",
            "projectSceneId": str(project_scene_id),
            "inherited": bool(inherited),
        }
    )
    return result


def semantic_match_binding(binding: dict | None) -> dict | None:
    """Strip storage-only ownership fields from reconstruction inputs."""

    if not isinstance(binding, dict) or not binding:
        return None
    return {
        key: deepcopy(value)
        for key, value in binding.items()
        if key not in PROJECT_BINDING_METADATA_FIELDS
    }


def copy_project_match_metadata(
    target: dict,
    source: dict,
    *,
    project_scene_id: str,
    inherited: bool,
) -> None:
    """Copy match and team metadata without replacing scene-local colors."""

    source_payload = source.get("payload", {})
    target_payload = target.setdefault("payload", {})
    binding = project_match_binding(
        source_payload.get("matchBinding"),
        project_scene_id,
        inherited=inherited,
    )
    if binding:
        target_payload["matchBinding"] = binding

    source_teams = source_payload.get("teams") or []
    target_teams = target_payload.get("teams") or []
    for index, target_team in enumerate(target_teams[:2]):
        if index >= len(source_teams):
            continue
        source_team = source_teams[index]
        target_team["name"] = source_team.get("name", target_team.get("name"))
        target_team["externalTeamId"] = source_team.get("externalTeamId")
