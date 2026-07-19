from __future__ import annotations

"""Compact reconstruction telemetry read helpers and post-publication hooks."""

import logging
from time import sleep
from typing import Any

from .analysis_run_repository import AnalysisRunRepository, analysis_runs
from .project_identity_repository import (
    ProjectIdentityRepository,
    project_identities,
)
from .project_match_repository import ProjectMatchRepository, project_matches
from .project_resource_repository import (
    ProjectResourceRepository,
    project_resources,
)
from .analysis_run_contract import AnalysisRunUpdate
from .project_store import ProjectStore, project_store


class AnalysisCancellationRequested(RuntimeError):
    """Cooperative cancellation checkpoint raised by progress publication."""


logger = logging.getLogger(__name__)


def _reconstruction(scene: dict[str, Any]) -> dict[str, Any]:
    return (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
    )


def _run_id(scene: dict[str, Any]) -> str:
    return str(_reconstruction(scene).get("runId") or "")


def publish_reconstruction_progress(
    scene: dict[str, Any],
    progress: dict[str, Any],
    *,
    expected_run_id: str | None = None,
    expected_input_fingerprint: str | None = None,
    expected_lease_owner_id: str | None = None,
    run_repository=None,
) -> bool:
    """Publish progress through the compact lease-fenced store operation.

    Missing fencing tokens are not repaired or inferred.  Callers that run an
    in-memory analysis without a durable job simply receive ``False``.
    """

    run_id = str(expected_run_id or _run_id(scene))
    input_fingerprint = str(
        expected_input_fingerprint
        or _reconstruction(scene).get("inputFingerprint")
        or ""
    )
    owner_id = str(expected_lease_owner_id or "")
    scene_id = str(scene.get("id") or "")
    if not scene_id or not run_id or not input_fingerprint or not owner_id:
        return False
    if run_repository is None:
        from .reconstruction_run_repository import reconstruction_runs

        run_repository = reconstruction_runs

    outcome = run_repository.publish_reconstruction_progress(
        scene_id,
        run_id,
        input_fingerprint,
        owner_id,
        progress,
    )
    if outcome == "cancelled":
        raise AnalysisCancellationRequested("Analysis cancellation was requested")
    return outcome == "published"


def publish_reconstruction_terminal(
    scene: dict[str, Any],
    status: str,
    *,
    error: str | None = None,
    projects: ProjectStore = project_store,
    resources: ProjectResourceRepository = project_resources,
    runs: AnalysisRunRepository = analysis_runs,
    matches: ProjectMatchRepository = project_matches,
    identities: ProjectIdentityRepository = project_identities,
) -> bool:
    """Run idempotent work that follows the atomic terminal publication.

    Scene, job, lease and AnalysisRun terminal state are committed by
    ``ReconstructionRunRepository.put_if_reconstruction_run``. This hook only synchronizes
    project identities and records diagnostics for a successful result.
    """

    del error  # terminal errors are committed by the fenced run publication
    expected_status = {
        "ready": "succeeded",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(status)
    if expected_status is None:
        return False
    if status != "ready":
        return True

    scene_id = str(scene.get("id") or "")
    project_id = resources.scene_owner(scene_id) if scene_id else None
    if project_id is None:
        return False
    try:
        from .project_identity import sync_project_identities_from_scene

        report = None
        last_error: Exception | None = None
        # One transient DB hiccup must not lose the sync of a successful
        # run; a crash inside this window is repaired by the startup sweep
        # in recover_missed_identity_sync.
        for attempt in range(3):
            try:
                report = sync_project_identities_from_scene(
                    scene,
                    project_id=project_id,
                    projects=projects,
                    resources=resources,
                    matches=matches,
                    identities=identities,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    sleep(0.5 * (attempt + 1))
        if report is None:
            assert last_error is not None
            raise last_error
        identity_sync: dict[str, Any] = {
            "status": "succeeded",
            "peopleCreated": report.people_created,
            "peopleUpdated": report.people_updated,
            "membershipsCreated": report.memberships_created,
            "membershipsUpdated": report.memberships_updated,
            "membershipsPreserved": report.memberships_preserved,
            "unverifiedRosterBindingCount": report.unverified_roster_binding_count,
        }
    except Exception as exc:  # identity persistence cannot invalidate 3D output
        logger.exception(
            "Project identity sync failed for scene %s in project %s",
            scene_id,
            project_id,
        )
        identity_sync = {"status": "failed", "error": str(exc)}
    run_id = _run_id(scene)
    run = runs.get(run_id) if run_id else None
    if run is not None:
        runs.update(
            run_id,
            AnalysisRunUpdate(
                diagnostics={
                    **dict(run.diagnostics or {}),
                    "identitySync": identity_sync,
                }
            ),
        )
    return True


def recover_missed_identity_sync(
    *,
    scenes=None,
    runs: AnalysisRunRepository = analysis_runs,
    projects: ProjectStore = project_store,
    resources: ProjectResourceRepository = project_resources,
    matches: ProjectMatchRepository = project_matches,
    identities: ProjectIdentityRepository = project_identities,
    limit: int = 20,
) -> int:
    """Repeat the idempotent identity-sync epilogue for crashed windows.

    A worker that died between the fenced terminal commit and the epilogue
    leaves a succeeded run without an ``identitySync: succeeded`` diagnostic.
    This bounded startup sweep re-runs the epilogue for those runs only while
    the scene still belongs to the same terminal run; a superseded or active
    scene is left alone (lifecycle state wins over stale telemetry).
    """

    if scenes is None:
        from .scene_repository import scenes as scene_repository

        scenes = scene_repository
    repaired = 0
    for run in runs.list_missed_identity_sync(limit=limit):
        scene_id = str(run.scene_id or "")
        if not scene_id:
            continue
        scene = scenes.get(scene_id)
        if scene is None:
            continue
        reconstruction = (
            scene.get("payload", {}).get("videoAsset", {}).get("reconstruction")
            or {}
        )
        if (
            reconstruction.get("status") != "ready"
            or str(reconstruction.get("runId") or "") != str(run.id)
        ):
            continue
        if publish_reconstruction_terminal(
            scene,
            "ready",
            projects=projects,
            resources=resources,
            runs=runs,
            matches=matches,
            identities=identities,
        ):
            repaired += 1
    return repaired
