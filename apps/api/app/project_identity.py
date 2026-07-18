from __future__ import annotations

from typing import Any

from .project_identity_repository import (
    ProjectIdentityConflict,
    ProjectIdentityRepository,
    project_identities,
)
from .project_match_repository import ProjectMatchRepository, project_matches
from .project_resource_repository import (
    ProjectResourceRepository,
    project_resources,
)
from .project_identity_contract import ProjectIdentitySyncReport, ProjectPersonSyncItem
from .project_store import ProjectStore, project_store


def _text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _canonical_roster(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("roster") or payload.get("players") or []
    return {
        str(row["id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("id")
    }


def sync_project_identities_from_scene(
    scene: dict[str, Any],
    *,
    project_id: str,
    projects: ProjectStore = project_store,
    resources: ProjectResourceRepository = project_resources,
    matches: ProjectMatchRepository = project_matches,
    identities: ProjectIdentityRepository = project_identities,
) -> ProjectIdentitySyncReport:
    """Persist a reconstructed scene's identities at project scope.

    Reconstruction's internal ``externalPlayerId`` roster-binding field is
    accepted only when its value is present in the project's canonical match
    snapshot. This validation turns it into the provider-neutral
    ``rosterPersonId`` persistence field and prevents upstream ids from leaking
    into public project DTOs.
    """

    scene_id = _text(scene.get("id"))
    if not scene_id:
        raise ProjectIdentityConflict("Scene id is required for identity sync")
    normalized_project_id = _text(project_id)
    if not normalized_project_id:
        raise ProjectIdentityConflict("Project id is required for identity sync")
    if not projects.project_exists(normalized_project_id):
        raise ProjectIdentityConflict(f"Project {normalized_project_id} was not found")
    if resources.scene_owner(scene_id) != normalized_project_id:
        raise ProjectIdentityConflict(
            f"Scene {scene_id} is not owned by project {normalized_project_id}"
        )

    roster = _canonical_roster(
        matches.current_payload(normalized_project_id)
    )
    people = scene.get("payload", {}).get("canonicalPeople") or []
    normalized: list[ProjectPersonSyncItem] = []
    unverified_roster_binding_count = 0
    for person in people:
        if not isinstance(person, dict):
            continue
        scene_person_id = _text(
            person.get("canonicalPersonId") or person.get("id")
        )
        if not scene_person_id:
            continue
        claimed_roster_id = _text(
            person.get("rosterPersonId") or person.get("externalPlayerId")
        )
        roster_person = roster.get(claimed_roster_id or "")
        if claimed_roster_id and roster_person is None:
            unverified_roster_binding_count += 1
        roster_person_id = claimed_roster_id if roster_person is not None else None
        identity_status = _text(person.get("identityStatus"))
        confidence_value = person.get("identityConfidence")
        try:
            identity_confidence = (
                max(0.0, min(1.0, float(confidence_value)))
                if confidence_value is not None
                else None
            )
        except (TypeError, ValueError):
            identity_confidence = None
        observations = person.get("observations") or []
        try:
            observation_count = max(
                0,
                int(person.get("observationCount", len(observations)) or 0),
            )
        except (TypeError, ValueError):
            observation_count = len(observations) if isinstance(observations, list) else 0

        normalized.append(
            ProjectPersonSyncItem(
                scene_person_id=scene_person_id,
                roster_person_id=roster_person_id,
                display_name=(
                    _text((roster_person or {}).get("name"))
                    or _text(person.get("displayName") or person.get("label"))
                    or scene_person_id
                ),
                team_id=(
                    _text((roster_person or {}).get("teamId"))
                    or _text(person.get("teamId"))
                ),
                role=(
                    _text((roster_person or {}).get("position"))
                    or _text(person.get("role"))
                ),
                jersey_number=(
                    _text((roster_person or {}).get("number"))
                    or _text(person.get("jerseyNumber"))
                ),
                status="excluded" if identity_status == "excluded" else "active",
                identity_status=identity_status,
                identity_confidence=identity_confidence,
                observation_count=observation_count,
            )
        )

    return identities.sync_scene_people(
        normalized_project_id,
        scene_id,
        normalized,
        canonical_roster_ids=roster,
        unverified_roster_binding_count=unverified_roster_binding_count,
    )
