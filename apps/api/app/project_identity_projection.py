from __future__ import annotations

"""Pure provider-neutral read projection for persisted project identities."""

from collections import defaultdict
from collections.abc import Iterable

from .project_models import ProjectPersonMembershipRow, ProjectPersonRow
from .project_identity_contract import ProjectPersonDocument, ProjectPersonMembershipDocument


def project_membership_document(
    row: ProjectPersonMembershipRow,
) -> ProjectPersonMembershipDocument:
    return ProjectPersonMembershipDocument(
        id=row.id,
        project_id=row.project_id,
        project_person_id=row.project_person_id,
        scene_id=row.scene_id,
        scene_person_id=row.scene_person_id,
        assignment_source=row.assignment_source,
        identity_status=row.identity_status,
        identity_confidence=row.identity_confidence,
        observation_count=int(row.observation_count or 0),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def project_person_documents(
    people: Iterable[ProjectPersonRow],
    memberships: Iterable[ProjectPersonMembershipRow],
) -> list[ProjectPersonDocument]:
    by_person: dict[str, list[ProjectPersonMembershipRow]] = defaultdict(list)
    for membership in memberships:
        by_person[str(membership.project_person_id)].append(membership)
    for rows in by_person.values():
        rows.sort(key=lambda row: (str(row.scene_id), str(row.scene_person_id)))

    return [
        ProjectPersonDocument(
            id=row.id,
            project_id=row.project_id,
            roster_person_id=row.roster_person_id,
            display_name=row.display_name,
            team_id=row.team_id,
            role=row.role,
            jersey_number=row.jersey_number,
            status=row.status,
            identity_confidence=row.identity_confidence,
            memberships=[
                project_membership_document(membership)
                for membership in by_person.get(str(row.id), [])
            ],
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in people
    ]
