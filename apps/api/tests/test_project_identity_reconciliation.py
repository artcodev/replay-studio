from app.project_identity_reconciliation import (
    ProjectMembershipState,
    ProjectPersonState,
    plan_project_identity_reconciliation,
)
from app.project_identity_contract import ProjectPersonSyncItem


def test_reconciliation_retargets_roster_membership_and_deletes_local_orphan() -> None:
    plan = plan_project_identity_reconciliation(
        project_id="project-1",
        scene_id="scene-a",
        items=[
            ProjectPersonSyncItem(
                scene_person_id="track-1",
                roster_person_id="roster-8",
                display_name="Eight",
            )
        ],
        canonical_roster_ids=["roster-8"],
        existing_people=[
            ProjectPersonState(
                id="local-person",
                roster_person_id=None,
                display_name="Unknown",
                team_id=None,
                role=None,
                jersey_number=None,
                status="active",
                identity_confidence=None,
            ),
            ProjectPersonState(
                id="roster-person",
                roster_person_id="roster-8",
                display_name="Eight",
                team_id=None,
                role=None,
                jersey_number=None,
                status="active",
                identity_confidence=None,
            ),
        ],
        existing_memberships=[
            ProjectMembershipState(
                id="membership-1",
                project_person_id="local-person",
                scene_id="scene-a",
                scene_person_id="track-1",
                assignment_source="scene-local",
                identity_status=None,
                identity_confidence=None,
                observation_count=1,
            )
        ],
    )

    assert plan.person_ids_to_delete == ("local-person",)
    assert plan.memberships_to_update[0].project_person_id == "roster-person"
    assert plan.memberships_to_update[0].assignment_source == "accepted-roster"
    assert plan.people_created == 0
