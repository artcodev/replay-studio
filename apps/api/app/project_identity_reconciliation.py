from __future__ import annotations

"""Pure reconciliation policy for project people and scene memberships."""

from dataclasses import dataclass, replace
from typing import Iterable

from .project_identifiers import stable_identifier
from .project_identity_contract import ProjectPersonSyncItem


class ProjectIdentityReconciliationConflict(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProjectPersonState:
    id: str
    roster_person_id: str | None
    display_name: str
    team_id: str | None
    role: str | None
    jersey_number: str | None
    status: str
    identity_confidence: float | None


@dataclass(frozen=True, slots=True)
class ProjectMembershipState:
    id: str
    project_person_id: str
    scene_id: str
    scene_person_id: str
    assignment_source: str
    identity_status: str | None
    identity_confidence: float | None
    observation_count: int


@dataclass(frozen=True, slots=True)
class ProjectIdentityReconciliationPlan:
    people_to_create: tuple[ProjectPersonState, ...]
    people_to_update: tuple[ProjectPersonState, ...]
    memberships_to_create: tuple[ProjectMembershipState, ...]
    memberships_to_update: tuple[ProjectMembershipState, ...]
    person_ids_to_delete: tuple[str, ...]
    memberships_preserved: int

    @property
    def people_created(self) -> int:
        return len(self.people_to_create)

    @property
    def people_updated(self) -> int:
        return len(self.people_to_update)

    @property
    def memberships_created(self) -> int:
        return len(self.memberships_to_create)

    @property
    def memberships_updated(self) -> int:
        return len(self.memberships_to_update)


def plan_project_identity_reconciliation(
    *,
    project_id: str,
    scene_id: str,
    items: Iterable[ProjectPersonSyncItem],
    canonical_roster_ids: Iterable[str],
    existing_people: Iterable[ProjectPersonState],
    existing_memberships: Iterable[ProjectMembershipState],
) -> ProjectIdentityReconciliationPlan:
    """Return deterministic mutations without importing persistence."""

    normalized_items = list(items)
    scene_person_ids = [item.scene_person_id for item in normalized_items]
    if len(scene_person_ids) != len(set(scene_person_ids)):
        raise ProjectIdentityReconciliationConflict(
            f"Scene {scene_id} contains duplicate canonical person ids"
        )
    accepted_roster_ids = {str(value) for value in canonical_roster_ids if value}
    if any(
        item.roster_person_id
        and str(item.roster_person_id) not in accepted_roster_ids
        for item in normalized_items
    ):
        raise ProjectIdentityReconciliationConflict(
            "Roster person ids must come from the project's canonical match snapshot"
        )

    people = {person.id: person for person in existing_people}
    memberships = {
        (membership.scene_id, membership.scene_person_id): membership
        for membership in existing_memberships
    }
    roster_people = {
        person.roster_person_id: person.id
        for person in people.values()
        if person.roster_person_id
    }
    original_people = dict(people)
    original_memberships = dict(memberships)
    created_person_ids: set[str] = set()
    touched_person_ids: set[str] = set()
    orphan_candidates: set[str] = set()
    memberships_preserved = 0

    def create_person(item: ProjectPersonSyncItem, person_id: str) -> ProjectPersonState:
        person = ProjectPersonState(
            id=person_id,
            roster_person_id=item.roster_person_id,
            display_name=item.display_name,
            team_id=item.team_id,
            role=item.role,
            jersey_number=item.jersey_number,
            status=item.status,
            identity_confidence=item.identity_confidence,
        )
        people[person_id] = person
        created_person_ids.add(person_id)
        touched_person_ids.add(person_id)
        if person.roster_person_id:
            roster_people[person.roster_person_id] = person_id
        return person

    def update_person(
        person: ProjectPersonState,
        item: ProjectPersonSyncItem,
    ) -> ProjectPersonState:
        authoritative = bool(item.roster_person_id) or not person.roster_person_id
        values = {
            "display_name": item.display_name,
            "team_id": item.team_id,
            "role": item.role,
            "jersey_number": item.jersey_number,
        }
        changes = {
            field: value
            for field, value in values.items()
            if authoritative and value is not None and getattr(person, field) != value
        }
        if item.identity_confidence is not None:
            confidence = max(
                float(person.identity_confidence or 0.0),
                float(item.identity_confidence),
            )
            if person.identity_confidence != confidence:
                changes["identity_confidence"] = confidence
        if changes:
            person = replace(person, **changes)
            people[person.id] = person
        return person

    for item in normalized_items:
        membership_key = (scene_id, item.scene_person_id)
        membership = memberships.get(membership_key)
        existing_person = (
            people.get(membership.project_person_id) if membership is not None else None
        )
        roster_person_id = (
            roster_people.get(item.roster_person_id) if item.roster_person_id else None
        )
        roster_person = people.get(roster_person_id) if roster_person_id else None

        if membership is not None and membership.assignment_source == "explicit":
            if existing_person is None:
                raise ProjectIdentityReconciliationConflict(
                    f"Explicit membership {membership.id} has no project person"
                )
            target = existing_person
            if item.roster_person_id:
                if target.roster_person_id not in {None, item.roster_person_id}:
                    raise ProjectIdentityReconciliationConflict(
                        f"Explicit project person {target.id} is already bound to "
                        "another canonical roster person"
                    )
                if roster_person is not None and roster_person.id != target.id:
                    raise ProjectIdentityReconciliationConflict(
                        f"Accepted roster person {item.roster_person_id} conflicts "
                        f"with explicit membership {membership.id}"
                    )
                if target.roster_person_id is None:
                    target = replace(target, roster_person_id=item.roster_person_id)
                    people[target.id] = target
                    roster_people[item.roster_person_id] = target.id
            memberships_preserved += 1
        elif item.roster_person_id:
            if roster_person is not None:
                target = roster_person
            elif existing_person is not None:
                if existing_person.roster_person_id not in {None, item.roster_person_id}:
                    raise ProjectIdentityReconciliationConflict(
                        f"Scene identity {item.scene_person_id} is already bound to "
                        "another canonical roster person"
                    )
                target = replace(
                    existing_person,
                    roster_person_id=item.roster_person_id,
                )
                people[target.id] = target
                roster_people[item.roster_person_id] = target.id
            else:
                target = create_person(
                    item,
                    stable_identifier(
                        "person",
                        project_id,
                        "roster",
                        item.roster_person_id,
                        length=32,
                    ),
                )
        elif existing_person is not None:
            target = existing_person
            memberships_preserved += 1
        else:
            target = create_person(
                item,
                stable_identifier(
                    "person",
                    project_id,
                    scene_id,
                    item.scene_person_id,
                    length=32,
                ),
            )

        touched_person_ids.add(target.id)
        target = update_person(target, item)
        desired_source = (
            "explicit"
            if membership is not None and membership.assignment_source == "explicit"
            else "accepted-roster"
            if item.roster_person_id
            else "scene-local"
        )
        desired_membership = ProjectMembershipState(
            id=(
                membership.id
                if membership is not None
                else stable_identifier(
                    "membership",
                    project_id,
                    scene_id,
                    item.scene_person_id,
                    length=32,
                )
            ),
            project_person_id=target.id,
            scene_id=scene_id,
            scene_person_id=item.scene_person_id,
            assignment_source=desired_source,
            identity_status=item.identity_status,
            identity_confidence=item.identity_confidence,
            observation_count=item.observation_count,
        )
        if membership is not None and membership.project_person_id != target.id:
            orphan_candidates.add(membership.project_person_id)
        memberships[membership_key] = desired_membership

    member_person_ids = {membership.project_person_id for membership in memberships.values()}
    deleted_person_ids = {
        person_id
        for person_id in orphan_candidates
        if person_id not in member_person_ids
        and person_id in people
        and not people[person_id].roster_person_id
    }
    for person_id in deleted_person_ids:
        people.pop(person_id, None)
        touched_person_ids.discard(person_id)

    for person_id in touched_person_ids:
        person = people.get(person_id)
        if person is None:
            continue
        statuses = [
            membership.identity_status
            for membership in memberships.values()
            if membership.project_person_id == person_id
        ]
        desired_status = (
            "excluded"
            if statuses and all(status == "excluded" for status in statuses)
            else "active"
        )
        if person.status != desired_status:
            people[person_id] = replace(person, status=desired_status)

    people_to_create = tuple(
        people[person_id]
        for person_id in sorted(created_person_ids - deleted_person_ids)
    )
    people_to_update = tuple(
        people[person_id]
        for person_id in sorted(touched_person_ids - created_person_ids)
        if people.get(person_id) != original_people.get(person_id)
    )
    memberships_to_create = tuple(
        membership
        for key, membership in sorted(memberships.items())
        if key not in original_memberships
    )
    memberships_to_update = tuple(
        membership
        for key, membership in sorted(memberships.items())
        if key in original_memberships and membership != original_memberships[key]
    )
    return ProjectIdentityReconciliationPlan(
        people_to_create=people_to_create,
        people_to_update=people_to_update,
        memberships_to_create=memberships_to_create,
        memberships_to_update=memberships_to_update,
        person_ids_to_delete=tuple(sorted(deleted_person_ids)),
        memberships_preserved=memberships_preserved,
    )
