from __future__ import annotations

"""Canonical persistence boundary for project-wide person identities."""

from datetime import UTC, datetime
from typing import Callable, Iterable

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from .database import SessionLocal
from .project_identifiers import stable_identifier
from .project_identity_projection import (
    project_membership_document,
    project_person_documents,
)
from .project_identity_reconciliation import (
    ProjectIdentityReconciliationConflict,
    ProjectMembershipState,
    ProjectPersonState,
    plan_project_identity_reconciliation,
)
from .project_models import (
    ProjectPersonMembershipRow,
    ProjectPersonRow,
    ProjectRow,
)
from .project_resource_repository import (
    ProjectResourceConflict,
    ProjectResourceRepository,
)
from .project_identity_contract import (
    ProjectIdentitySyncReport,
    ProjectPersonDocument,
    ProjectPersonMembershipDocument,
    ProjectPersonSyncItem,
)


class ProjectIdentityNotFound(LookupError):
    pass


class ProjectIdentityConflict(RuntimeError):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ProjectIdentityRepository:
    """Queries and atomic writes for people and scene memberships."""

    def __init__(self, session_factory: Callable | None = None) -> None:
        self._session_factory = session_factory
        self._resources = ProjectResourceRepository(session_factory)

    def _session(self):
        return (self._session_factory or SessionLocal)()

    @staticmethod
    def _begin_atomic_write(session) -> None:
        if session.get_bind().dialect.name == "sqlite":
            session.execute(text("BEGIN IMMEDIATE"))
        else:
            session.begin()

    @staticmethod
    def _require_project(session, project_id: str, *, for_update: bool = False):
        statement = select(ProjectRow).where(ProjectRow.id == project_id)
        if for_update:
            statement = statement.with_for_update()
        project = session.scalar(statement)
        if project is None:
            raise ProjectIdentityNotFound(f"Project {project_id} was not found")
        return project

    def _require_owned_scene(
        self,
        session,
        project_id: str,
        scene_id: str,
        *,
        for_update: bool = False,
    ) -> None:
        try:
            owner = self._resources.scene_owner_in_transaction(
                session,
                scene_id,
                for_update=for_update,
            )
        except ProjectResourceConflict as exc:
            raise ProjectIdentityConflict(str(exc)) from exc
        if owner != project_id:
            raise ProjectIdentityConflict(
                f"Scene {scene_id} does not belong to project {project_id}"
            )

    @staticmethod
    def _person_state(row: ProjectPersonRow) -> ProjectPersonState:
        return ProjectPersonState(
            id=str(row.id),
            roster_person_id=row.roster_person_id,
            display_name=row.display_name,
            team_id=row.team_id,
            role=row.role,
            jersey_number=row.jersey_number,
            status=row.status,
            identity_confidence=row.identity_confidence,
        )

    @staticmethod
    def _membership_state(
        row: ProjectPersonMembershipRow,
    ) -> ProjectMembershipState:
        return ProjectMembershipState(
            id=str(row.id),
            project_person_id=str(row.project_person_id),
            scene_id=str(row.scene_id),
            scene_person_id=str(row.scene_person_id),
            assignment_source=str(row.assignment_source),
            identity_status=row.identity_status,
            identity_confidence=row.identity_confidence,
            observation_count=int(row.observation_count or 0),
        )

    @staticmethod
    def _documents(
        session,
        rows: Iterable[ProjectPersonRow],
    ) -> list[ProjectPersonDocument]:
        people = list(rows)
        person_ids = [str(row.id) for row in people]
        memberships = (
            session.scalars(
                select(ProjectPersonMembershipRow)
                .where(ProjectPersonMembershipRow.project_person_id.in_(person_ids))
                .order_by(
                    ProjectPersonMembershipRow.project_person_id,
                    ProjectPersonMembershipRow.scene_id,
                    ProjectPersonMembershipRow.scene_person_id,
                )
            ).all()
            if person_ids
            else []
        )
        return project_person_documents(people, memberships)

    def list_for_project(self, project_id: str) -> list[ProjectPersonDocument]:
        """Return project identities without integration/provider provenance."""

        with self._session() as session:
            self._require_project(session, project_id)
            rows = session.scalars(
                select(ProjectPersonRow)
                .where(ProjectPersonRow.project_id == project_id)
                .order_by(
                    ProjectPersonRow.team_id,
                    ProjectPersonRow.display_name,
                    ProjectPersonRow.id,
                )
            ).all()
            return self._documents(session, rows)

    def get(
        self,
        project_id: str,
        person_id: str,
    ) -> ProjectPersonDocument | None:
        with self._session() as session:
            row = session.get(ProjectPersonRow, person_id)
            if row is None or row.project_id != project_id:
                return None
            return self._documents(session, [row])[0]

    def assign_membership(
        self,
        project_id: str,
        project_person_id: str,
        scene_id: str,
        scene_person_id: str,
    ) -> ProjectPersonMembershipDocument:
        """Explicitly attach a scene identity to an existing project person.

        A later automatic sync treats this mapping as authoritative. This is
        the persistence primitive needed by a future merge/reassign UI.
        """

        normalized_scene_person_id = str(scene_person_id or "").strip()
        if not normalized_scene_person_id:
            raise ProjectIdentityConflict("Scene person id is required")
        with self._session() as session:
            self._begin_atomic_write(session)
            self._require_project(session, project_id, for_update=True)
            person = session.scalar(
                select(ProjectPersonRow)
                .where(ProjectPersonRow.id == project_person_id)
                .with_for_update()
            )
            if person is None or person.project_id != project_id:
                raise ProjectIdentityConflict(
                    f"Project person {project_person_id} was not found in {project_id}"
                )
            self._require_owned_scene(
                session,
                project_id,
                scene_id,
                for_update=True,
            )
            membership = session.scalar(
                select(ProjectPersonMembershipRow).where(
                    ProjectPersonMembershipRow.project_id == project_id,
                    ProjectPersonMembershipRow.scene_id == scene_id,
                    ProjectPersonMembershipRow.scene_person_id
                    == normalized_scene_person_id,
                ).with_for_update()
            )
            previous_person_id = (
                membership.project_person_id if membership is not None else None
            )
            if membership is None:
                membership = ProjectPersonMembershipRow(
                    id=stable_identifier(
                        "membership",
                        project_id,
                        scene_id,
                        normalized_scene_person_id,
                        length=32,
                    ),
                    project_id=project_id,
                    project_person_id=project_person_id,
                    scene_id=scene_id,
                    scene_person_id=normalized_scene_person_id,
                    assignment_source="explicit",
                )
                session.add(membership)
            else:
                membership.project_person_id = project_person_id
                membership.assignment_source = "explicit"
                membership.updated_at = _utcnow()
            session.flush()
            if previous_person_id and previous_person_id != project_person_id:
                previous = session.get(ProjectPersonRow, previous_person_id)
                remaining = int(
                    session.scalar(
                        select(func.count())
                        .select_from(ProjectPersonMembershipRow)
                        .where(
                            ProjectPersonMembershipRow.project_person_id
                            == previous_person_id
                        )
                    )
                    or 0
                )
                if previous is not None and remaining == 0 and not previous.roster_person_id:
                    session.delete(previous)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise ProjectIdentityConflict(
                    "Scene identity membership changed concurrently"
                ) from exc
            session.refresh(membership)
            return project_membership_document(membership)

    def sync_scene_people(
        self,
        project_id: str,
        scene_id: str,
        people: Iterable[ProjectPersonSyncItem],
        *,
        canonical_roster_ids: Iterable[str] = (),
        unverified_roster_binding_count: int = 0,
    ) -> ProjectIdentitySyncReport:
        """Atomically fold normalized scene identities into project identities.

        The only automatic cross-scene merge key is ``roster_person_id``. An
        existing membership is preserved, especially when it was assigned
        explicitly. Equal scene-local ids from different scenes are never a
        merge signal.
        """
        items = list(people)
        with self._session() as session:
            self._begin_atomic_write(session)
            self._require_project(session, project_id, for_update=True)
            self._require_owned_scene(session, project_id, scene_id)
            person_rows = session.scalars(
                select(ProjectPersonRow)
                .where(ProjectPersonRow.project_id == project_id)
                .with_for_update()
            ).all()
            membership_rows = session.scalars(
                select(ProjectPersonMembershipRow)
                .where(ProjectPersonMembershipRow.project_id == project_id)
                .with_for_update()
            ).all()
            try:
                plan = plan_project_identity_reconciliation(
                    project_id=project_id,
                    scene_id=scene_id,
                    items=items,
                    canonical_roster_ids=canonical_roster_ids,
                    existing_people=(self._person_state(row) for row in person_rows),
                    existing_memberships=(
                        self._membership_state(row) for row in membership_rows
                    ),
                )
            except ProjectIdentityReconciliationConflict as exc:
                session.rollback()
                raise ProjectIdentityConflict(str(exc)) from exc

            people_by_id = {str(row.id): row for row in person_rows}
            memberships_by_id = {str(row.id): row for row in membership_rows}
            for state in plan.people_to_create:
                row = ProjectPersonRow(
                    id=state.id,
                    project_id=project_id,
                    roster_person_id=state.roster_person_id,
                    display_name=state.display_name,
                    team_id=state.team_id,
                    role=state.role,
                    jersey_number=state.jersey_number,
                    status=state.status,
                    identity_confidence=state.identity_confidence,
                )
                session.add(row)
                people_by_id[state.id] = row
            for state in plan.people_to_update:
                row = people_by_id[state.id]
                row.roster_person_id = state.roster_person_id
                row.display_name = state.display_name
                row.team_id = state.team_id
                row.role = state.role
                row.jersey_number = state.jersey_number
                row.status = state.status
                row.identity_confidence = state.identity_confidence
                row.updated_at = _utcnow()
            session.flush()

            for state in plan.memberships_to_create:
                row = ProjectPersonMembershipRow(
                    id=state.id,
                    project_id=project_id,
                    project_person_id=state.project_person_id,
                    scene_id=state.scene_id,
                    scene_person_id=state.scene_person_id,
                    assignment_source=state.assignment_source,
                    identity_status=state.identity_status,
                    identity_confidence=state.identity_confidence,
                    observation_count=state.observation_count,
                )
                session.add(row)
                memberships_by_id[state.id] = row
            for state in plan.memberships_to_update:
                row = memberships_by_id[state.id]
                row.project_person_id = state.project_person_id
                row.assignment_source = state.assignment_source
                row.identity_status = state.identity_status
                row.identity_confidence = state.identity_confidence
                row.observation_count = state.observation_count
                row.updated_at = _utcnow()
            session.flush()
            for person_id in plan.person_ids_to_delete:
                row = people_by_id.get(person_id)
                if row is not None:
                    session.delete(row)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise ProjectIdentityConflict(
                    "Project identity sync conflicted with another writer"
                ) from exc

            rows = session.scalars(
                select(ProjectPersonRow)
                .where(ProjectPersonRow.project_id == project_id)
                .order_by(
                    ProjectPersonRow.team_id,
                    ProjectPersonRow.display_name,
                    ProjectPersonRow.id,
                )
            ).all()
            return ProjectIdentitySyncReport(
                project_id=project_id,
                scene_id=scene_id,
                people_created=plan.people_created,
                people_updated=plan.people_updated,
                memberships_created=plan.memberships_created,
                memberships_updated=plan.memberships_updated,
                memberships_preserved=plan.memberships_preserved,
                unverified_roster_binding_count=max(
                    0, int(unverified_roster_binding_count)
                ),
                people=self._documents(session, rows),
            )



project_identities = ProjectIdentityRepository()
