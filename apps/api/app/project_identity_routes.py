from fastapi import APIRouter, HTTPException

from .project_identity_repository import (
    ProjectIdentityConflict,
    ProjectIdentityNotFound,
    project_identities,
)
from .project_http_contracts import ProjectPersonMembershipAssignment
from .project_identity_contract import (
    ProjectPersonDocument,
    ProjectPersonMembershipDocument,
)


router = APIRouter(prefix="/api/projects/{project_id}/identities", tags=["identities"])


@router.get("", response_model=list[ProjectPersonDocument])
def list_project_identities(project_id: str) -> list[ProjectPersonDocument]:
    try:
        return project_identities.list_for_project(project_id)
    except ProjectIdentityNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/{project_person_id}/memberships",
    response_model=ProjectPersonMembershipDocument,
)
def assign_project_identity_membership(
    project_id: str,
    project_person_id: str,
    request: ProjectPersonMembershipAssignment,
) -> ProjectPersonMembershipDocument:
    try:
        return project_identities.assign_membership(
            project_id,
            project_person_id,
            request.scene_id,
            request.scene_person_id,
        )
    except ProjectIdentityNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProjectIdentityConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
