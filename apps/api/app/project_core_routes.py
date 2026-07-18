from fastapi import APIRouter

from .project_http_contracts import PublicProject
from .project_http_views import public_project
from .project_http_access import project_or_404
from .project_http_errors import project_http_error
from .project_lifecycle_contract import ProjectCreate, ProjectUpdate
from .project_store import ProjectConflict, ProjectNotFound, project_store


router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=list[PublicProject])
def list_projects() -> list[PublicProject]:
    return [public_project(project) for project in project_store.list_projects()]


@router.post("", response_model=PublicProject, status_code=201)
def create_project(request: ProjectCreate) -> PublicProject:
    try:
        return public_project(project_store.create_project(request))
    except (ProjectConflict, ProjectNotFound) as exc:
        raise project_http_error(exc) from exc


@router.get("/{project_id}", response_model=PublicProject)
def get_project(project_id: str) -> PublicProject:
    return public_project(project_or_404(project_id, store=project_store))


@router.patch("/{project_id}", response_model=PublicProject)
def update_project(project_id: str, request: ProjectUpdate) -> PublicProject:
    try:
        return public_project(project_store.update_project(project_id, request))
    except (ProjectConflict, ProjectNotFound) as exc:
        raise project_http_error(exc) from exc


@router.post("/{project_id}/archive", response_model=PublicProject)
def archive_project(project_id: str) -> PublicProject:
    try:
        return public_project(project_store.archive_project(project_id))
    except (ProjectConflict, ProjectNotFound) as exc:
        raise project_http_error(exc) from exc
