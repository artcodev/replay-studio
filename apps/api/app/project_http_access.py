from fastapi import HTTPException

from .project_lifecycle_contract import ProjectHeader
from .project_store import ProjectStore, project_store


def project_or_404(
    project_id: str,
    *,
    store: ProjectStore | None = None,
) -> ProjectHeader:
    project = (store or project_store).get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project
