from fastapi import HTTPException

from .analysis_run_repository import AnalysisRunConflict, AnalysisRunNotFound
from .external_reference_repository import ExternalReferenceConflict
from .project_match_repository import ProjectMatchConflict, ProjectMatchNotFound
from .project_resource_repository import (
    ProjectResourceConflict,
    ProjectResourceNotFound,
)
from .project_store import ProjectConflict, ProjectNotFound
from .providers.base import MatchDataError


def project_http_error(error: Exception) -> HTTPException:
    if isinstance(
        error,
        (
            ProjectNotFound,
            ProjectMatchNotFound,
            ProjectResourceNotFound,
            AnalysisRunNotFound,
        ),
    ):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(
        error,
        (
            ProjectConflict,
            ProjectMatchConflict,
            ProjectResourceConflict,
            ExternalReferenceConflict,
            AnalysisRunConflict,
        ),
    ):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, MatchDataError):
        status = 404 if error.code == "event-not-found" else 503
        if error.code in {"unknown-provider", "team-pair-required"}:
            status = 422
        return HTTPException(status_code=status, detail=str(error))
    return HTTPException(
        status_code=500,
        detail="Unexpected project operation failure",
    )
