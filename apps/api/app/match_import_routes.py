from fastapi import APIRouter, HTTPException

from .canonical_match_persistence import persist_canonical_match
from .external_reference_repository import external_references
from .manual_match_import import build_manual_match_bundle
from .project_http_contracts import PublicCanonicalMatch
from .project_http_errors import project_http_error
from .project_http_views import canonical_match_view
from .project_match_repository import (
    ProjectMatchConflict,
    ProjectMatchNotFound,
    project_matches,
)
from .project_store import project_store
from .match_contracts import ManualMatchImportRequest


router = APIRouter(prefix="/api/projects/{project_id}/match", tags=["match"])


@router.post("/import", response_model=PublicCanonicalMatch)
def import_project_match(
    project_id: str,
    request: ManualMatchImportRequest,
) -> PublicCanonicalMatch:
    if not project_store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        bundle, _provenance = build_manual_match_bundle(request)
        canonical = persist_canonical_match(
            project_id,
            bundle,
            matches=project_matches,
            references=external_references,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ProjectMatchConflict, ProjectMatchNotFound) as exc:
        raise project_http_error(exc) from exc
    updated = project_store.get_project(project_id)
    assert updated is not None
    result = canonical_match_view(
        updated,
        canonical,
        project_matches.current_summary(project_id),
        manual=True,
    )
    assert result is not None
    return result
