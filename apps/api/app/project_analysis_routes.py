from fastapi import APIRouter, HTTPException

from .analysis_cancellation import analysis_cancellation
from .analysis_run_repository import (
    AnalysisRunConflict,
    AnalysisRunNotFound,
    analysis_runs,
)
from .pipeline_terminal_service import pipeline_terminals
from .project_http_contracts import PublicAnalysisJob
from .project_http_views import public_analysis_job
from .project_http_errors import project_http_error


router = APIRouter(
    prefix="/api/projects/{project_id}/analysis-runs",
    tags=["analysis"],
)


@router.get("", response_model=list[PublicAnalysisJob])
def list_project_analysis_runs(project_id: str) -> list[PublicAnalysisJob]:
    try:
        runs = analysis_runs.list_for_project(project_id)
    except AnalysisRunNotFound as exc:
        raise project_http_error(exc) from exc
    return [public_analysis_job(run) for run in runs]


@router.post("/{run_id}/cancel", response_model=PublicAnalysisJob)
def cancel_analysis_run(project_id: str, run_id: str) -> PublicAnalysisJob:
    run = analysis_runs.get(run_id)
    if run is None or run.project_id != project_id:
        raise HTTPException(
            status_code=404,
            detail="Analysis run not found in project",
        )
    pipeline_job = pipeline_terminals.cancel(run_id)
    if pipeline_job is not None:
        updated = analysis_runs.get(run_id)
        if updated is None or updated.project_id != project_id:
            raise HTTPException(
                status_code=404,
                detail="Analysis run not found in project",
            )
        return public_analysis_job(updated)
    try:
        return public_analysis_job(analysis_cancellation.cancel(run_id))
    except (AnalysisRunConflict, AnalysisRunNotFound) as exc:
        raise project_http_error(exc) from exc
