from __future__ import annotations

from datetime import date as date_type

from fastapi import APIRouter, HTTPException, Query

from .canonical_match_persistence import persist_canonical_match
from .external_reference_repository import (
    ExternalReferenceConflict,
    ExternalReferenceRepository,
    external_references,
)
from .project_identifiers import stable_identifier
from .project_http_contracts import (
    MatchCandidate,
    MatchSelection,
    PublicCanonicalMatch,
)
from .project_http_views import canonical_match_view
from .project_http_access import project_or_404
from .project_http_errors import project_http_error
from .project_integration_queries import ProjectIntegrationDiagnosticsQuery
from .project_match_repository import (
    ProjectMatchConflict,
    ProjectMatchNotFound,
    project_matches,
)
from .project_match_persistence_contract import (
    ExternalReferenceCreate,
    IntegrationDiagnostics,
)
from .project_store import (
    ProjectConflict,
    ProjectNotFound,
    project_store,
)
from .providers.base import MatchDataError
from .providers.registry import sports_provider
from .match_contracts import ExternalEvent


router = APIRouter(prefix="/api/projects/{project_id}", tags=["match"])


def remember_match_candidates(
    events: list[ExternalEvent],
    *,
    provider: str,
    references: ExternalReferenceRepository | None = None,
) -> list[MatchCandidate]:
    target = references or external_references
    result: list[MatchCandidate] = []
    for event in events:
        candidate_id = stable_identifier(
            "match-candidate",
            provider,
            event.id,
            event.date,
            event.home.name,
            event.away.name,
            length=32,
        )
        target.upsert(
            ExternalReferenceCreate(
                resource_type="match-candidate",
                resource_id=candidate_id,
                provider=provider,
                external_type="event",
                external_id=str(event.id),
                payload={
                    "name": event.name,
                    "date": event.date,
                    "home": event.home.name,
                    "away": event.away.name,
                },
            )
        )
        result.append(
            MatchCandidate(
                id=candidate_id,
                name=event.name,
                date=event.date,
                time=event.time,
                status=event.status,
                competition=event.league,
                season=event.season,
                home_team={"name": event.home.name, "badge": event.home.badge},
                away_team={"name": event.away.name, "badge": event.away.badge},
                score={"home": event.home_score, "away": event.away_score},
                thumbnail=event.thumbnail,
            )
        )
    return result


@router.get("/match/search", response_model=list[MatchCandidate])
async def search_matches(
    project_id: str,
    q: str | None = Query(default=None, min_length=3, max_length=120),
    date: str | None = Query(default=None, min_length=10, max_length=10),
) -> list[MatchCandidate]:
    if not project_store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    if q is None and date is None:
        date = date_type.today().isoformat()
    try:
        if q is not None:
            provider_id = sports_provider.get().id
            events = await sports_provider.search_events(q)
        else:
            provider_id, events = await sports_provider.events_by_date_with_fallback(
                str(date)
            )
        return remember_match_candidates(events, provider=provider_id)
    except (MatchDataError, ExternalReferenceConflict) as exc:
        raise project_http_error(exc) from exc


@router.get("/match", response_model=PublicCanonicalMatch | None)
def get_project_match(project_id: str) -> PublicCanonicalMatch | None:
    project = project_or_404(project_id, store=project_store)
    snapshot = project_matches.current_snapshot(project_id)
    return canonical_match_view(
        project,
        dict(snapshot.payload) if snapshot is not None else None,
        snapshot,
        manual=bool(snapshot and snapshot.provider == "manual"),
    )


@router.put("/match", response_model=PublicCanonicalMatch)
async def select_project_match(
    project_id: str,
    request: MatchSelection,
) -> PublicCanonicalMatch:
    project_or_404(project_id, store=project_store)
    references = external_references.for_resource(
        "match-candidate",
        request.match_id,
    )
    if not references:
        raise HTTPException(
            status_code=404,
            detail="Match candidate expired or was not found; search again",
        )
    reference = sorted(
        references,
        key=lambda item: (item.provider, item.external_id),
    )[0]
    try:
        bundle = await sports_provider.event_bundle_for(
            reference.provider,
            reference.external_id,
        )
        canonical = persist_canonical_match(
            project_id,
            bundle,
            matches=project_matches,
            references=external_references,
        )
    except (
        MatchDataError,
        ProjectConflict,
        ProjectNotFound,
        ProjectMatchConflict,
        ProjectMatchNotFound,
        ExternalReferenceConflict,
    ) as exc:
        raise project_http_error(exc) from exc
    result = canonical_match_view(
        project_or_404(project_id, store=project_store),
        canonical,
        project_matches.current_summary(project_id),
    )
    assert result is not None
    return result


@router.post("/match/refresh", response_model=PublicCanonicalMatch)
async def refresh_project_match(project_id: str) -> PublicCanonicalMatch:
    project_or_404(project_id, store=project_store)
    source = project_matches.current_source(project_id)
    if source is None or not source.external_event_id:
        raise HTTPException(
            status_code=409,
            detail="The current match has no refreshable integration source",
        )
    try:
        bundle = await sports_provider.event_bundle_for(
            source.provider,
            source.external_event_id,
        )
        canonical = persist_canonical_match(
            project_id,
            bundle,
            matches=project_matches,
            references=external_references,
        )
    except (
        MatchDataError,
        ProjectConflict,
        ProjectNotFound,
        ProjectMatchConflict,
        ProjectMatchNotFound,
        ExternalReferenceConflict,
    ) as exc:
        raise project_http_error(exc) from exc
    result = canonical_match_view(
        project_or_404(project_id, store=project_store),
        canonical,
        project_matches.current_summary(project_id),
    )
    assert result is not None
    return result


@router.get("/integration-diagnostics", response_model=IntegrationDiagnostics)
def get_project_integration_diagnostics(project_id: str) -> IntegrationDiagnostics:
    result = ProjectIntegrationDiagnosticsQuery(
        project_store,
        project_matches,
        external_references,
    ).get(project_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return result
