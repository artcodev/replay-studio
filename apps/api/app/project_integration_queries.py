from __future__ import annotations

"""Administrative integration views composed from canonical repositories."""

from .external_reference_repository import (
    ExternalReferenceRepository,
    external_references,
)
from .project_match_repository import ProjectMatchRepository, project_matches
from .project_match_persistence_contract import IntegrationDiagnostics
from .project_store import ProjectStore, project_store


class ProjectIntegrationDiagnosticsQuery:
    def __init__(
        self,
        projects: ProjectStore = project_store,
        matches: ProjectMatchRepository = project_matches,
        references: ExternalReferenceRepository = external_references,
    ) -> None:
        self._projects = projects
        self._matches = matches
        self._references = references

    def get(self, project_id: str) -> IntegrationDiagnostics | None:
        project = self._projects.get_project(project_id)
        if project is None:
            return None
        resource_ids = [project_id]
        if project.match_id:
            resource_ids.append(project.match_id)
        return IntegrationDiagnostics(
            project_id=project_id,
            current_match_snapshot=self._matches.current_snapshot(project_id),
            external_references=self._references.for_resource_ids(resource_ids),
        )


project_integration_diagnostics = ProjectIntegrationDiagnosticsQuery()
