import { request } from './transport'
import { projectPath } from './paths'
import type {
  AnalysisJob,
  Project,
  ProjectAsset,
  ProjectIdentity,
  ProjectIdentityMembership,
  ProjectSegment,
} from '../../types/project'

export const projectClient = {
  list: (signal?: AbortSignal) => request<Project[]>('/api/projects', { signal }),
  get: (projectId: string, signal?: AbortSignal) => request<Project>(projectPath(projectId), { signal }),
  create: (title: string) => request<Project>('/api/projects', {
    method: 'POST',
    body: JSON.stringify({ title }),
  }),
  assets: (projectId: string, signal?: AbortSignal) => request<ProjectAsset[]>(
    projectPath(projectId, '/assets'),
    { signal },
  ),
  segments: (projectId: string, signal?: AbortSignal) => request<ProjectSegment[]>(
    projectPath(projectId, '/segments'),
    { signal },
  ),
  identities: (projectId: string, signal?: AbortSignal) => request<ProjectIdentity[]>(
    projectPath(projectId, '/identities'),
    { signal },
  ),
  assignIdentityMembership: (
    projectId: string,
    projectPersonId: string,
    sceneId: string,
    scenePersonId: string,
  ) => request<ProjectIdentityMembership>(
    projectPath(projectId, `/identities/${encodeURIComponent(projectPersonId)}/memberships`),
    { method: 'POST', body: JSON.stringify({ sceneId, scenePersonId }) },
  ),
  analysisRuns: (projectId: string, signal?: AbortSignal) => request<AnalysisJob[]>(
    projectPath(projectId, '/analysis-runs'),
    { signal },
  ),
  cancelAnalysisRun: (projectId: string, runId: string, signal?: AbortSignal) => request<AnalysisJob>(
    projectPath(projectId, `/analysis-runs/${encodeURIComponent(runId)}/cancel`),
    { method: 'POST', signal },
  ),
}
