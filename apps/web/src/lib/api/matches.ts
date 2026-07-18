import { request } from './transport'
import { projectPath } from './paths'
import type { CanonicalMatch, ManualMatchImportRequest, MatchCandidate } from '../../types/match'

export const matchClient = {
  get: (projectId: string, signal?: AbortSignal) => request<CanonicalMatch | null>(
    projectPath(projectId, '/match'),
    { signal },
  ),
  search: (projectId: string, { query, date, signal }: {
    query?: string
    date?: string
    signal?: AbortSignal
  }) => {
    const parameters = new URLSearchParams()
    if (query?.trim()) parameters.set('q', query.trim())
    else if (date) parameters.set('date', date)
    const suffix = parameters.size ? `?${parameters.toString()}` : ''
    return request<MatchCandidate[]>(projectPath(projectId, `/match/search${suffix}`), { signal })
  },
  select: (projectId: string, matchId: string) => request<CanonicalMatch>(
    projectPath(projectId, '/match'),
    { method: 'PUT', body: JSON.stringify({ matchId }) },
  ),
  refresh: (projectId: string) => request<CanonicalMatch>(
    projectPath(projectId, '/match/refresh'),
    { method: 'POST' },
  ),
  import: (projectId: string, payload: ManualMatchImportRequest) => request<CanonicalMatch>(
    projectPath(projectId, '/match/import'),
    { method: 'POST', body: JSON.stringify(payload) },
  ),
}
