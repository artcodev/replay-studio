import { afterEach, describe, expect, it, vi } from 'vitest'
import type { CanonicalMatch, MatchCandidate } from '../types/match'
import type { AnalysisJob, Project } from '../types/project'
import { projectClient } from './api/projects'
import { matchClient } from './api/matches'
import { sceneClient } from './api/scenes'
import { mediaClient } from './api/media'

afterEach(() => {
  vi.unstubAllGlobals()
})

function ok(body: unknown) {
  return { ok: true, json: async () => body }
}

describe('project API', () => {
  it('uses encoded canonical project resources without a provider parameter', async () => {
    const project: Project = {
      id: 'project/one',
      title: 'Project One',
      revision: 1,
      matchId: null,
      activeSegmentId: null,
      createdAt: '2026-07-17T08:00:00Z',
      updatedAt: '2026-07-17T08:00:00Z',
    }
    const match: CanonicalMatch = {
      id: 'match-1',
      revision: 1,
      snapshotId: 'snapshot-1',
      snapshotHash: 'sha256:one',
      score: { home: null, away: null },
      homeTeam: { id: 'home', name: 'Home' },
      awayTeam: { id: 'away', name: 'Away' },
      roster: [],
      events: [],
      substitutions: [],
      sync: { state: 'not-configured', syncedAt: null, stale: false, warnings: [] },
    }
    const fetch = vi.fn()
      .mockResolvedValueOnce(ok([project]))
      .mockResolvedValueOnce(ok(project))
      .mockResolvedValueOnce(ok(match))
      .mockResolvedValueOnce(ok([]))
      .mockResolvedValueOnce(ok([]))
      .mockResolvedValueOnce(ok([]))
      .mockResolvedValueOnce(ok([]))
      .mockResolvedValueOnce(ok([]))
    vi.stubGlobal('fetch', fetch)
    const abort = new AbortController()

    await projectClient.list(abort.signal)
    await projectClient.get(project.id, abort.signal)
    await matchClient.get(project.id, abort.signal)
    await projectClient.assets(project.id, abort.signal)
    await projectClient.segments(project.id, abort.signal)
    await projectClient.identities(project.id, abort.signal)
    await sceneClient.list(project.id, abort.signal)
    await projectClient.analysisRuns(project.id, abort.signal)

    expect(fetch.mock.calls.map((call) => call[0])).toEqual([
      '/api/projects',
      '/api/projects/project%2Fone',
      '/api/projects/project%2Fone/match',
      '/api/projects/project%2Fone/assets',
      '/api/projects/project%2Fone/segments',
      '/api/projects/project%2Fone/identities',
      '/api/projects/project%2Fone/scenes',
      '/api/projects/project%2Fone/analysis-runs',
    ])
    expect(fetch.mock.calls.every((call) => (call[1] as RequestInit).signal === abort.signal)).toBe(true)
    expect(fetch.mock.calls.every((call) => !String(call[0]).includes('provider'))).toBe(true)
  })

  it('assigns a scene identity through the provider-neutral project graph', async () => {
    const fetch = vi.fn().mockResolvedValueOnce(ok({ id: 'membership-1' }))
    vi.stubGlobal('fetch', fetch)

    await projectClient.assignIdentityMembership(
      'project/one',
      'person/eight',
      'scene/1-A',
      'canonical/player-8',
    )

    expect(fetch.mock.calls[0][0]).toBe(
      '/api/projects/project%2Fone/identities/person%2Feight/memberships',
    )
    expect(fetch.mock.calls[0][1]).toMatchObject({
      method: 'POST',
      body: JSON.stringify({
        sceneId: 'scene/1-A',
        scenePersonId: 'canonical/player-8',
      }),
    })
    expect(String(fetch.mock.calls[0][0])).not.toContain('provider')
  })

  it('searches and selects matches through provider-neutral project routes', async () => {
    const candidate = {
      id: 'candidate/one',
      name: 'Spain vs Belgium',
    } as MatchCandidate
    const match = { id: 'match-1' } as CanonicalMatch
    const fetch = vi.fn()
      .mockResolvedValueOnce(ok([candidate]))
      .mockResolvedValueOnce(ok(match))
      .mockResolvedValueOnce(ok(match))
    vi.stubGlobal('fetch', fetch)

    await matchClient.search('project-1', { query: ' Spain vs Belgium ' })
    await matchClient.select('project/one', candidate.id)
    await matchClient.refresh('project/one')

    expect(fetch.mock.calls[0][0]).toBe('/api/projects/project-1/match/search?q=Spain+vs+Belgium')
    expect(fetch.mock.calls[1][0]).toBe('/api/projects/project%2Fone/match')
    expect(fetch.mock.calls[1][1]).toMatchObject({
      method: 'PUT',
      body: JSON.stringify({ matchId: candidate.id }),
    })
    expect(fetch.mock.calls[2][0]).toBe('/api/projects/project%2Fone/match/refresh')
    expect(fetch.mock.calls[2][1]).toMatchObject({ method: 'POST' })
    expect(fetch.mock.calls.flatMap((call) => String(call[0]))).not.toContain('provider=')
  })

  it('creates a project and requests cancellation through explicit mutation routes', async () => {
    const project = { id: 'project-1', title: 'Match lab' }
    const job = {
      id: 'run/one',
      projectId: 'project-1',
      status: 'cancelled',
    } as AnalysisJob
    const fetch = vi.fn()
      .mockResolvedValueOnce(ok(project))
      .mockResolvedValueOnce(ok(job))
    vi.stubGlobal('fetch', fetch)

    await projectClient.create('Match lab')
    await projectClient.cancelAnalysisRun('project-1', job.id)

    expect(fetch.mock.calls[0][0]).toBe('/api/projects')
    expect(fetch.mock.calls[0][1]).toMatchObject({
      method: 'POST',
      body: JSON.stringify({ title: 'Match lab' }),
    })
    expect(fetch.mock.calls[1][0]).toBe('/api/projects/project-1/analysis-runs/run%2Fone/cancel')
    expect(fetch.mock.calls[1][1]).toMatchObject({ method: 'POST' })
  })

  it('binds an uploaded video to the selected project', async () => {
    let sent: FormData | null = null
    let openedUrl = ''
    class UploadRequest {
      status = 201
      response = { id: 'asset-1', status: 'queued' }
      responseType = ''
      upload = { addEventListener: vi.fn() }
      listeners = new Map<string, EventListener>()

      open(_method: string, url: string) { openedUrl = url }

      addEventListener(type: string, listener: EventListener) {
        this.listeners.set(type, listener)
      }

      send(body: Document | XMLHttpRequestBodyInit | null) {
        sent = body as FormData
        this.listeners.get('load')?.(new Event('load'))
      }
    }
    vi.stubGlobal('XMLHttpRequest', UploadRequest)

    await mediaClient.upload(
      'project-1',
      new File(['video'], 'highlight.mp4', { type: 'video/mp4' }),
      'Opening goal',
      vi.fn(),
    )

    const form = sent as FormData | null
    expect(openedUrl).toBe('/api/projects/project-1/videos')
    expect(form?.has('project_id')).toBe(false)
    expect(form?.get('title')).toBe('Opening goal')
    expect((form?.get('file') as File).name).toBe('highlight.mp4')
  })
})
