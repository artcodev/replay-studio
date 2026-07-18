import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Router } from 'vue-router'
import { useProjectWorkspace } from './useProjectWorkspace'

const mocks = vi.hoisted(() => ({
  listProjects: vi.fn(),
  getProject: vi.fn(),
  createProject: vi.fn(),
  listAssets: vi.fn(),
  listSegments: vi.fn(),
  listIdentities: vi.fn(),
  assignIdentityMembership: vi.fn(),
  getMatch: vi.fn(),
  searchMatches: vi.fn(),
  selectMatch: vi.fn(),
  refreshMatch: vi.fn(),
  startJobs: vi.fn(),
  stopJobs: vi.fn(),
  refreshJobs: vi.fn(),
}))

vi.mock('../lib/api/projects', () => ({
  projectClient: {
    list: mocks.listProjects,
    get: mocks.getProject,
    create: mocks.createProject,
    assets: mocks.listAssets,
    segments: mocks.listSegments,
    identities: mocks.listIdentities,
    assignIdentityMembership: mocks.assignIdentityMembership,
  },
}))

vi.mock('../lib/api/matches', () => ({
  matchClient: {
    get: mocks.getMatch,
    search: mocks.searchMatches,
    select: mocks.selectMatch,
    refresh: mocks.refreshMatch,
  },
}))

vi.mock('../lib/analysisJobs', () => ({
  useAnalysisJobs: () => ({
    jobs: { value: [] },
    loading: { value: false },
    error: { value: null },
    cancelingJobIds: { value: [] },
    lastUpdatedAt: { value: null },
    start: mocks.startJobs,
    stop: mocks.stopJobs,
    refresh: mocks.refreshJobs,
    cancel: vi.fn(),
  }),
}))

function project(id = 'project-1') {
  return { id, title: 'Replay', matchId: null, activeSegmentId: null, updatedAt: '2026-07-18T00:00:00Z' }
}

describe('useProjectWorkspace', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listProjects.mockResolvedValue([project()])
    mocks.getProject.mockResolvedValue(project())
    mocks.getMatch.mockResolvedValue(null)
    mocks.listAssets.mockResolvedValue([{ id: 'asset-1', projectId: 'project-1' }])
    mocks.listSegments.mockResolvedValue([{ id: 'segment-1', projectId: 'project-1' }])
    mocks.listIdentities.mockResolvedValue([{ id: 'identity-1', memberships: [] }])
    mocks.startJobs.mockResolvedValue(undefined)
  })

  it('hydrates independently owned resources for the selected project', async () => {
    const router = { push: vi.fn() } as unknown as Router
    const workspace = useProjectWorkspace(router)

    await expect(workspace.load('project-1')).resolves.toBe('loaded')

    expect(workspace.catalog.project.value?.id).toBe('project-1')
    expect(workspace.match.snapshot.value).toBeNull()
    expect(workspace.media.assets.value).toHaveLength(1)
    expect(workspace.media.segments.value).toHaveLength(1)
    expect(workspace.identities.rows.value).toHaveLength(1)
    expect(mocks.startJobs).toHaveBeenCalledWith('project-1')
  })

  it('clears resource state when an explicit project does not exist', async () => {
    const workspace = useProjectWorkspace({ push: vi.fn() } as unknown as Router)
    await workspace.load('project-1')

    await expect(workspace.load('missing')).resolves.toBe('not-found')

    expect(workspace.catalog.project.value).toBeNull()
    expect(workspace.media.assets.value).toEqual([])
    expect(workspace.media.segments.value).toEqual([])
    expect(workspace.identities.rows.value).toEqual([])
    expect(workspace.match.snapshot.value).toBeNull()
    expect(mocks.stopJobs).toHaveBeenCalled()
  })
})
