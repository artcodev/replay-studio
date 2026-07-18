import { describe, expect, it } from 'vitest'
import type { Project } from '../types/project'
import { resolveExplicitWorkspaceProjectId } from './projectSelection'

const ghost: Project = {
  id: 'project-empty',
  title: 'Imported highlight shell',
  revision: 1,
  matchId: null,
  createdAt: '2026-07-18T00:00:00Z',
  updatedAt: '2026-07-18T00:00:00Z',
}

const complete: Project = {
  ...ghost,
  id: 'project-with-match',
  title: 'Spain vs Belgium',
  matchId: 'match-1',
}

describe('workspace project selection', () => {
  it('keeps the projects collection neutral even when a match-backed project exists', () => {
    expect(resolveExplicitWorkspaceProjectId([ghost, complete])).toBeNull()
    expect(resolveExplicitWorkspaceProjectId([ghost, complete], null)).toBeNull()
  })

  it('accepts only an explicitly requested available project', () => {
    expect(resolveExplicitWorkspaceProjectId([ghost, complete], ghost.id)).toBe(ghost.id)
    expect(resolveExplicitWorkspaceProjectId([ghost, complete], complete.id)).toBe(complete.id)
  })

  it('rejects stale and unavailable ids without falling back to another project', () => {
    expect(resolveExplicitWorkspaceProjectId([ghost, complete], 'deleted-project')).toBeNull()
    expect(resolveExplicitWorkspaceProjectId([], complete.id)).toBeNull()
  })
})
