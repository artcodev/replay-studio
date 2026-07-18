import { createMemoryHistory } from 'vue-router'
import { describe, expect, it } from 'vitest'
import {
  appRouteLocation,
  projectSegmentRoute,
  projectWorkspaceRoute,
  videoTimelineRoute,
} from './lib/appRoutes'
import { createAppRouter } from './router'

describe('Replay Studio router', () => {
  it('resolves canonical workspace and editor locations', () => {
    const router = createAppRouter(createMemoryHistory())

    expect(router.resolve(appRouteLocation(projectWorkspaceRoute('project-1', 'match'))).href)
      .toBe('/projects/project-1/match')
    expect(router.resolve(appRouteLocation(videoTimelineRoute('project-1', 'asset-1'))).href)
      .toBe('/projects/project-1/videos/asset-1/timeline')
    expect(router.resolve(appRouteLocation(projectSegmentRoute('project-1', 'segment-2'))).href)
      .toBe('/projects/project-1/segments/segment-2')
  })

  it('normalizes project roots and unknown paths', async () => {
    const router = createAppRouter(createMemoryHistory())

    await router.push('/projects/project-1')
    expect(router.currentRoute.value.fullPath).toBe('/projects/project-1/overview')

    await router.push('/legacy-editor/shot-2')
    expect(router.currentRoute.value.fullPath).toBe('/projects')
  })

  it('keeps a direct Match workspace URL active during initial navigation', async () => {
    const router = createAppRouter(createMemoryHistory())

    await router.push('/projects/project-1/match')

    expect(router.currentRoute.value).toMatchObject({
      name: 'project-workspace',
      fullPath: '/projects/project-1/match',
      params: { projectId: 'project-1', tab: 'match' },
    })
  })

  it('keeps the projects collection as its own neutral route', async () => {
    const router = createAppRouter(createMemoryHistory())

    await router.push('/projects')

    expect(router.currentRoute.value).toMatchObject({
      name: 'projects',
      fullPath: '/projects',
      params: {},
    })
  })
})
