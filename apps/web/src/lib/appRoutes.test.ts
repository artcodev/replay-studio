import { describe, expect, it } from 'vitest'
import {
  APP_ROUTE_NAMES,
  APP_ROUTE_PATHS,
  appRouteLocation,
  appRoutePath,
  editorRouteQuery,
  parseAppRoute,
  parseEditorRouteView,
  projectSceneRoute,
  projectSegmentRoute,
  projectsRoute,
  projectWorkspaceRoute,
  videoTimelineRoute,
} from './appRoutes'

describe('Replay Studio route contract', () => {
  it('keeps every normalized project resource on a distinct canonical path', () => {
    expect(APP_ROUTE_PATHS).toEqual({
      root: '/',
      projects: '/projects',
      projectWorkspace: '/projects/:projectId/:tab(overview|match|identities|analysis)',
      videoTimeline: '/projects/:projectId/videos/:assetId/timeline',
      projectSegment: '/projects/:projectId/segments/:segmentId',
      projectScene: '/projects/:projectId/scenes/:sceneId',
    })

    expect(appRoutePath(projectsRoute())).toBe('/projects')
    expect(appRoutePath(projectWorkspaceRoute('project-1'))).toBe('/projects/project-1/overview')
    expect(appRoutePath(projectWorkspaceRoute('project-1', 'match'))).toBe('/projects/project-1/match')
    expect(appRoutePath(videoTimelineRoute('project-1', 'video-1'))).toBe('/projects/project-1/videos/video-1/timeline')
    expect(appRoutePath(projectSegmentRoute('project-1', 'segment-1'))).toBe('/projects/project-1/segments/segment-1')
    expect(appRoutePath(projectSceneRoute('project-1', 'scene-1'))).toBe('/projects/project-1/scenes/scene-1')
  })

  it('uses an asset timeline route for Open editor and an explicit scene route for internal scenes', () => {
    const timeline = videoTimelineRoute('project-1', 'asset-1')
    const shot = projectSceneRoute('project-1', 'shot-2')

    expect(timeline.name).toBe(APP_ROUTE_NAMES.videoTimeline)
    expect(appRouteLocation(timeline)).toEqual({
      name: 'video-timeline',
      params: { projectId: 'project-1', assetId: 'asset-1' },
    })
    expect(appRouteLocation(shot)).toEqual({
      name: 'project-scene',
      params: { projectId: 'project-1', sceneId: 'shot-2' },
    })
  })

  it('round-trips opaque and unicode resource identifiers without path ambiguity', () => {
    const route = videoTimelineRoute('project/Spain 2026', 'cam A/верх', {
      time: 12.25,
      panel: 'quality',
      subject: 'player/8',
    })
    const path = appRoutePath(route)

    expect(path).toBe('/projects/project%2FSpain%202026/videos/cam%20A%2F%D0%B2%D0%B5%D1%80%D1%85/timeline?t=12.25&panel=quality&subject=player%2F8')
    expect(parseAppRoute(path)).toEqual(route)
  })

  it('accepts redirect sources but always returns the explicit overview intent', () => {
    expect(parseAppRoute('/')).toEqual(projectsRoute())
    expect(parseAppRoute('/projects/')).toEqual(projectsRoute())
    expect(parseAppRoute('/projects/project-1')).toEqual(projectWorkspaceRoute('project-1', 'overview'))
    expect(parseAppRoute('http://localhost:5188/projects/project-1/analysis')).toEqual(
      projectWorkspaceRoute('project-1', 'analysis'),
    )
  })

  it('rejects unknown, malformed and legacy-ambiguous paths', () => {
    for (const path of [
      '/project/project-1',
      '/projects/project-1/settings',
      '/projects/project-1/editor',
      '/projects/project-1/editor/shot-2',
      '/projects/project-1/videos/asset-1',
      '/projects/project-1/videos/asset-1/timeline/extra',
      '/projects//overview',
      '/projects/%E0%A4%A/overview',
    ]) {
      expect(parseAppRoute(path), path).toBeNull()
    }
  })
})

describe('editor route view state', () => {
  it('allows only finite non-negative time, an allowlisted panel and a non-empty subject', () => {
    expect(parseEditorRouteView(new URLSearchParams({
      t: '3.5',
      panel: 'inspector',
      subject: 'person-8',
      provider: 'must-not-become-resource-identity',
    }))).toEqual({ time: 3.5, panel: 'inspector', subject: 'person-8' })

    expect(editorRouteQuery({
      time: 0,
      panel: 'events',
      subject: 'ball',
    })).toEqual({ t: '0', panel: 'events', subject: 'ball' })
  })

  it('drops invalid or duplicate view state instead of guessing', () => {
    expect(parseEditorRouteView(new URLSearchParams('t=-1&panel=settings&subject=%20'))).toBeUndefined()
    expect(parseEditorRouteView(new URLSearchParams('t=1&t=2&panel=quality&panel=events&subject=a&subject=b'))).toBeUndefined()
    expect(editorRouteQuery({ time: Number.POSITIVE_INFINITY })).toBeUndefined()
  })
})
