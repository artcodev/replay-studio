import { effectScope, reactive, ref, shallowRef } from 'vue'
import type { RouteLocationNormalizedLoaded, Router } from 'vue-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ProjectWorkspace } from './useProjectWorkspace'
import type { useSceneSession } from './useSceneSession'
import { useEditorRouteSession } from './useEditorRouteSession'
import type { Project, ProjectAsset, ProjectSegment } from '../types/project'
import type { SceneDocument, SceneSummary } from '../types/scene'

const vueMocks = vi.hoisted(() => ({
  nextTick: vi.fn(async (): Promise<void> => undefined),
}))

vi.mock('vue', async () => ({
  ...await vi.importActual<typeof import('vue')>('vue'),
  nextTick: vueMocks.nextTick,
}))

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

function scene(
  id: string,
  values: {
    duration: number
    sourceStart?: number
    selectedSegmentId?: string
    canonicalPersonId?: string
  },
): SceneDocument {
  const canonicalPersonId = values.canonicalPersonId
  return {
    id,
    title: id,
    version: 1,
    revision: 1,
    duration: values.duration,
    payload: {
      pitch: { length: 105, width: 68 },
      videoAsset: {
        id: 'asset-1',
        filename: 'match.mp4',
        mediaUrl: '/match.mp4',
        posterUrl: '/poster.jpg',
        fps: 25,
        frameCount: 1500,
        processingState: 'ready',
        sourceStart: values.sourceStart,
        selectedSegmentId: values.selectedSegmentId,
      },
      teams: [],
      canonicalPeople: canonicalPersonId ? [{
        canonicalPersonId,
        displayName: canonicalPersonId,
        identityStatus: 'provisional',
        identityConfidence: null,
        identitySource: null,
        teamId: null,
        role: null,
        jerseyNumber: null,
        externalPlayerId: null,
        memberTrackletIds: [`tracklet-${canonicalPersonId}`],
        evidence: [],
        rosterCandidates: [],
        conflicts: [],
      }] : [],
      tracks: canonicalPersonId ? [{
        id: `track-${canonicalPersonId}`,
        label: canonicalPersonId,
        teamId: 'home',
        color: '#ffffff',
        number: 1,
        canonicalPersonId,
        externalPlayerId: null,
        keyframes: [],
      }] : [],
      ball: { keyframes: [] },
      eventBindings: [],
      cameraCuts: [],
    },
  }
}

describe('editor route session scene ownership', () => {
  beforeEach(() => {
    vueMocks.nextTick.mockReset()
    vueMocks.nextTick.mockResolvedValue(undefined)
  })

  it('keeps the timeline model without selecting actors unless the route names one', async () => {
    const project: Project = {
      id: 'project-1',
      title: 'Match',
      revision: 1,
      createdAt: '2026-07-18T00:00:00Z',
      updatedAt: '2026-07-18T00:00:00Z',
    }
    const asset: ProjectAsset = {
      id: 'asset-1',
      projectId: project.id,
      timelineSceneId: 'timeline-scene',
      filename: 'match.mp4',
      duration: 60,
      status: 'ready',
      createdAt: project.createdAt,
    }
    const segment: ProjectSegment = {
      id: 'segment-1',
      projectId: project.id,
      assetId: asset.id,
      sourceSegmentId: 'shot-1',
      sceneId: 'segment-scene',
      label: '1-A',
      start: 10,
      end: 15,
      status: 'ready',
    }
    const documents = new Map([
      ['timeline-scene', scene('timeline-scene', {
        duration: 60,
        canonicalPersonId: 'timeline-person',
      })],
      ['segment-scene', scene('segment-scene', {
        duration: 5,
        sourceStart: 10,
        selectedSegmentId: 'shot-1',
        canonicalPersonId: 'segment-person',
      })],
    ])
    const summaries: SceneSummary[] = [
      { id: 'timeline-scene', title: 'Timeline', duration: 60, kind: 'video' },
      { id: 'segment-scene', title: '1-A', duration: 5, kind: 'segment' },
    ]
    const activeScene = shallowRef<SceneDocument | null>(null)
    const timelineScene = shallowRef<SceneDocument | null>(null)
    const sceneSession = {
      saving: ref(false),
      list: vi.fn(async () => summaries),
      load: vi.fn(async (id: string) => {
        const document = documents.get(id) ?? null
        activeScene.value = document
        return document
      }),
      read: vi.fn(async (id: string) => {
        const document = documents.get(id)
        if (!document) throw new Error(`Unknown scene ${id}`)
        return document
      }),
      refresh: vi.fn(),
      save: vi.fn(),
      cancelPendingLoad: vi.fn(),
    } satisfies ReturnType<typeof useSceneSession>
    const workspace = {
      catalog: {
        projects: ref([project]),
        project: ref<Project | null>(project),
        loading: ref(false),
        error: ref<string | null>(null),
      },
      media: {
        assets: ref([asset]),
        segments: ref([segment]),
      },
      load: vi.fn(async () => 'loaded' as const),
    } as unknown as ProjectWorkspace
    const route = reactive({
      fullPath: '/projects/project-1/videos/asset-1/timeline',
    }) as RouteLocationNormalizedLoaded
    const router = {
      push: vi.fn(async () => undefined),
      replace: vi.fn(async () => undefined),
    } as unknown as Router
    const scope = effectScope()
    const selectedTrackId = ref<string | null>('stale-track')
    const selectedCanonicalPersonId = ref<string | null>('stale-person')
    const activeTab = ref<'binding' | 'qa' | 'events'>('binding')

    scope.run(() => useEditorRouteSession({
      route,
      router,
      workspace,
      scene: activeScene,
      timelineScene,
      sceneSession,
      playing: ref(false),
      sourceVideo: ref<HTMLVideoElement | null>(null),
      error: ref<string | null>(null),
      selectedTrackId,
      selectedCanonicalPersonId,
      activeTab,
      seekTo: vi.fn(),
      exitEditor: vi.fn(),
    }))

    await vi.waitFor(() => {
      expect(activeScene.value?.id).toBe('timeline-scene')
      expect(timelineScene.value?.id).toBe('timeline-scene')
    })
    expect(selectedTrackId.value).toBeNull()
    expect(selectedCanonicalPersonId.value).toBeNull()

    route.fullPath = '/projects/project-1/segments/segment-1'
    await vi.waitFor(() => {
      expect(activeScene.value?.id).toBe('segment-scene')
      expect(timelineScene.value?.id).toBe('timeline-scene')
    })
    expect(selectedTrackId.value).toBeNull()
    expect(selectedCanonicalPersonId.value).toBeNull()

    route.fullPath = '/projects/project-1/segments/segment-1?subject=segment-person'
    await vi.waitFor(() => {
      expect(selectedTrackId.value).toBe('track-segment-person')
      expect(selectedCanonicalPersonId.value).toBe('segment-person')
    })

    route.fullPath = '/projects/project-1/segments/segment-1?panel=quality'
    await vi.waitFor(() => {
      expect(activeTab.value).toBe('qa')
    })
    expect(selectedTrackId.value).toBe('track-segment-person')
    expect(selectedCanonicalPersonId.value).toBe('segment-person')
    expect(sceneSession.load).toHaveBeenNthCalledWith(1, 'timeline-scene')
    expect(sceneSession.load).toHaveBeenNthCalledWith(2, 'segment-scene')
    expect(sceneSession.read).not.toHaveBeenCalled()

    scope.stop()
  })

  it('does not apply a stale time, panel, or subject after navigation wins a route race', async () => {
    const project: Project = {
      id: 'project-1',
      title: 'Match',
      revision: 1,
      createdAt: '2026-07-18T00:00:00Z',
      updatedAt: '2026-07-18T00:00:00Z',
    }
    const sceneA = scene('scene-a', {
      duration: 5,
      selectedSegmentId: 'shot-a',
      canonicalPersonId: 'shared-person',
    })
    const sceneB = scene('scene-b', {
      duration: 5,
      selectedSegmentId: 'shot-b',
      canonicalPersonId: 'shared-person',
    })
    const documents = new Map([
      [sceneA.id, sceneA],
      [sceneB.id, sceneB],
    ])
    const summaries: SceneSummary[] = [
      { id: sceneA.id, title: 'A', duration: 5, kind: 'segment' },
      { id: sceneB.id, title: 'B', duration: 5, kind: 'segment' },
    ]
    const activeScene = shallowRef<SceneDocument | null>(null)
    const timelineScene = shallowRef<SceneDocument | null>(null)
    const sceneSession = {
      saving: ref(false),
      list: vi.fn(async () => summaries),
      load: vi.fn(async (id: string) => {
        const document = documents.get(id) ?? null
        activeScene.value = document
        return document
      }),
      read: vi.fn(async (id: string) => {
        const document = documents.get(id)
        if (!document) throw new Error(`Unknown scene ${id}`)
        return document
      }),
      refresh: vi.fn(),
      save: vi.fn(),
      cancelPendingLoad: vi.fn(),
    } satisfies ReturnType<typeof useSceneSession>
    const workspace = {
      catalog: {
        projects: ref([project]),
        project: ref<Project | null>(project),
        loading: ref(false),
        error: ref<string | null>(null),
      },
      media: { assets: ref([]), segments: ref([]) },
      load: vi.fn(async () => 'loaded' as const),
    } as unknown as ProjectWorkspace
    const route = reactive({
      fullPath: '/projects/project-1/scenes/scene-a?t=3.5&panel=events&subject=shared-person',
    }) as RouteLocationNormalizedLoaded
    const router = {
      push: vi.fn(async () => undefined),
      replace: vi.fn(async () => undefined),
    } as unknown as Router
    const selectedTrackId = ref<string | null>(null)
    const selectedCanonicalPersonId = ref<string | null>(null)
    const activeTab = ref<'binding' | 'qa' | 'events'>('binding')
    const seekTo = vi.fn()
    const delayedTick = deferred<void>()
    vueMocks.nextTick.mockReturnValueOnce(delayedTick.promise)
    const scope = effectScope()
    let routeSession!: ReturnType<typeof useEditorRouteSession>

    scope.run(() => {
      routeSession = useEditorRouteSession({
        route,
        router,
        workspace,
        scene: activeScene,
        timelineScene,
        sceneSession,
        playing: ref(false),
        sourceVideo: ref<HTMLVideoElement | null>(null),
        error: ref<string | null>(null),
        selectedTrackId,
        selectedCanonicalPersonId,
        activeTab,
        seekTo,
        exitEditor: vi.fn(),
      })
    })
    await vi.waitFor(() => expect(vueMocks.nextTick).toHaveBeenCalled())

    route.fullPath = '/projects/project-1/scenes/scene-b'
    await routeSession.sync()
    expect(activeScene.value?.id).toBe('scene-b')
    delayedTick.resolve()
    await delayedTick.promise
    await Promise.resolve()

    expect(seekTo).not.toHaveBeenCalledWith(3.5)
    expect(activeTab.value).toBe('binding')
    expect(selectedTrackId.value).toBeNull()
    expect(selectedCanonicalPersonId.value).toBeNull()

    scope.stop()
  })
})
