import { effectScope, nextTick, ref, shallowRef } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useEditorIdentityContext } from './useEditorIdentityContext'
import type { EditorAnalysisContext } from '../analysis/useEditorAnalysisContext'
import type { EditorCompositionContext } from '../composition/useEditorCompositionContext'
import type { EditorSessionContext } from '../session/useEditorSessionContext'
import type { EditorViewportContext } from '../viewport/useEditorViewportContext'
import type { SceneDocument } from '../../../types/scene'

const mocks = vi.hoisted(() => ({
  ensureLoaded: vi.fn(async () => undefined),
  invalidate: vi.fn(),
}))

vi.mock('../../../composables/useIdentityReviewEditor', () => ({
  useIdentityReviewEditor: () => ({
    snapshot: ref(null),
    loading: ref(false),
    error: ref(null),
    decisionSaving: ref(false),
    rosterBindingSaving: ref(false),
    ensureLoaded: mocks.ensureLoaded,
    reload: vi.fn(async () => undefined),
    load: vi.fn(async () => undefined),
    invalidate: mocks.invalidate,
    confirm: vi.fn(),
    reject: vi.fn(),
    unbind: vi.fn(),
    clearBinding: vi.fn(),
    updateBinding: vi.fn(),
  }),
}))

vi.mock('../../../composables/useIdentityReviewPresentation', () => ({
  useIdentityReviewPresentation: () => ({
    item: ref(null),
    person: ref(null),
    observations: ref(null),
    workers: ref(null),
    dedicatedUnbindActive: ref(false),
  }),
}))

vi.mock('../../../composables/useProjectMatchEditor', () => ({
  useProjectMatchEditor: () => ({
    refreshing: ref(false),
    importing: ref(false),
    importError: ref(null),
    refresh: vi.fn(),
    importFile: vi.fn(),
  }),
}))

function readyScene(): SceneDocument {
  return {
    id: 'segment-scene',
    title: '1-A',
    version: 1,
    revision: 7,
    duration: 4,
    payload: {
      pitch: { length: 105, width: 68 },
      videoAsset: {
        id: 'asset-1',
        filename: 'match.mp4',
        mediaUrl: '/match.mp4',
        posterUrl: '/poster.jpg',
        fps: 25,
        frameCount: 100,
        processingState: 'ready',
        selectedSegmentId: 'shot-1',
        reconstruction: {
          status: 'ready',
          artifactManifest: {
            schemaVersion: 1,
            artifacts: {
              identityDiagnostics: {
                id: 'sha256:diagnostics-1',
                kind: 'reconstruction-identity-diagnostics',
                schemaVersion: 1,
                uri: 'artifact://sha256/diagnostics-1',
                sha256: 'diagnostics-1',
                byteSize: 512,
                contentType: 'application/json',
              },
              identityTimeline: {
                id: 'sha256:timeline-1',
                kind: 'reconstruction-identity-timeline',
                schemaVersion: 1,
                uri: 'artifact://sha256/timeline-1',
                sha256: 'timeline-1',
                byteSize: 1024,
                contentType: 'application/json',
              },
            },
          },
        },
      },
      teams: [],
      tracks: [],
      canonicalPeople: [],
      ball: { keyframes: [] },
      eventBindings: [],
      cameraCuts: [],
    },
  }
}

describe('useEditorIdentityContext review synchronization', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('rechecks demand only after a successful reconstruction terminal sync', async () => {
    const scene = shallowRef<SceneDocument | null>(readyScene())
    const terminalSync = ref<{
      runId: string
      sceneId: string
      status: 'succeeded' | 'failed' | 'cancelled'
    } | null>(null)
    const session = {
      scene,
      workspaceMatch: ref(null),
      editorProjectId: () => 'project-1',
      saveState: ref(''),
      error: ref(null),
      selectedProject: ref(null),
      loadProjectsWorkspace: vi.fn(),
      router: { push: vi.fn() },
    } as unknown as EditorSessionContext
    const viewport = {
      selectedCanonicalPersonId: ref<string | null>('canonical-person-1'),
      selectedTrackId: ref<string | null>(null),
      selectedFramePersonId: ref<string | null>(null),
      activeTab: ref<'binding' | 'qa' | 'events'>('binding'),
      playing: ref(false),
      sourceVideo: ref<HTMLVideoElement | null>(null),
      viewMode: ref<'video' | 'split' | '3d'>('split'),
      currentTime: ref(0),
      seekTo: vi.fn(),
    } as unknown as EditorViewportContext
    const analysis = {
      reconstruction: {
        mutationLocked: ref(false),
        running: ref(false),
        reconstructing: ref(false),
        terminalSync,
        selectedModel: ref('yolo26m.pt'),
        selectedBallBackend: ref('dedicated-ultralytics'),
        startPolling: vi.fn(async () => undefined),
      },
      frameAnalysis: {
        canonicalPersonById: vi.fn(() => null),
        renderTrackForCanonicalPerson: vi.fn(() => null),
        clear: vi.fn(),
        analyze: vi.fn(),
      },
    } as unknown as EditorAnalysisContext
    const composition = {
      selection: { selectedCanonicalPerson: ref(null) },
      canonicalHasActiveDedicatedUnbind: vi.fn(() => false),
    } as unknown as EditorCompositionContext
    const scope = effectScope()
    scope.run(() => useEditorIdentityContext(session, viewport, analysis, composition))
    await nextTick()
    mocks.ensureLoaded.mockClear()

    terminalSync.value = { runId: 'run-1', sceneId: 'segment-scene', status: 'failed' }
    await nextTick()
    terminalSync.value = { runId: 'run-1', sceneId: 'segment-scene', status: 'cancelled' }
    await nextTick()
    expect(mocks.ensureLoaded).not.toHaveBeenCalled()

    terminalSync.value = { runId: 'run-1', sceneId: 'segment-scene', status: 'succeeded' }
    await nextTick()
    expect(mocks.ensureLoaded).toHaveBeenCalledOnce()
    expect(mocks.ensureLoaded).toHaveBeenCalledWith('segment-scene')

    scope.stop()
  })
})
