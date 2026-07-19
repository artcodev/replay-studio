import { computed, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { AnalysisJob } from '../types/project'
import type { SceneDocument } from '../types/scene'
import { useModelComparison } from './useModelComparison'

const mocks = vi.hoisted(() => ({
  getScene: vi.fn(),
}))

vi.mock('../lib/api/scenes', () => ({
  sceneClient: { get: mocks.getScene },
}))

vi.mock('../lib/api/reconstruction', () => ({
  reconstructionClient: { compareModels: vi.fn() },
}))

function scene(): SceneDocument {
  return {
    id: 'segment-scene',
    title: '1-A',
    version: 1,
    revision: 2,
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
      },
      teams: [],
      canonicalPeople: [{ canonicalPersonId: 'person-first' }],
      tracks: [{ id: 'track-first', canonicalPersonId: 'person-first' }],
      ball: { keyframes: [] },
      eventBindings: [],
      cameraCuts: [],
    },
  } as unknown as SceneDocument
}

const terminalJob = {
  id: 'comparison-run',
  projectId: 'project-1',
  segmentId: 'segment-1',
  kind: 'model-comparison',
  status: 'succeeded',
  phase: null,
  progress: {
    completed: 1,
    total: 1,
    percent: 100,
    label: 'Complete',
    detail: null,
    etaSeconds: null,
  },
  createdAt: '2026-07-18T00:00:00Z',
} satisfies AnalysisJob

describe('useModelComparison selection', () => {
  beforeEach(() => vi.clearAllMocks())

  it('publishes the terminal report without selecting or focusing the first actor', async () => {
    const document = ref<SceneDocument | null>(scene())
    const selectedTrackId = ref<string | null>(null)
    const selectedCanonicalPersonId = ref<string | null>(null)
    mocks.getScene.mockResolvedValue(scene())
    const editor = useModelComparison({
      projectId: () => 'project-1',
      scene: document,
      sceneVideo: computed(() => document.value?.payload.videoAsset ?? null),
      jobs: ref([]),
      selectedTrackId,
      selectedCanonicalPersonId,
      playing: ref(false),
      sourceVideo: ref<HTMLVideoElement | null>(null),
      saveState: ref(''),
      error: ref(null),
      refreshJobs: vi.fn(async () => undefined),
    })
    editor.runId.value = terminalJob.id

    await editor.syncAfterTerminal(terminalJob)

    expect(selectedTrackId.value).toBeNull()
    expect(selectedCanonicalPersonId.value).toBeNull()
    expect(document.value?.revision).toBe(2)
  })
})
