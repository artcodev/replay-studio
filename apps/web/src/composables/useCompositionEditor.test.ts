import { computed, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { VideoSegment } from '../types/media'
import type { SceneDocument } from '../types/scene'
import { useCompositionEditor } from './useCompositionEditor'

const mocks = vi.hoisted(() => ({
  createSegmentScene: vi.fn(),
}))

vi.mock('../lib/api/media', () => ({
  mediaClient: {
    createSegmentScene: mocks.createSegmentScene,
    createComposition: vi.fn(),
  },
}))

vi.mock('../lib/api/scenes', () => ({
  sceneClient: { get: vi.fn() },
}))

function scene(id: string): SceneDocument {
  return {
    id,
    title: id,
    version: 1,
    revision: 1,
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

const segment: VideoSegment = {
  id: 'shot-1',
  label: '1-A',
  start: 0,
  end: 4,
  duration: 4,
  score: 0.9,
}

describe('useCompositionEditor selection', () => {
  beforeEach(() => vi.clearAllMocks())

  it('does not select the first actor when a segment Scene is created', async () => {
    const root = scene('timeline-scene')
    const document = ref<SceneDocument | null>(root)
    const selectedTrackId = ref<string | null>('track-first')
    const selectedCanonicalPersonId = ref<string | null>('person-first')
    mocks.createSegmentScene.mockResolvedValue(scene('segment-scene'))
    const navigateToScene = vi.fn(async () => undefined)
    const editor = useCompositionEditor({
      projectId: () => 'project-1',
      scene: document,
      sceneVideo: computed(() => root.payload.videoAsset ?? null),
      projectSegments: ref([]),
      multiPassSelection: ref([]),
      selectedTrackId,
      selectedCanonicalPersonId,
      currentTime: ref(2),
      reconstructing: ref(false),
      saveState: ref(''),
      error: ref(null),
      navigateToScene,
      startReconstructionPolling: vi.fn(async () => undefined),
      seekTo: vi.fn(),
    })

    await editor.createSceneFromSegment(segment)

    expect(document.value?.id).toBe('segment-scene')
    expect(selectedTrackId.value).toBeNull()
    expect(selectedCanonicalPersonId.value).toBeNull()
    expect(navigateToScene).toHaveBeenCalledWith('segment-scene')
  })
})
