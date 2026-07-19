import { ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { SceneDocument } from '../types/scene'
import { usePitchCalibrationEditor } from './usePitchCalibrationEditor'

const mocks = vi.hoisted(() => ({
  setAttackingGoal: vi.fn(),
}))

vi.mock('../lib/api/calibration', () => ({
  calibrationClient: {
    auto: vi.fn(),
    preview: vi.fn(),
    apply: vi.fn(),
    setAttackingGoal: mocks.setAttackingGoal,
  },
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

describe('usePitchCalibrationEditor selection', () => {
  beforeEach(() => vi.clearAllMocks())

  it('does not select the first actor after changing the attacking goal', async () => {
    const document = ref<SceneDocument | null>(scene())
    const selectedTrackId = ref<string | null>(null)
    const selectedCanonicalPersonId = ref<string | null>(null)
    mocks.setAttackingGoal.mockResolvedValue(scene())
    const editor = usePitchCalibrationEditor({
      scene: document,
      currentTime: ref(0),
      activeTab: ref('binding'),
      playing: ref(false),
      sourceVideo: ref<HTMLVideoElement | null>(null),
      viewMode: ref('split'),
      reconstructing: ref(false),
      selectedTrackId,
      selectedCanonicalPersonId,
      calibrationFrames: ref([]),
      saveState: ref(''),
      error: ref(null),
      projectId: () => 'project-1',
      seekTo: vi.fn(),
      clearFrameAnalysis: vi.fn(),
      startReconstructionPolling: vi.fn(async () => undefined),
    })

    await editor.changeAttackingGoal('left')

    expect(selectedTrackId.value).toBeNull()
    expect(selectedCanonicalPersonId.value).toBeNull()
  })
})
