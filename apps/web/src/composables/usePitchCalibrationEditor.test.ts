import { ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { SceneDocument } from '../types/scene'
import { usePitchCalibrationEditor } from './usePitchCalibrationEditor'

const mocks = vi.hoisted(() => ({
  setAttackingGoal: vi.fn(),
  saveDraft: vi.fn(),
}))

vi.mock('../lib/api/calibration', () => ({
  calibrationClient: {
    auto: vi.fn(),
    preview: vi.fn(),
    saveDraft: mocks.saveDraft,
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
    })

    await editor.changeAttackingGoal('left')

    expect(selectedTrackId.value).toBeNull()
    expect(selectedCanonicalPersonId.value).toBeNull()
  })

  it('keeps the selected actor while staging a frame correction without polling', async () => {
    const document = ref<SceneDocument | null>(scene())
    const selectedTrackId = ref<string | null>('track-first')
    const selectedCanonicalPersonId = ref<string | null>('person-first')
    const saveState = ref('')
    const polling = vi.fn(async () => undefined)
    mocks.saveDraft.mockResolvedValue(scene())
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
      saveState,
      error: ref(null),
      projectId: () => 'project-1',
      seekTo: vi.fn(),
      clearFrameAnalysis: vi.fn(),
    })
    editor.draft.value = {
      sceneId: 'segment-scene',
      sceneTime: 0,
      frameIndex: 1,
      frameWidth: 1280,
      frameHeight: 720,
      source: 'manual',
      preset: 'center-circle',
      confidence: 0.9,
      alignmentError: 1,
      quality: 'good',
      anchors: [],
      markings: [],
      imageToPitch: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      warnings: [],
    }

    await editor.apply()

    expect(selectedTrackId.value).toBe('track-first')
    expect(selectedCanonicalPersonId.value).toBe('person-first')
    expect(saveState.value).toContain('Frame correction staged')
    expect(polling).not.toHaveBeenCalled()
  })
})
