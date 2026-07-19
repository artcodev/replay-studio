import { ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useIdentityReviewEditor } from './useIdentityReviewEditor'
import type { IdentityReviewResponse } from '../types/identityReview'
import type { SceneDocument } from '../types/scene'

const mocks = vi.hoisted(() => ({
  review: vi.fn(),
}))

vi.mock('../lib/api/identities', () => ({
  identityClient: {
    review: mocks.review,
    updateRosterBinding: vi.fn(),
    clearRosterBinding: vi.fn(),
    rejectRosterCandidate: vi.fn(),
  },
}))

const PERSON_ID = 'canonical-person-1'

function review(sceneId: string, revision: number): IdentityReviewResponse {
  return {
    sceneId,
    revision,
    availability: { state: 'ready', available: true },
    matchSnapshot: {
      roster: {
        status: 'ready',
        playerCount: 22,
        complete: true,
        automaticIdentityEligible: true,
        manualIdentityEligible: true,
        reasons: [],
        warnings: [],
      },
    },
    workers: {},
    summary: {
      canonicalPersonCount: 1,
      boundCount: 0,
      suggestedCount: 1,
      conflictCount: 0,
      anonymousCount: 0,
      excludedCount: 0,
    },
    items: [],
  }
}

function scene(
  status: 'queued' | 'processing' | 'ready' | 'cancelled' | 'failed' = 'ready',
): SceneDocument {
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
          status,
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
      canonicalPeople: [{
        canonicalPersonId: PERSON_ID,
        displayName: 'Person 1',
        identityStatus: 'provisional',
        identityConfidence: null,
        identitySource: null,
        teamId: null,
        role: 'player',
        jerseyNumber: null,
        externalPlayerId: null,
        memberTrackletIds: [],
        evidence: [],
        rosterCandidates: [],
        conflicts: [],
      }],
      tracks: [],
      ball: { keyframes: [] },
      eventBindings: [],
      cameraCuts: [],
    },
  }
}

function harness(initialScene: SceneDocument | null = scene()) {
  const document = ref<SceneDocument | null>(initialScene)
  const selectedCanonicalPersonId = ref<string | null>(PERSON_ID)
  const activeTab = ref<'binding' | 'qa' | 'events'>('binding')
  const reconstructing = ref(false)
  const running = ref(false)
  const editor = useIdentityReviewEditor({
    projectId: () => 'project-1',
    scene: document,
    rosterPlayers: () => [],
    mutationLocked: () => false,
    reconstructionRunning: () => running.value,
    reconstructing,
    selectedCanonicalPersonId,
    selectedTrackId: ref(null),
    selectedFramePersonId: ref(null),
    activeTab,
    saveState: ref(''),
    error: ref(null),
    canonicalPersonById: () => null,
    renderTrackForCanonicalPerson: () => null,
    hasDedicatedUnbind: () => false,
    clearFrameAnalysis: vi.fn(),
    startReconstructionPolling: vi.fn(async () => undefined),
  })
  return {
    activeTab,
    document,
    editor,
    reconstructing,
    running,
    selectedCanonicalPersonId,
  }
}

describe('useIdentityReviewEditor demand loading', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('does not request review without demand or a ready segment diagnostics artifact', async () => {
    const state = harness()

    state.selectedCanonicalPersonId.value = null
    await state.editor.ensureLoaded()
    state.selectedCanonicalPersonId.value = PERSON_ID
    state.activeTab.value = 'qa'
    await state.editor.ensureLoaded()

    state.activeTab.value = 'binding'
    state.document.value!.payload.videoAsset!.selectedSegmentId = undefined
    await state.editor.ensureLoaded()
    state.document.value!.payload.videoAsset!.selectedSegmentId = 'shot-1'
    state.document.value!.payload.videoAsset!.reconstruction = undefined
    await state.editor.ensureLoaded()

    for (const status of ['queued', 'processing', 'failed', 'cancelled'] as const) {
      state.document.value = scene(status)
      await state.editor.ensureLoaded()
    }

    state.document.value = scene('ready')
    const manifest = state.document.value.payload.videoAsset!.reconstruction!.artifactManifest!
    manifest.artifacts.identityTimeline = undefined
    await state.editor.ensureLoaded()
    expect(mocks.review).not.toHaveBeenCalled()
    manifest.artifacts.identityTimeline = scene('ready').payload.videoAsset!
      .reconstruction!.artifactManifest!.artifacts.identityTimeline
    manifest.artifacts.identityDiagnostics = undefined
    await state.editor.ensureLoaded()
    state.document.value = scene('ready')
    state.running.value = true
    await state.editor.ensureLoaded()
    state.running.value = false
    state.reconstructing.value = true
    await state.editor.ensureLoaded()

    expect(mocks.review).not.toHaveBeenCalled()
    expect(state.editor.snapshot.value).toBeNull()
  })

  it('loads once per scene revision and artifact, deduplicates, and force reloads on Retry', async () => {
    const state = harness()
    mocks.review.mockImplementation(async (_projectId: string, sceneId: string) => (
      review(sceneId, state.document.value!.revision)
    ))

    await Promise.all([
      state.editor.ensureLoaded(),
      state.editor.ensureLoaded('segment-scene'),
    ])
    await state.editor.ensureLoaded()

    expect(mocks.review).toHaveBeenCalledTimes(1)
    expect(state.editor.snapshot.value?.revision).toBe(7)

    await state.editor.load('segment-scene')
    expect(mocks.review).toHaveBeenCalledTimes(2)

    state.document.value = { ...state.document.value!, revision: 8 }
    await state.editor.ensureLoaded()
    expect(mocks.review).toHaveBeenCalledTimes(3)
    expect(state.editor.snapshot.value?.revision).toBe(8)

    const reconstruction = state.document.value.payload.videoAsset!.reconstruction!
    const diagnostics = reconstruction.artifactManifest!.artifacts.identityDiagnostics!
    reconstruction.artifactManifest = {
      ...reconstruction.artifactManifest!,
      artifacts: {
        ...reconstruction.artifactManifest!.artifacts,
        identityDiagnostics: {
          ...diagnostics,
          id: 'sha256:diagnostics-2',
          uri: 'artifact://sha256/diagnostics-2',
          sha256: 'diagnostics-2',
        },
      },
    }
    await state.editor.ensureLoaded()
    expect(mocks.review).toHaveBeenCalledTimes(4)
  })

  it('does not publish a stale response after the scene revision changes', async () => {
    const state = harness()
    let resolveFirst!: (value: IdentityReviewResponse) => void
    mocks.review
      .mockImplementationOnce(() => new Promise<IdentityReviewResponse>((resolve) => {
        resolveFirst = resolve
      }))
      .mockResolvedValueOnce(review('segment-scene', 8))

    const staleRequest = state.editor.ensureLoaded()
    state.document.value = { ...state.document.value!, revision: 8 }
    await state.editor.ensureLoaded()
    resolveFirst(review('segment-scene', 7))
    await staleRequest

    expect(mocks.review).toHaveBeenCalledTimes(2)
    expect(state.editor.snapshot.value?.revision).toBe(8)
    expect(state.editor.error.value).toBeNull()
  })
})
