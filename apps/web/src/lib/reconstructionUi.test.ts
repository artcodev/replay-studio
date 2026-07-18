import { describe, expect, it } from 'vitest'
import type { SceneDocument } from '../types/scene'
import {
  identityValidationSummary,
  matchingReconstructionAnalysisJob,
  mergeFrameReconstructionMetadata,
  reconstructionLocksMutations,
  reconstructionStatusFromAnalysisJob,
} from './reconstructionUi'

function scene(): SceneDocument {
  return {
    id: 'scene-1',
    title: 'Moment',
    version: 4,
    revision: 1,
    duration: 3,
    payload: {
      pitch: { length: 105, width: 68 },
      videoAsset: {
        id: 'video-1',
        filename: 'clip.mp4',
        mediaUrl: '/clip.mp4',
        posterUrl: '/clip.jpg',
        fps: 25,
        frameCount: 75,
        processingState: 'ready',
        reconstruction: {
          status: 'ready',
          processingStatus: 'completed',
          qualityVerdict: 'pass',
          quality: { verdict: 'pass' },
          progress: {
            phase: 'finalizing',
            phaseIndex: 6,
            phaseCount: 6,
            label: 'Complete',
            detail: 'Previous run complete',
            completed: 1,
            total: 1,
            phasePercent: 100,
            overallPercent: 100,
            elapsedSeconds: 12,
            etaSeconds: 0,
            updatedAt: '2026-07-17T00:00:00Z',
            phases: [],
          },
          runId: 'run-old',
          runRevision: 4,
          model: 'yolo26m.pt',
        },
      },
      teams: [],
      tracks: [],
      canonicalPeople: [],
      ball: { mode: 'automatic', keyframes: [] },
      eventBindings: [],
      cameraCuts: [],
    },
  }
}

describe('reconstruction editor state', () => {
  it('locks mutations throughout optimistic, queued, and processing reconstruction', () => {
    expect(reconstructionLocksMutations('ready')).toBe(false)
    expect(reconstructionLocksMutations('cancelled')).toBe(false)
    expect(reconstructionLocksMutations('failed')).toBe(false)
    expect(reconstructionLocksMutations('queued')).toBe(true)
    expect(reconstructionLocksMutations('processing')).toBe(true)
    expect(reconstructionLocksMutations('ready', true)).toBe(true)
  })

  it('lets the matching compact job release a stale scene lock after cancellation', () => {
    const current = scene()
    current.payload.videoAsset!.reconstruction!.status = 'processing'
    current.payload.videoAsset!.reconstruction!.runId = 'run-current'
    const job = {
      id: 'run-current',
      projectId: 'project-1',
      segmentId: 'segment-1',
      kind: 'reconstruction',
      status: 'cancelled' as const,
      phase: 'cancelled',
      progress: {
        completed: 0,
        total: 1,
        percent: 0,
        label: 'Analysis cancelled',
        detail: null,
        etaSeconds: 0,
      },
      createdAt: '2026-07-18T00:00:00Z',
    }

    expect(matchingReconstructionAnalysisJob(current, [job])?.id).toBe('run-current')
    const status = reconstructionStatusFromAnalysisJob(
      current.payload.videoAsset?.reconstruction?.status,
      job.status,
    )
    expect(status).toBe('cancelled')
    expect(reconstructionLocksMutations(status)).toBe(false)
  })

  it('does not apply a terminal job from another reconstruction run', () => {
    const current = scene()
    current.payload.videoAsset!.reconstruction!.status = 'processing'
    current.payload.videoAsset!.reconstruction!.runId = 'run-current'
    const staleJob = {
      id: 'run-old',
      projectId: 'project-1',
      segmentId: 'segment-1',
      kind: 'reconstruction',
      status: 'cancelled' as const,
      phase: 'cancelled',
      progress: {
        completed: 0,
        total: 1,
        percent: 0,
        label: 'Analysis cancelled',
        detail: null,
        etaSeconds: 0,
      },
      createdAt: '2026-07-18T00:00:00Z',
    }

    expect(matchingReconstructionAnalysisJob(current, [staleJob])).toBeNull()
    expect(reconstructionStatusFromAnalysisJob('processing', undefined)).toBe('processing')
  })

  it('merges queued correction metadata without dropping the current reconstruction', () => {
    const original = scene()
    const updated = mergeFrameReconstructionMetadata(original, {
      reconstruction: {
        status: 'queued',
        runId: 'run-new',
        runRevision: 5,
        inputFingerprint: 'fingerprint-new',
      },
    })

    expect(updated).not.toBe(original)
    expect(updated.payload.videoAsset?.reconstruction).toMatchObject({
      status: 'queued',
      processingStatus: 'queued',
      qualityVerdict: 'pending',
      runId: 'run-new',
      runRevision: 5,
      inputFingerprint: 'fingerprint-new',
      model: 'yolo26m.pt',
    })
    expect(updated.payload.videoAsset?.reconstruction?.quality).toBeUndefined()
    expect(updated.payload.videoAsset?.reconstruction?.progress).toBeUndefined()
    expect(original.payload.videoAsset?.reconstruction?.status).toBe('ready')
  })

  it('does not present unavailable or invalid validation as zero percent', () => {
    expect(identityValidationSummary({
      groundTruthAvailable: false,
      status: 'unavailable',
      idf1: null,
      reason: 'no labelled rows',
    })).toBe('Identity accuracy · ground truth unavailable · no labelled rows')
    expect(identityValidationSummary({
      groundTruthAvailable: true,
      status: 'invalid',
      idf1: null,
      reason: 'conflicting duplicate labels',
    })).toBe('Identity validation invalid · conflicting duplicate labels')
    expect(identityValidationSummary({
      groundTruthAvailable: true,
      status: 'evaluated',
      idf1: 0.873,
      idSwitchCount: 2,
    })).toBe('Labelled IDF1 87% · 2 ID switches')
    expect(identityValidationSummary({
      groundTruthAvailable: true,
      status: 'evaluated',
      idf1: 0.5,
    })).toBe('Labelled IDF1 50% · ID switches unavailable')
    expect(identityValidationSummary(undefined)).toBe('Identity validation unavailable')
  })
})
