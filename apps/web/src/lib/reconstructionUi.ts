import type { FrameAnalysis } from '../types/analysis'
import type { ReconstructionQuality } from '../types/reconstruction'
import type { SceneDocument } from '../types/scene'
import type { AnalysisJob } from '../types/project'

export function reconstructionLocksMutations(
  status: 'queued' | 'processing' | 'ready' | 'cancelled' | 'failed' | undefined,
  optimisticRunning = false,
): boolean {
  return optimisticRunning || status === 'queued' || status === 'processing'
}

export type ReconstructionSceneStatus =
  | 'queued'
  | 'processing'
  | 'ready'
  | 'cancelled'
  | 'failed'
  | undefined

/**
 * The compact AnalysisRun record is the cancellation authority. Scene payloads
 * can lag behind it while the worker acknowledges a cancel request, so an
 * exact run-id match must win over stale queued/processing metadata.
 */
export function reconstructionStatusFromAnalysisJob(
  sceneStatus: ReconstructionSceneStatus,
  jobStatus: AnalysisJob['status'] | undefined,
): ReconstructionSceneStatus {
  if (!jobStatus) return sceneStatus
  if (jobStatus === 'queued') return 'queued'
  if (jobStatus === 'running' || jobStatus === 'cancelling') return 'processing'
  if (jobStatus === 'cancelled') return 'cancelled'
  if (jobStatus === 'succeeded') return 'ready'
  return 'failed'
}

export function matchingReconstructionAnalysisJob(
  scene: SceneDocument | null,
  jobs: AnalysisJob[],
): AnalysisJob | null {
  const runId = scene?.payload.videoAsset?.reconstruction?.runId
  if (!runId) return null
  return jobs.find((job) => job.id === runId) ?? null
}

/**
 * Frame-correction endpoints queue a reconstruction and return the new run
 * metadata with the refreshed frame payload. Merge that metadata immediately
 * so the editor cannot save an older whole-scene document over the queued run.
 */
export function mergeFrameReconstructionMetadata(
  scene: SceneDocument,
  analysis: Pick<FrameAnalysis, 'reconstruction'>,
): SceneDocument {
  const metadata = analysis.reconstruction
  const videoAsset = scene.payload.videoAsset
  if (!metadata || !videoAsset) return scene

  return {
    ...scene,
    payload: {
      ...scene.payload,
      videoAsset: {
        ...videoAsset,
        reconstruction: {
          ...videoAsset.reconstruction,
          status: metadata.status,
          processingStatus: metadata.status === 'ready' ? 'completed' : metadata.status,
          ...(metadata.runId === undefined ? {} : { runId: metadata.runId }),
          ...(metadata.runRevision === undefined ? {} : { runRevision: metadata.runRevision }),
          ...(metadata.inputFingerprint === undefined ? {} : { inputFingerprint: metadata.inputFingerprint }),
          ...(metadata.status === 'queued' || metadata.status === 'processing'
            ? {
                qualityVerdict: 'pending' as const,
                quality: undefined,
                qualityReport: undefined,
                progress: undefined,
                error: null,
              }
            : {}),
        },
      },
    },
  }
}

export function identityValidationSummary(
  validation: ReconstructionQuality['identityValidation'] | null | undefined,
): string {
  if (!validation) return 'Identity validation unavailable'
  if (validation?.status === 'invalid') {
    return `Identity validation invalid${validation.reason ? ` · ${validation.reason}` : ''}`
  }
  if (validation.status === 'unavailable' || !validation.groundTruthAvailable) {
    return `Identity accuracy · ground truth unavailable${validation.reason ? ` · ${validation.reason}` : ''}`
  }
  if (validation.idf1 === null || validation.idf1 === undefined) {
    return 'Identity validation invalid · evaluated result has no IDF1 value'
  }
  const switches = validation.idSwitchCount === null || validation.idSwitchCount === undefined
    ? 'ID switches unavailable'
    : `${validation.idSwitchCount} ID switches`
  return `Labelled IDF1 ${Math.round(validation.idf1 * 100)}% · ${switches}`
}
