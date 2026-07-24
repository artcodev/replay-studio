import { computed, ref, type Ref } from 'vue'
import { sceneClient } from '../lib/api/scenes'
import { reconstructionClient } from '../lib/api/reconstruction'
import { calibrationClient } from '../lib/api/calibration'
import { mediaClient } from '../lib/api/media'
import type {
  CalibrationDraftSource,
  PitchCalibrationAnchor,
  PitchCalibrationPreset,
} from '../types/calibration'
import { isAnalysisJobCancelable } from '../lib/analysisJobs'
import {
  matchingReconstructionAnalysisJob,
  reconstructionLocksMutations,
  reconstructionStatusFromAnalysisJob,
} from '../lib/reconstructionUi'
import type { BallDetectionBackend, CalibrationReview, ProcessingStatus, QualityVerdict, ReconstructionMode, ReconstructionModel, ReconstructionPhase } from '../types/reconstruction'
import type { SceneDocument } from '../types/scene'
import type { AnalysisJob } from '../types/project'
import { calibrationReviewFromEvidence } from '../features/calibration/calibrationQaPresentation'

type ReconstructionControllerOptions = {
  scene: Ref<SceneDocument | null>
  jobs: Readonly<Ref<AnalysisJob[]>>
  cancelingJobIds: Readonly<Ref<string[]>>
  refreshJobs: () => Promise<void>
  cancelJob: (runId: string) => Promise<AnalysisJob | null>
  saveState: Ref<string>
  error: Ref<string | null>
  projectId: () => string
  clearFrameAnalysis: () => void
  seekTo: (time: number) => void
}

// The full Reconstruct pipeline as presented to the operator. Calibration is a
// separate gated stage now, so a full run never shows a "Calibrate pitch" step:
// it validates and loads the immutable artifact under "Prepare inputs".
const FULL_RUN_PHASES: { id: string; label: string }[] = [
  { id: 'preparing', label: 'Prepare inputs' },
  { id: 'detection', label: 'Detect objects' },
  { id: 'tracking', label: 'Build tracks' },
  { id: 'projection', label: 'Reconstruct 3D' },
  { id: 'finalizing', label: 'Save result' },
]
// Backend phase id → the presented phase it advances. A full run has no
// calibration phase; the terminal tick completes the list.
const PRESENTED_PHASE: Record<string, string> = {
  queued: 'preparing',
  preparing: 'preparing',
  detection: 'detection',
  tracking: 'tracking',
  projection: 'projection',
  finalizing: 'finalizing',
  complete: 'finalizing',
}

/** Durable reconstruction-job orchestration. Dense scene data is never polled here. */
export function useReconstructionController(options: ReconstructionControllerOptions) {
  const reconstructing = ref(false)
  const selectedModel = ref<ReconstructionModel>('yolo26m.pt')
  const selectedBallBackend = ref<BallDetectionBackend>('dedicated-ultralytics')
  // Pre-selected, always visible while the manual ball trajectory is
  // authoritative: the most expensive analysis phase is skipped explicitly.
  const skipBallDetection = ref(true)
  const manualBallAuthoritative = computed(() => (
    options.scene.value?.payload.ball?.mode === 'manual'
  ))
  // Off by default: disabling OCR trades automatic shirt-number merge
  // evidence for a cheaper run when the roster is bound manually.
  const skipJerseyOcr = ref(false)
  // Experimental: project RTMPose feet instead of the bbox bottom-centre.
  // Ineligible crops fall back to bbox explicitly (see contactPoint
  // diagnostics); acceptance is visual in the split view.
  const poseContactPoint = ref(false)
  // 0 is the explicit UI representation of "native source cadence". Positive
  // values are operator-selected reduced cadences and are sent with the job.
  const selectedFrameRate = ref(0)
  // Zero is the accuracy-first product default: direct PnLCalib on every
  // selected frame. Positive values are explicit sparse performance modes.
  const selectedDirectCalibrationMaxGapSeconds = ref(0)
  const confirmingReview = ref(false)
  const savingFrame = ref(false)
  const updatingFrameExclusion = ref(false)
  const finalizingCalibrationEdits = ref(false)
  const resettingCalibration = ref(false)
  const frameGenerationRunId = ref<string | null>(null)
  let pollingEpoch = 0
  let terminalSyncRequestId = 0
  const terminalSync = ref<{
    runId: string
    sceneId: string
    status: AnalysisJob['status']
  } | null>(null)

  const sceneVideo = computed(() => options.scene.value?.payload.videoAsset ?? null)
  const sourceFrameRate = computed(() => Number(sceneVideo.value?.fps ?? 0))
  const materializedFrameRate = computed(() => Number(sceneVideo.value?.analysisFps ?? 0))
  const effectiveFrameRate = computed(() => (
    selectedFrameRate.value > 0 ? selectedFrameRate.value : sourceFrameRate.value
  ))
  const frameRateRequiresRegeneration = computed(() => (
    effectiveFrameRate.value > materializedFrameRate.value + 1e-3
  ))
  const calibrationFrameRateCurrent = computed(() => {
    const calibratedFrameRate = Number(
      sceneVideo.value?.reconstruction?.samplingFrameRate ?? 0,
    )
    return calibratedFrameRate > 0
      && Math.abs(calibratedFrameRate - effectiveFrameRate.value) <= 1e-3
  })
  const calibrationDirectSamplingCurrent = computed(() => {
    const persisted = sceneVideo.value?.reconstruction
      ?.directCalibrationMaxGapSeconds
    return persisted != null
      && Math.abs(
        Number(persisted) - selectedDirectCalibrationMaxGapSeconds.value,
      ) <= 1e-6
  })
  const sourceFrameInputReady = computed(() => {
    const input = sceneVideo.value?.analysisFrameInput
    return Boolean(
      input
      && input.schemaVersion === 1
      && input.source === 'uploaded-video'
      && input.coordinateSpace === 'source-video-pixels'
      && input.resize === 'none'
      && input.width > 0
      && input.height > 0,
    )
  })
  const frameGenerationJob = computed<AnalysisJob | null>(() => {
    const runId = frameGenerationRunId.value
    if (!runId) return null
    return options.jobs.value.find((job) => job.id === runId) ?? null
  })
  const regeneratingFrames = computed(() => (
    frameGenerationJob.value?.status === 'queued'
    || frameGenerationJob.value?.status === 'running'
    || frameGenerationJob.value?.status === 'cancelling'
  ))
  const currentJob = computed<AnalysisJob | null>(() => (
    matchingReconstructionAnalysisJob(options.scene.value, options.jobs.value)
  ))
  const status = computed(() => reconstructionStatusFromAnalysisJob(
    sceneVideo.value?.reconstruction?.status,
    currentJob.value?.status,
  ))
  const processingStatus = computed<ProcessingStatus>(() => {
    const jobStatus = currentJob.value?.status
    if (jobStatus === 'queued') return 'queued'
    if (jobStatus === 'running' || jobStatus === 'cancelling') return 'processing'
    if (jobStatus === 'succeeded') return 'completed'
    if (jobStatus === 'cancelled' || jobStatus === 'failed') return jobStatus
    const reconstruction = sceneVideo.value?.reconstruction
    if (reconstruction?.processingStatus) return reconstruction.processingStatus
    if (reconstruction?.status === 'ready') return 'completed'
    return reconstruction?.status ?? 'completed'
  })
  const qualityVerdict = computed<QualityVerdict>(() => (
    sceneVideo.value?.reconstruction?.qualityVerdict
    ?? sceneVideo.value?.reconstruction?.quality?.verdict
    ?? sceneVideo.value?.reconstruction?.qualityReport?.verdict
    ?? 'unknown'
  ))
  // The two-stage gate. `stage` is 'calibration' only after a calibrate run; a
  // full run leaves 'reconstruction' behind. The gate is authoritative only
  // while it describes the current inputs.
  const stage = computed(() => sceneVideo.value?.reconstruction?.stage ?? null)
  const pendingCalibrationEditSession = computed(() => (
    sceneVideo.value?.reconstruction?.pendingCalibrationEditSession ?? null
  ))
  const calibrationReview = computed<CalibrationReview | null>(() => {
    const reconstruction = sceneVideo.value?.reconstruction
    if (!reconstruction) return null
    if (reconstruction.calibrationReview) return reconstruction.calibrationReview
    // Evidence recovery is only for completed legacy/full reconstruction read
    // models. During the calibration stage an absent gate means the latest run
    // did not publish one; rebuilding a review from an older retained artifact
    // would expose a stale Authorize action after a failed recalibration.
    if (reconstruction.stage === 'calibration') return null
    return calibrationReviewFromEvidence(
      reconstruction.calibration?.frameEvidence ?? [],
      {
        inputFingerprint: reconstruction.inputFingerprint ?? null,
        calibrationInputFingerprint: reconstruction.calibrationProvenance
          ?.calibrationInputFingerprint ?? reconstruction.calibrationInputFingerprint ?? null,
        warnings: reconstruction.calibrationWarnings ?? [],
        fallbackConsent: reconstruction.calibrationFallbackConsent ?? null,
      },
    )
  })
  const calibrationReviewCurrent = computed(() => {
    const reviewFingerprint = calibrationReview.value
      ?.calibrationInputFingerprint
    const currentFingerprint = sceneVideo.value?.reconstruction
      ?.calibrationInputFingerprint
    return Boolean(
      reviewFingerprint
      && currentFingerprint
      && reviewFingerprint === currentFingerprint
      && calibrationFrameRateCurrent.value
      && calibrationDirectSamplingCurrent.value,
    )
  })
  const reviewPending = computed(() => (
    stage.value === 'calibration'
    && calibrationReviewCurrent.value
    && calibrationReview.value?.status === 'review'
  ))
  // A full run requires either the current calibration gate or a completed run
  // carrying the same explicit coordinate policy. There is no single-button
  // image-space fallback for an uncalibrated scene.
  const calibrationCleared = computed(() => (
    pendingCalibrationEditSession.value
      ? false
      : stage.value === 'calibration'
      ? (calibrationReview.value?.status === 'ready'
        || calibrationReview.value?.status === 'confirmed')
        && calibrationReviewCurrent.value
        && calibrationFrameRateCurrent.value
        && calibrationDirectSamplingCurrent.value
        && Boolean(sceneVideo.value?.reconstruction?.calibrationProvenance)
      : stage.value === 'reconstruction'
        && sceneVideo.value?.reconstruction?.inputState !== 'stale'
        && calibrationFrameRateCurrent.value
        && calibrationDirectSamplingCurrent.value
        && Boolean(sceneVideo.value?.reconstruction?.trackingCoordinatePolicy)
        && Boolean(sceneVideo.value?.reconstruction?.calibrationProvenance)
  ))
  const progress = computed(() => {
    const persisted = sceneVideo.value?.reconstruction?.progress ?? null
    const job = currentJob.value
    if (!job || !['queued', 'running', 'cancelling'].includes(job.status)) return persisted
    const backendPhase = job.phase ?? persisted?.phase ?? 'queued'
    return {
      phase: backendPhase,
      phaseIndex: persisted?.phaseIndex ?? 1,
      phaseCount: persisted?.phaseCount ?? 5,
      label: job.progress.label || persisted?.label || 'Waiting to start',
      detail: job.progress.detail ?? persisted?.detail ?? null,
      completed: job.progress.completed,
      total: job.progress.total,
      phasePercent: job.progress.percent,
      overallPercent: job.progress.percent,
      elapsedSeconds: persisted?.elapsedSeconds ?? 0,
      etaSeconds: job.progress.etaSeconds,
      updatedAt: persisted?.updatedAt ?? job.startedAt ?? job.createdAt,
      phases: persisted?.phases ?? [],
    }
  })
  const inputState = computed(() => sceneVideo.value?.reconstruction?.inputState ?? null)
  const resultState = computed(() => (
    sceneVideo.value?.reconstruction?.resultState
    ?? (stage.value === 'calibration' ? 'calibration-only' : inputState.value)
  ))
  const running = computed(() => status.value === 'queued' || status.value === 'processing')
  const activeRunId = computed(() => sceneVideo.value?.reconstruction?.runId ?? null)
  const cancelling = computed(() => {
    const runId = activeRunId.value
    return currentJob.value?.status === 'cancelling'
      || Boolean(runId && options.cancelingJobIds.value.includes(runId))
  })
  const canCancel = computed(() => {
    if (!activeRunId.value || !running.value) return false
    return !currentJob.value || isAnalysisJobCancelable(currentJob.value)
  })
  const mutationLocked = computed(() => reconstructionLocksMutations(status.value, reconstructing.value))
  // The checklist advances from the live compact job phase, not the scene's
  // persisted progress (that is frozen at queue time and only refreshes when the
  // run ends), so completed steps tick over in real time during the run.
  const phases = computed<ReconstructionPhase[]>(() => {
    const activePhase = PRESENTED_PHASE[progress.value?.phase ?? 'queued'] ?? 'preparing'
    const finished = (progress.value?.phase ?? '') === 'complete' || status.value === 'ready'
    const activeIndex = FULL_RUN_PHASES.findIndex((phase) => phase.id === activePhase)
    const current = activeIndex < 0 ? 0 : activeIndex
    return FULL_RUN_PHASES.map((phase, index) => ({
      id: phase.id,
      label: phase.label,
      status: finished || index < current
        ? 'completed'
        : index === current
          ? 'current'
          : 'pending',
    }))
  })

  function stop() {
    pollingEpoch += 1
    reconstructing.value = false
  }

  function reset() {
    stop()
    terminalSyncRequestId += 1
    terminalSync.value = null
    frameGenerationRunId.value = null
  }

  async function startPolling(sceneId: string) {
    const expectedEpoch = ++pollingEpoch
    reconstructing.value = true
    await options.refreshJobs()
    if (expectedEpoch !== pollingEpoch || options.scene.value?.id !== sceneId) return
    if (currentJob.value) return
    reconstructing.value = false
    const runId = options.scene.value.payload.videoAsset?.reconstruction?.runId ?? 'unknown'
    options.saveState.value = 'Analysis state is inconsistent'
    options.error.value = `Analysis run ${runId} has no compact job telemetry. Retry the analysis.`
  }

  function resumePolling() {
    const scene = options.scene.value
    if (!scene || !running.value) {
      stop()
      return
    }
    options.saveState.value = progress.value
      ? `${progress.value.label} · ${progress.value.overallPercent}%`
      : 'Analysis running…'
    void startPolling(scene.id)
  }

  async function cancel(runId: string) {
    const activeSceneId = options.scene.value?.id ?? null
    const isActiveSceneRun = activeRunId.value === runId
    if (isActiveSceneRun) options.saveState.value = 'Requesting analysis cancellation…'
    const updated = await options.cancelJob(runId)
    if (!updated || !isActiveSceneRun || options.scene.value?.id !== activeSceneId) return
    if (updated.status === 'cancelling') {
      options.saveState.value = 'Cancelling analysis safely…'
      if (!reconstructing.value && activeSceneId) void startPolling(activeSceneId)
    }
  }

  async function cancelActive() {
    if (activeRunId.value && canCancel.value) await cancel(activeRunId.value)
  }

  async function syncAfterTerminal(job: AnalysisJob) {
    const sceneId = options.scene.value?.id
    if (!sceneId || activeRunId.value !== job.id) return
    const requestId = ++terminalSyncRequestId
    stop()
    options.saveState.value = job.status === 'cancelled'
      ? 'Analysis cancelled · ready to run again'
      : job.status === 'failed'
        ? job.error || 'Analysis failed · ready to retry'
        : 'Compute complete · loading result…'
    try {
      const updated = await sceneClient.get(options.projectId(), sceneId)
      if (
        requestId !== terminalSyncRequestId
        || options.scene.value?.id !== sceneId
        || activeRunId.value !== job.id
      ) return
      options.scene.value = updated
      options.clearFrameAnalysis()
      terminalSync.value = { runId: job.id, sceneId, status: job.status }
    } catch (cause) {
      if (requestId !== terminalSyncRequestId || options.scene.value?.id !== sceneId) return
      options.error.value = cause instanceof Error
        ? cause.message
        : 'Could not refresh the scene after analysis stopped'
    }
  }

  async function reconstruct(mode: ReconstructionMode = 'full') {
    const scene = options.scene.value
    if (!scene?.payload.videoAsset?.selectedSegmentId || reconstructing.value) return
    options.clearFrameAnalysis()
    reconstructing.value = true
    const skipBall = manualBallAuthoritative.value && skipBallDetection.value
    options.saveState.value = mode === 'calibrate'
      ? 'Calibrating the pitch…'
      : skipBall
        ? `Detecting people with ${selectedModel.value.replace('.pt', '')} · manual ball (detection skipped)…`
        : `Detecting people with ${selectedModel.value.replace('.pt', '')} · ball ${selectedBallBackend.value}…`
    try {
      options.scene.value = await reconstructionClient.reconstruct(
        options.projectId(),
        scene.id,
        selectedModel.value,
        selectedBallBackend.value,
        skipBall ? 'skip-manual-authoritative' : 'automatic',
        skipJerseyOcr.value ? 'off' : 'automatic',
        poseContactPoint.value ? 'pose-feet' : 'bbox-bottom',
        mode,
        mode === 'calibrate'
          ? (selectedFrameRate.value > 0 ? selectedFrameRate.value : null)
          : null,
        mode === 'calibrate'
          ? selectedDirectCalibrationMaxGapSeconds.value
          : null,
      )
      await startPolling(scene.id)
    } catch (cause) {
      reconstructing.value = false
      options.error.value = cause instanceof Error ? cause.message : 'Could not start reconstruction'
    }
  }

  async function regenerateAnalysisFrames() {
    const assetId = sceneVideo.value?.id
    if (!assetId || regeneratingFrames.value) return
    options.saveState.value = 'Queueing source-resolution frame extraction…'
    try {
      const queued = await mediaClient.regenerateAnalysisFrames(
        options.projectId(),
        assetId,
      )
      frameGenerationRunId.value = queued.runId
      await options.refreshJobs()
    } catch (cause) {
      options.error.value = cause instanceof Error
        ? cause.message
        : 'Could not regenerate source-resolution analysis frames'
    }
  }

  async function syncAfterFrameGeneration(job: AnalysisJob) {
    const sceneId = options.scene.value?.id
    if (!sceneId || job.id !== frameGenerationRunId.value) return
    if (job.status === 'failed' || job.status === 'cancelled') {
      options.error.value = job.status === 'failed'
        ? job.error || 'Source-resolution frame extraction failed'
        : null
      options.saveState.value = job.status === 'cancelled'
        ? 'Source-resolution frame extraction cancelled'
        : 'Source-resolution frame extraction failed'
      frameGenerationRunId.value = null
      return
    }
    try {
      options.scene.value = await sceneClient.get(options.projectId(), sceneId)
      options.clearFrameAnalysis()
      options.saveState.value = 'Source-resolution frames ready · calibration required'
      frameGenerationRunId.value = null
    } catch (cause) {
      options.error.value = cause instanceof Error
        ? cause.message
        : 'Could not reload the regenerated analysis frames'
    }
  }

  /** Run only the gated calibration process, then stop for review. */
  function calibrate() {
    return reconstruct('calibrate')
  }

  /** Persist one frame correction as a draft. This never queues calibration. */
  async function saveFrameCalibration(
    sceneTime: number,
    preset: PitchCalibrationPreset,
    anchors: PitchCalibrationAnchor[],
    source: CalibrationDraftSource,
    acceptQualityWarning = false,
  ) {
    const scene = options.scene.value
    if (!scene || savingFrame.value) return
    savingFrame.value = true
    try {
      options.scene.value = await calibrationClient.saveDraft(
        options.projectId(),
        scene.id,
        sceneTime,
        preset,
        anchors,
        source,
        acceptQualityWarning,
      )
      options.saveState.value = 'Frame correction staged · finalize when all edits are ready'
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not save the frame'
      throw cause
    } finally {
      savingFrame.value = false
    }
  }

  /** Exclude or restore one immutable source frame for both pipeline stages. */
  async function setFrameExcluded(sourceFrameIndex: number, excluded: boolean) {
    const scene = options.scene.value
    if (!scene || updatingFrameExclusion.value) return
    updatingFrameExclusion.value = true
    try {
      options.scene.value = await sceneClient.setFrameExcluded(
        options.projectId(),
        scene.id,
        sourceFrameIndex,
        excluded,
      )
      options.clearFrameAnalysis()
      options.saveState.value = excluded
        ? `Frame #${sourceFrameIndex} excluded · full recalibration required`
        : `Frame #${sourceFrameIndex} restored · full recalibration required`
    } catch (cause) {
      options.error.value = cause instanceof Error
        ? cause.message
        : 'Could not update the frame exclusion'
      throw cause
    } finally {
      updatingFrameExclusion.value = false
    }
  }

  /** Explicitly recompute only frames that can depend on staged corrections. */
  async function finalizeCalibrationEdits() {
    const scene = options.scene.value
    const edited = pendingCalibrationEditSession.value?.editedSampleIndices ?? []
    if (!scene || !edited.length || finalizingCalibrationEdits.value) return
    finalizingCalibrationEdits.value = true
    options.clearFrameAnalysis()
    options.saveState.value = `Finalizing ${edited.length} staged calibration correction(s)…`
    try {
      options.scene.value = await calibrationClient.finalizeDrafts(
        options.projectId(),
        scene.id,
      )
      await startPolling(scene.id)
    } catch (cause) {
      options.error.value = cause instanceof Error
        ? cause.message
        : 'Could not finalize staged calibration corrections'
      throw cause
    } finally {
      finalizingCalibrationEdits.value = false
    }
  }

  /** Clear the calibration stage (overrides + gate) so it can be redone. */
  async function resetCalibration() {
    const scene = options.scene.value
    if (!scene || resettingCalibration.value) return
    resettingCalibration.value = true
    try {
      options.scene.value = await calibrationClient.resetCalibration(
        options.projectId(),
        scene.id,
      )
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not reset calibration'
    } finally {
      resettingCalibration.value = false
    }
  }

  /**
   * Explicitly authorize image-space fallback on the unresolved calibration
   * frames and unblock the full run. This is a synchronous scene edit (no job).
   */
  async function confirmReview() {
    const scene = options.scene.value
    if (!scene || confirmingReview.value) return
    if (!calibrationReviewCurrent.value) {
      options.error.value = 'This calibration review is stale. Run calibration successfully before authorizing image fallback.'
      return
    }
    confirmingReview.value = true
    try {
      options.scene.value = await reconstructionClient.confirmCalibrationReview(
        options.projectId(),
        scene.id,
      )
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not confirm the calibration review'
    } finally {
      confirmingReview.value = false
    }
  }

  function queueIdentityCorrectionRebuild(sceneId: string, sceneTime: number, label: string) {
    if (options.scene.value?.id !== sceneId) return
    reconstructing.value = true
    options.seekTo(sceneTime)
    options.saveState.value = `${label} saved · rebuilding the affected tracking…`
    void startPolling(sceneId)
  }

  function resetModelSelection() {
    selectedModel.value = sceneVideo.value?.reconstruction?.model ?? 'yolo26m.pt'
    selectedBallBackend.value = sceneVideo.value?.reconstruction?.ballBackend ?? 'dedicated-ultralytics'
    const persistedFrameRate = Number(
      sceneVideo.value?.reconstruction?.samplingFrameRate ?? 0,
    )
    selectedFrameRate.value = (
      persistedFrameRate > 0
      && Math.abs(persistedFrameRate - sourceFrameRate.value) > 1e-3
    ) ? persistedFrameRate : 0
    const persistedDirectGap = sceneVideo.value?.reconstruction
      ?.directCalibrationMaxGapSeconds
    selectedDirectCalibrationMaxGapSeconds.value = persistedDirectGap == null
      ? 0
      : Number(persistedDirectGap)
  }

  return {
    reconstructing,
    selectedModel,
    selectedBallBackend,
    selectedFrameRate,
    selectedDirectCalibrationMaxGapSeconds,
    sourceFrameRate,
    materializedFrameRate,
    effectiveFrameRate,
    frameRateRequiresRegeneration,
    calibrationFrameRateCurrent,
    calibrationDirectSamplingCurrent,
    skipBallDetection,
    manualBallAuthoritative,
    skipJerseyOcr,
    poseContactPoint,
    currentJob,
    sourceFrameInputReady,
    frameGenerationJob,
    regeneratingFrames,
    terminalSync,
    status,
    processingStatus,
    qualityVerdict,
    stage,
    pendingCalibrationEditSession,
    calibrationReview,
    calibrationReviewCurrent,
    reviewPending,
    calibrationCleared,
    confirmingReview,
    progress,
    inputState,
    resultState,
    running,
    activeRunId,
    cancelling,
    canCancel,
    mutationLocked,
    phases,
    stop,
    reset,
    startPolling,
    resumePolling,
    cancel,
    cancelActive,
    syncAfterTerminal,
    projectId: options.projectId,
    reconstruct,
    calibrate,
    regenerateAnalysisFrames,
    syncAfterFrameGeneration,
    saveFrameCalibration,
    finalizeCalibrationEdits,
    resetCalibration,
    savingFrame,
    updatingFrameExclusion,
    setFrameExcluded,
    finalizingCalibrationEdits,
    resettingCalibration,
    confirmReview,
    queueIdentityCorrectionRebuild,
    resetModelSelection,
  }
}
