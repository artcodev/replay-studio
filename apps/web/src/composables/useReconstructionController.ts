import { computed, ref, type Ref } from 'vue'
import { sceneClient } from '../lib/api/scenes'
import { reconstructionClient } from '../lib/api/reconstruction'
import { isAnalysisJobCancelable } from '../lib/analysisJobs'
import {
  matchingReconstructionAnalysisJob,
  reconstructionLocksMutations,
  reconstructionStatusFromAnalysisJob,
} from '../lib/reconstructionUi'
import type { BallDetectionBackend, ProcessingStatus, QualityVerdict, ReconstructionModel, ReconstructionPhase } from '../types/reconstruction'
import type { SceneDocument } from '../types/scene'
import type { AnalysisJob } from '../types/project'

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

/** Durable reconstruction-job orchestration. Dense scene data is never polled here. */
export function useReconstructionController(options: ReconstructionControllerOptions) {
  const reconstructing = ref(false)
  const selectedModel = ref<ReconstructionModel>('yolo26m.pt')
  const selectedBallBackend = ref<BallDetectionBackend>('dedicated-ultralytics')
  let pollingEpoch = 0
  let terminalSyncRequestId = 0
  const terminalSync = ref<{
    runId: string
    sceneId: string
    status: AnalysisJob['status']
  } | null>(null)

  const sceneVideo = computed(() => options.scene.value?.payload.videoAsset ?? null)
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
  const progress = computed(() => {
    const persisted = sceneVideo.value?.reconstruction?.progress ?? null
    const job = currentJob.value
    if (!job || !['queued', 'running', 'cancelling'].includes(job.status)) return persisted
    return {
      phase: job.phase ?? persisted?.phase ?? 'queued',
      phaseIndex: persisted?.phaseIndex ?? 1,
      phaseCount: persisted?.phaseCount ?? 6,
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
  const phases = computed<ReconstructionPhase[]>(() => progress.value?.phases?.length
    ? progress.value.phases
    : [
        { id: 'preparing', label: 'Prepare inputs', status: 'current' },
        { id: 'calibration', label: 'Calibrate pitch', status: 'pending' },
        { id: 'detection', label: 'Detect objects', status: 'pending' },
        { id: 'tracking', label: 'Build tracks', status: 'pending' },
        { id: 'projection', label: 'Reconstruct 3D', status: 'pending' },
        { id: 'finalizing', label: 'Save result', status: 'pending' },
      ])

  function stop() {
    pollingEpoch += 1
    reconstructing.value = false
  }

  function reset() {
    stop()
    terminalSyncRequestId += 1
    terminalSync.value = null
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

  async function reconstruct() {
    const scene = options.scene.value
    if (!scene?.payload.videoAsset?.selectedSegmentId || reconstructing.value) return
    options.clearFrameAnalysis()
    reconstructing.value = true
    options.saveState.value = `Detecting people with ${selectedModel.value.replace('.pt', '')} · ball ${selectedBallBackend.value}…`
    try {
      options.scene.value = await reconstructionClient.reconstruct(
        options.projectId(),
        scene.id,
        selectedModel.value,
        selectedBallBackend.value,
      )
      await startPolling(scene.id)
    } catch (cause) {
      reconstructing.value = false
      options.error.value = cause instanceof Error ? cause.message : 'Could not start reconstruction'
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
  }

  return {
    reconstructing,
    selectedModel,
    selectedBallBackend,
    currentJob,
    terminalSync,
    status,
    processingStatus,
    qualityVerdict,
    progress,
    inputState,
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
    reconstruct,
    queueIdentityCorrectionRebuild,
    resetModelSelection,
  }
}
