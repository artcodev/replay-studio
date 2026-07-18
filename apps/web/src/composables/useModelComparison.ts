import { computed, ref, type Ref } from 'vue'
import { sceneClient } from '../lib/api/scenes'
import { reconstructionClient } from '../lib/api/reconstruction'
import type { SceneDocument, SceneVideoAsset } from '../types/scene'
import type { AnalysisJob } from '../types/project'

type ModelComparisonOptions = {
  projectId: () => string
  scene: Ref<SceneDocument | null>
  sceneVideo: Readonly<Ref<SceneVideoAsset | null>>
  jobs: Readonly<Ref<AnalysisJob[]>>
  selectedTrackId: Ref<string | null>
  activeTab: Ref<'binding' | 'qa' | 'events'>
  playing: Ref<boolean>
  sourceVideo: Ref<HTMLVideoElement | null>
  saveState: Ref<string>
  error: Ref<string | null>
  refreshJobs: () => Promise<void>
}

/** Queues detector comparisons and reconciles their terminal report into the scene. */
export function useModelComparison(options: ModelComparisonOptions) {
  const queueing = ref(false)
  const runId = ref<string | null>(null)
  let syncRequestId = 0

  const report = computed(() => options.sceneVideo.value?.reconstruction?.modelComparison ?? null)
  const job = computed<AnalysisJob | null>(() => {
    const id = runId.value
    return id ? options.jobs.value.find((item) => item.id === id) ?? null : null
  })
  const running = computed(() => Boolean(
    job.value && ['queued', 'running', 'cancelling'].includes(job.value.status),
  ))
  const frameCount = computed(() => {
    const video = options.sceneVideo.value
    const reconstructed = video?.reconstruction?.frameCount
    if (reconstructed !== undefined) return reconstructed
    if (video?.selectedSegmentId) {
      const fps = Math.min(video.analysisFps ?? 10, 5)
      return Math.max(1, Math.ceil((options.scene.value?.duration ?? 0) * fps))
    }
    return video?.frameCount ?? 0
  })

  function reset() {
    syncRequestId += 1
    runId.value = null
    queueing.value = false
  }

  async function syncAfterTerminal(terminalJob: AnalysisJob) {
    const sceneId = options.scene.value?.id
    if (!sceneId || runId.value !== terminalJob.id) return
    const requestId = ++syncRequestId
    if (terminalJob.status === 'failed' || terminalJob.status === 'cancelled') {
      options.saveState.value = terminalJob.status === 'cancelled'
        ? 'Detector comparison cancelled'
        : terminalJob.error || 'Detector comparison failed'
      return
    }
    options.saveState.value = 'Detector comparison complete · loading report…'
    try {
      const updated = await sceneClient.get(options.projectId(), sceneId)
      if (
        requestId !== syncRequestId
        || options.scene.value?.id !== sceneId
        || runId.value !== terminalJob.id
      ) return
      options.scene.value = updated
      options.selectedTrackId.value ??= updated.payload.tracks[0]?.id ?? null
      options.activeTab.value = 'binding'
      const comparison = updated.payload.videoAsset?.reconstruction?.modelComparison
      const gain = comparison?.comparison.inPitchObservationGain
      options.saveState.value = gain === undefined
        ? 'Detector comparison complete'
        : `Model comparison ready · in-pitch observation delta ${gain >= 0 ? '+' : ''}${gain}`
    } catch (cause) {
      if (requestId !== syncRequestId || options.scene.value?.id !== sceneId) return
      options.error.value = cause instanceof Error
        ? cause.message
        : 'Could not load the completed detector comparison'
    }
  }

  async function compare() {
    const current = options.scene.value
    if (!current || !options.sceneVideo.value?.selectedSegmentId || queueing.value || running.value) return
    const sceneId = current.id
    options.playing.value = false
    options.sourceVideo.value?.pause()
    queueing.value = true
    options.saveState.value = 'Queueing detector comparison…'
    try {
      const queued = await reconstructionClient.compareModels(options.projectId(), sceneId)
      if (options.scene.value?.id !== sceneId) return
      runId.value = queued.runId
      options.saveState.value = `Detector comparison queued · ${frameCount.value} frames`
      await options.refreshJobs()
      const observed = job.value
      if (observed && ['cancelled', 'succeeded', 'failed'].includes(observed.status)) {
        await syncAfterTerminal(observed)
      }
    } catch (cause) {
      options.error.value = cause instanceof Error
        ? cause.message
        : 'Could not compare recognition models'
    } finally {
      queueing.value = false
    }
  }

  return { report, job, running, frameCount, queueing, runId, compare, syncAfterTerminal, reset }
}
