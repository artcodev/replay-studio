import {
  computed,
  getCurrentScope,
  onScopeDispose,
  ref,
  shallowRef,
} from 'vue'
import type { AnalysisJob } from '../types/project'
import { projectClient } from './api/projects'

export type AnalysisJobsTransport = {
  list: (projectId: string, signal: AbortSignal) => Promise<AnalysisJob[]>
  cancel: (projectId: string, runId: string, signal: AbortSignal) => Promise<AnalysisJob>
}

export type AnalysisJobsScheduler = {
  set: (callback: () => void, delayMs: number) => unknown
  clear: (handle: unknown) => void
}

export type AnalysisJobsControllerOptions = {
  transport?: AnalysisJobsTransport
  scheduler?: AnalysisJobsScheduler
  activePollMs?: number
  idlePollMs?: number
  retryPollMs?: number
}

const ACTIVE_STATUSES = new Set<AnalysisJob['status']>([
  'queued',
  'running',
  'cancelling',
])

const defaultTransport: AnalysisJobsTransport = {
  list: (projectId, signal) => projectClient.analysisRuns(projectId, signal),
  cancel: (projectId, runId, signal) => projectClient.cancelAnalysisRun(projectId, runId, signal),
}

const defaultScheduler: AnalysisJobsScheduler = {
  set: (callback, delayMs) => window.setTimeout(callback, delayMs),
  clear: (handle) => window.clearTimeout(handle as number),
}

function errorMessage(cause: unknown) {
  return cause instanceof Error ? cause.message : 'Could not load analysis jobs.'
}

function isAbort(cause: unknown) {
  return Boolean(cause && typeof cause === 'object' && 'name' in cause && cause.name === 'AbortError')
}

export function isAnalysisJobActive(job: AnalysisJob) {
  return ACTIVE_STATUSES.has(job.status)
}

export function isAnalysisJobCancelable(job: AnalysisJob) {
  return job.status === 'queued' || job.status === 'running'
}

export function isAnalysisJobTerminalTransition(
  previous: AnalysisJob | null | undefined,
  current: AnalysisJob | null | undefined,
) {
  return Boolean(
    previous
    && current
    && previous.id === current.id
    && ACTIVE_STATUSES.has(previous.status)
    && ['cancelled', 'succeeded', 'failed'].includes(current.status),
  )
}

export function orderAnalysisJobs(jobs: AnalysisJob[]) {
  return [...jobs].sort((left, right) => {
    const activeDelta = Number(isAnalysisJobActive(right)) - Number(isAnalysisJobActive(left))
    if (activeDelta) return activeDelta
    return Date.parse(right.createdAt) - Date.parse(left.createdAt)
  })
}

/**
 * Owns exactly one polling epoch. A project switch, refresh or stop aborts the
 * previous request and makes every late response observationally inert.
 */
export function createAnalysisJobsController(options: AnalysisJobsControllerOptions = {}) {
  const transport = options.transport ?? defaultTransport
  const scheduler = options.scheduler ?? defaultScheduler
  const activePollMs = options.activePollMs ?? 700
  const idlePollMs = options.idlePollMs ?? 5_000
  const retryPollMs = options.retryPollMs ?? 3_000

  const jobs = shallowRef<AnalysisJob[]>([])
  const projectId = ref<string | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  const lastUpdatedAt = ref<string | null>(null)
  const cancelingJobIds = ref<string[]>([])
  const activeJobs = computed(() => jobs.value.filter(isAnalysisJobActive))

  let epoch = 0
  let listRequestVersion = 0
  let scheduled: unknown = null
  let listRequest: AbortController | null = null
  const cancelRequests = new Map<string, AbortController>()

  function clearScheduled() {
    if (scheduled === null) return
    scheduler.clear(scheduled)
    scheduled = null
  }

  function abortRequests() {
    listRequestVersion += 1
    listRequest?.abort()
    listRequest = null
    cancelRequests.forEach((request) => request.abort())
    cancelRequests.clear()
  }

  function nextEpoch() {
    epoch += 1
    clearScheduled()
    abortRequests()
    return epoch
  }

  function schedule(expectedEpoch: number, delayMs: number) {
    if (expectedEpoch !== epoch || !projectId.value) return
    clearScheduled()
    scheduled = scheduler.set(() => {
      scheduled = null
      void poll(expectedEpoch)
    }, delayMs)
  }

  async function poll(expectedEpoch: number) {
    const requestedProjectId = projectId.value
    if (!requestedProjectId || expectedEpoch !== epoch) return

    listRequest?.abort()
    const request = new AbortController()
    const requestVersion = ++listRequestVersion
    listRequest = request
    loading.value = true
    try {
      const response = await transport.list(requestedProjectId, request.signal)
      if (
        expectedEpoch !== epoch
        || requestVersion !== listRequestVersion
        || projectId.value !== requestedProjectId
      ) return
      jobs.value = orderAnalysisJobs(response)
      error.value = null
      lastUpdatedAt.value = new Date().toISOString()
      schedule(expectedEpoch, response.some(isAnalysisJobActive) ? activePollMs : idlePollMs)
    } catch (cause) {
      if (
        expectedEpoch !== epoch
        || requestVersion !== listRequestVersion
        || request.signal.aborted
        || isAbort(cause)
      ) return
      error.value = errorMessage(cause)
      schedule(expectedEpoch, retryPollMs)
    } finally {
      if (listRequest === request) listRequest = null
      if (expectedEpoch === epoch && requestVersion === listRequestVersion) loading.value = false
    }
  }

  async function start(nextProjectId: string) {
    const expectedEpoch = nextEpoch()
    projectId.value = nextProjectId
    jobs.value = []
    cancelingJobIds.value = []
    error.value = null
    lastUpdatedAt.value = null
    await poll(expectedEpoch)
  }

  async function refresh() {
    if (!projectId.value) return
    const expectedEpoch = nextEpoch()
    await poll(expectedEpoch)
  }

  function stop() {
    nextEpoch()
    projectId.value = null
    jobs.value = []
    cancelingJobIds.value = []
    loading.value = false
    error.value = null
    lastUpdatedAt.value = null
  }

  function replaceJob(updated: AnalysisJob) {
    const index = jobs.value.findIndex((job) => job.id === updated.id)
    jobs.value = orderAnalysisJobs(index === -1
      ? [...jobs.value, updated]
      : jobs.value.map((job) => job.id === updated.id ? updated : job))
  }

  async function cancel(runId: string) {
    const current = jobs.value.find((job) => job.id === runId)
    // The editor can learn about a just-queued run from its SceneDocument
    // before the compact project poll observes it. Allow that authoritative
    // run id through; the server still validates active/terminal state.
    if ((current && !isAnalysisJobCancelable(current)) || cancelRequests.has(runId)) return current ?? null
    const expectedEpoch = epoch
    const requestedProjectId = projectId.value
    if (!requestedProjectId) return null

    const request = new AbortController()
    // A list request that started before this mutation must not restore the
    // pre-cancel status after the cancellation response wins the race.
    listRequestVersion += 1
    listRequest?.abort()
    listRequest = null
    clearScheduled()
    loading.value = false
    cancelRequests.set(runId, request)
    cancelingJobIds.value = [...cancelingJobIds.value, runId]
    try {
      const updated = await transport.cancel(requestedProjectId, runId, request.signal)
      if (expectedEpoch !== epoch || projectId.value !== requestedProjectId) return null
      replaceJob(updated)
      error.value = null
      schedule(expectedEpoch, updated.status === 'cancelled' ? idlePollMs : activePollMs)
      return updated
    } catch (cause) {
      if (expectedEpoch !== epoch || request.signal.aborted || isAbort(cause)) return null
      error.value = cause instanceof Error ? cause.message : 'Could not cancel the analysis job.'
      // Cancelling temporarily owns (and aborts) the list request. Restore a
      // bounded poll after a transport failure so one failed Cancel click
      // cannot freeze the controller on its last known running state.
      schedule(expectedEpoch, retryPollMs)
      return null
    } finally {
      if (cancelRequests.get(runId) === request) cancelRequests.delete(runId)
      if (expectedEpoch === epoch) {
        cancelingJobIds.value = cancelingJobIds.value.filter((id) => id !== runId)
      }
    }
  }

  return {
    jobs,
    projectId,
    loading,
    error,
    lastUpdatedAt,
    cancelingJobIds,
    activeJobs,
    start,
    refresh,
    stop,
    cancel,
  }
}

export function useAnalysisJobs(options: AnalysisJobsControllerOptions = {}) {
  const controller = createAnalysisJobsController(options)
  if (getCurrentScope()) onScopeDispose(controller.stop)
  return controller
}
