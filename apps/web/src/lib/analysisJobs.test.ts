import { describe, expect, it, vi } from 'vitest'
import type { AnalysisJob } from '../types/project'
import {
  createAnalysisJobsController,
  isAnalysisJobActive,
  isAnalysisJobCancelable,
  isAnalysisJobTerminalTransition,
  orderAnalysisJobs,
  type AnalysisJobsScheduler,
} from './analysisJobs'

function analysisJob(overrides: Partial<AnalysisJob> = {}): AnalysisJob {
  return {
    id: 'run-1',
    projectId: 'project-1',
    segmentId: 'moment-1',
    kind: 'reconstruction',
    status: 'running',
    phase: 'tracking',
    progress: {
      completed: 24,
      total: 100,
      percent: 24,
      label: 'Tracking players',
      detail: 'Associating detections across frames',
      etaSeconds: 18,
    },
    createdAt: '2026-07-17T10:00:00Z',
    startedAt: '2026-07-17T10:00:02Z',
    ...overrides,
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve
    reject = nextReject
  })
  return { promise, resolve, reject }
}

class FakeScheduler implements AnalysisJobsScheduler {
  entries: Array<{ callback: () => void; delayMs: number; cleared: boolean }> = []

  set = (callback: () => void, delayMs: number) => {
    const entry = { callback, delayMs, cleared: false }
    this.entries.push(entry)
    return entry
  }

  clear = (handle: unknown) => {
    const entry = handle as (typeof this.entries)[number]
    entry.cleared = true
  }

  get latest() {
    return this.entries[this.entries.length - 1]
  }
}

describe('analysis job domain helpers', () => {
  it('distinguishes active, cancelable and terminal jobs and orders active work first', () => {
    const queued = analysisJob({ id: 'queued', status: 'queued', createdAt: '2026-07-17T09:00:00Z' })
    const cancelling = analysisJob({ id: 'cancelling', status: 'cancelling' })
    const finished = analysisJob({ id: 'finished', status: 'succeeded', createdAt: '2026-07-17T11:00:00Z' })

    expect(isAnalysisJobActive(queued)).toBe(true)
    expect(isAnalysisJobCancelable(queued)).toBe(true)
    expect(isAnalysisJobActive(cancelling)).toBe(true)
    expect(isAnalysisJobCancelable(cancelling)).toBe(false)
    expect(isAnalysisJobActive(finished)).toBe(false)
    expect(isAnalysisJobCancelable(finished)).toBe(false)
    expect(orderAnalysisJobs([finished, queued, cancelling]).map((job) => job.id)).toEqual([
      'cancelling',
      'queued',
      'finished',
    ])
  })

  it('recognizes only an observed active-to-terminal transition', () => {
    const running = analysisJob({ status: 'running' })
    const succeeded = analysisJob({ status: 'succeeded' })

    expect(isAnalysisJobTerminalTransition(running, succeeded)).toBe(true)
    expect(isAnalysisJobTerminalTransition(null, succeeded)).toBe(false)
    expect(isAnalysisJobTerminalTransition(succeeded, succeeded)).toBe(false)
    expect(isAnalysisJobTerminalTransition(
      running,
      analysisJob({ id: 'different-run', status: 'succeeded' }),
    )).toBe(false)
  })
})

describe('createAnalysisJobsController', () => {
  it('polls the compact project endpoint at active and idle cadences', async () => {
    const scheduler = new FakeScheduler()
    const list = vi.fn()
      .mockResolvedValueOnce([analysisJob()])
      .mockResolvedValueOnce([analysisJob({ status: 'succeeded', progress: {
        completed: 100,
        total: 100,
        percent: 100,
        label: 'Complete',
        detail: null,
        etaSeconds: null,
      } })])
    const controller = createAnalysisJobsController({
      scheduler,
      activePollMs: 700,
      idlePollMs: 5_000,
      transport: { list, cancel: vi.fn() },
    })

    await controller.start('project-1')

    expect(list).toHaveBeenCalledTimes(1)
    expect(list.mock.calls[0][0]).toBe('project-1')
    expect(controller.jobs.value[0].status).toBe('running')
    expect(controller.activeJobs.value).toHaveLength(1)
    expect(scheduler.latest.delayMs).toBe(700)

    scheduler.latest.callback()
    await vi.waitFor(() => expect(list).toHaveBeenCalledTimes(2))

    expect(controller.jobs.value[0].status).toBe('succeeded')
    expect(controller.activeJobs.value).toHaveLength(0)
    expect(scheduler.latest.delayMs).toBe(5_000)
  })

  it('ignores a late response from the previous project even when transport ignores abort', async () => {
    const projectA = deferred<AnalysisJob[]>()
    const projectB = deferred<AnalysisJob[]>()
    const signals: AbortSignal[] = []
    const list = vi.fn((projectId: string, signal: AbortSignal) => {
      signals.push(signal)
      return projectId === 'project-a' ? projectA.promise : projectB.promise
    })
    const controller = createAnalysisJobsController({
      scheduler: new FakeScheduler(),
      transport: { list, cancel: vi.fn() },
    })

    const firstStart = controller.start('project-a')
    const secondStart = controller.start('project-b')
    projectB.resolve([analysisJob({ id: 'run-b', projectId: 'project-b' })])
    await secondStart
    projectA.resolve([analysisJob({ id: 'run-a', projectId: 'project-a' })])
    await firstStart

    expect(signals[0].aborted).toBe(true)
    expect(controller.projectId.value).toBe('project-b')
    expect(controller.jobs.value.map((job) => job.id)).toEqual(['run-b'])
  })

  it('makes callbacks and responses inert after stop', async () => {
    const scheduler = new FakeScheduler()
    const pending = deferred<AnalysisJob[]>()
    const list = vi.fn((_projectId: string, _signal: AbortSignal) => pending.promise)
    const controller = createAnalysisJobsController({
      scheduler,
      transport: { list, cancel: vi.fn() },
    })

    const started = controller.start('project-1')
    const requestSignal = list.mock.calls[0][1] as AbortSignal
    controller.stop()
    pending.resolve([analysisJob()])
    await started

    expect(requestSignal.aborted).toBe(true)
    expect(controller.projectId.value).toBeNull()
    expect(controller.jobs.value).toEqual([])
    expect(controller.loading.value).toBe(false)
  })

  it('applies a server-confirmed cancellation and never calls cancel for terminal work', async () => {
    const scheduler = new FakeScheduler()
    const running = analysisJob()
    const finished = analysisJob({ id: 'run-finished', status: 'succeeded' })
    const cancel = vi.fn().mockResolvedValue({
      ...running,
      status: 'cancelling',
      progress: { ...running.progress, label: 'Cancellation requested' },
    })
    const controller = createAnalysisJobsController({
      scheduler,
      transport: { list: vi.fn().mockResolvedValue([running, finished]), cancel },
    })
    await controller.start('project-1')

    const updated = await controller.cancel(running.id)
    const unchanged = await controller.cancel(finished.id)

    expect(cancel).toHaveBeenCalledOnce()
    expect(updated?.status).toBe('cancelling')
    expect(controller.jobs.value.find((job) => job.id === running.id)?.status).toBe('cancelling')
    expect(unchanged?.status).toBe('succeeded')
    expect(controller.cancelingJobIds.value).toEqual([])
    expect(scheduler.latest.delayMs).toBe(700)
  })

  it('can cancel a just-queued editor run before the next compact list poll sees it', async () => {
    const queued = analysisJob({ id: 'run-new', status: 'queued' })
    const cancel = vi.fn().mockResolvedValue({ ...queued, status: 'cancelled' })
    const controller = createAnalysisJobsController({
      scheduler: new FakeScheduler(),
      transport: { list: vi.fn().mockResolvedValue([]), cancel },
    })
    await controller.start('project-1')

    const updated = await controller.cancel(queued.id)

    expect(cancel).toHaveBeenCalledOnce()
    expect(updated?.status).toBe('cancelled')
    expect(controller.jobs.value).toContainEqual(expect.objectContaining({
      id: queued.id,
      status: 'cancelled',
    }))
  })

  it('does not let a pre-cancel list response restore the old status', async () => {
    const staleList = deferred<AnalysisJob[]>()
    const cancelResponse = deferred<AnalysisJob>()
    const running = analysisJob()
    const list = vi.fn()
      .mockResolvedValueOnce([running])
      .mockImplementationOnce(() => staleList.promise)
    const cancel = vi.fn(() => cancelResponse.promise)
    const scheduler = new FakeScheduler()
    const controller = createAnalysisJobsController({
      scheduler,
      transport: { list, cancel },
    })
    await controller.start('project-1')
    scheduler.latest.callback()
    await vi.waitFor(() => expect(list).toHaveBeenCalledTimes(2))

    const cancelling = { ...running, status: 'cancelled' as const }
    const cancellation = controller.cancel(running.id)
    cancelResponse.resolve(cancelling)
    await cancellation
    staleList.resolve([running])
    await Promise.resolve()

    expect(controller.jobs.value[0].status).toBe('cancelled')
  })

  it('restores polling when the cancellation request fails', async () => {
    const scheduler = new FakeScheduler()
    const running = analysisJob()
    const list = vi.fn().mockResolvedValue([running])
    const controller = createAnalysisJobsController({
      scheduler,
      retryPollMs: 2_500,
      transport: {
        list,
        cancel: vi.fn().mockRejectedValue(new Error('Cancel service unavailable')),
      },
    })
    await controller.start('project-1')

    await controller.cancel(running.id)

    expect(controller.error.value).toBe('Cancel service unavailable')
    expect(controller.cancelingJobIds.value).toEqual([])
    expect(scheduler.latest.delayMs).toBe(2_500)
    scheduler.latest.callback()
    await vi.waitFor(() => expect(list).toHaveBeenCalledTimes(2))
  })

  it('surfaces transport errors and schedules a bounded retry', async () => {
    const scheduler = new FakeScheduler()
    const controller = createAnalysisJobsController({
      scheduler,
      retryPollMs: 2_500,
      transport: {
        list: vi.fn().mockRejectedValue(new Error('Backend unavailable')),
        cancel: vi.fn(),
      },
    })

    await controller.start('project-1')

    expect(controller.error.value).toBe('Backend unavailable')
    expect(controller.loading.value).toBe(false)
    expect(scheduler.latest.delayMs).toBe(2_500)
  })
})
