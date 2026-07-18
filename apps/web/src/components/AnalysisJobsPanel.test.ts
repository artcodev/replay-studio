import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import type { AnalysisJob } from '../types/project'
import AnalysisJobsPanel from './AnalysisJobsPanel.vue'

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
      etaSeconds: 78,
    },
    createdAt: '2026-07-17T10:00:00Z',
    ...overrides,
  }
}

describe('AnalysisJobsPanel', () => {
  it('renders compact phase progress and cancellation only for cancelable work', async () => {
    const jobs = [
      analysisJob(),
      analysisJob({ id: 'run-cancelling', status: 'cancelling', kind: 'multi-pass' }),
      analysisJob({ id: 'run-comparison', kind: 'model-comparison' }),
      analysisJob({
        id: 'run-complete',
        status: 'succeeded',
        segmentId: null,
        progress: {
          completed: 120,
          total: 120,
          percent: 100,
          label: 'Complete',
          detail: null,
          etaSeconds: null,
        },
      }),
      analysisJob({
        id: 'run-failed',
        status: 'failed',
        error: 'Calibration worker stopped',
      }),
    ]
    const html = await renderToString(createSSRApp(AnalysisJobsPanel, {
      jobs,
      cancelingJobIds: ['run-1'],
      lastUpdatedAt: '2026-07-17T10:10:00Z',
    }))

    expect(html).toContain('Analysis jobs')
    expect(html).toContain('Tracking players')
    expect(html).toContain('Associating detections across frames')
    expect(html).toContain('Phase · tracking')
    expect(html).toContain('1m 18s left')
    expect(html).toContain('aria-valuenow="24"')
    expect(html).toContain('Moment moment-1')
    expect(html).toContain('Multi-angle analysis')
    expect(html).toContain('Compare detection models')
    expect(html).toContain('Calibration worker stopped')
    expect(html).toContain('aria-label="Cancel Reconstruct moment"')
    expect(html).toContain('Cancelling…')
    expect((html.match(/class="cancel-job"/g) ?? []).length).toBe(2)
  })

  it('renders distinct loading, empty and retryable error states', async () => {
    const loading = await renderToString(createSSRApp(AnalysisJobsPanel, { loading: true }))
    const empty = await renderToString(createSSRApp(AnalysisJobsPanel))
    const failed = await renderToString(createSSRApp(AnalysisJobsPanel, {
      error: 'Could not reach the API',
    }))

    expect(loading).toContain('Loading analysis jobs…')
    expect(empty).toContain('No analysis jobs yet')
    expect(empty).toContain('without downloading the full scene')
    expect(failed).toContain('role="alert"')
    expect(failed).toContain('Could not reach the API')
    expect(failed).toContain('>Retry</button>')
  })

  it('declares cancel and retry as its complete integration surface', () => {
    const events = (AnalysisJobsPanel as unknown as { emits: string[] }).emits
    expect(events).toEqual(expect.arrayContaining(['cancel', 'retry']))
    expect(events).toHaveLength(2)
  })
})
