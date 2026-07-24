import { describe, expect, it } from 'vitest'
import { calibrationWorkflowSteps } from './calibrationProgressPresentation'

describe('calibrationWorkflowSteps', () => {
  it('explains full calibration retries as distinct ordered work', () => {
    const steps = calibrationWorkflowSteps({
      phase: 'calibration',
      phaseIndex: 1,
      phaseCount: 1,
      label: 'Recheck shot-wide p95 outliers',
      detail: 'Retry 1/2',
      completed: 2,
      total: 10,
      phasePercent: 20,
      overallPercent: 60,
      elapsedSeconds: 5,
      etaSeconds: 10,
      updatedAt: 'now',
      phases: [],
    }, 'full-request')

    expect(steps.find((step) => step.id === 'p95-retry')?.state).toBe('current')
    expect(steps.find((step) => step.id === 'local-retry')?.state).toBe('done')
    expect(steps.find((step) => step.id === 'temporal')?.state).toBe('upcoming')
  })

  it('shows the smaller incremental finalization workflow', () => {
    const steps = calibrationWorkflowSteps(null, 'manual-draft-finalize')
    expect(steps.map((step) => step.id)).toEqual(['prepare', 'apply', 'affected', 'publish'])
  })
})
