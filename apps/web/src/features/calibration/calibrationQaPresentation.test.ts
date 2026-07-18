import { describe, expect, it } from 'vitest'
import {
  formatCalibrationThreshold,
  nearestCalibrationFrame,
  normalizeCalibrationGates,
} from './calibrationQaPresentation'

describe('calibration QA presentation', () => {
  it('selects the nearest available evidence frame', () => {
    const frames = [
      { sceneTime: 1, sourceFrameIndex: 1 },
      { sceneTime: 4, sourceFrameIndex: 4 },
    ]
    expect(nearestCalibrationFrame(frames as never, 3.2)?.sourceFrameIndex).toBe(4)
  })

  it('normalizes gate maps and threshold units for the view', () => {
    const quality = {
      gates: {
        reprojection: {
          label: 'Reprojection',
          status: 'warning',
          value: 4.25,
          unit: 'pixels',
          threshold: { passAtMost: 3 },
        },
      },
    }
    expect(normalizeCalibrationGates(quality as never)).toEqual([{
      id: 'reprojection',
      label: 'Reprojection',
      status: 'review',
      value: '4.3 px',
      threshold: 'pass ≤ 3.0 px',
      detail: null,
    }])
    expect(formatCalibrationThreshold({ rejectBelow: 0.75 }, 'ratio')).toBe('reject < 75%')
  })
})
