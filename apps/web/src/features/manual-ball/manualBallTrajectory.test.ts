import { describe, expect, it } from 'vitest'
import {
  manualBallKeyframeAt,
  normalizeManualBallKeyframes,
} from './manualBallTrajectory'

const bounds = { duration: 10, pitchLength: 100, pitchWidth: 60 }

describe('manual ball trajectory domain', () => {
  it('sorts, clamps and replaces samples at the same canonical timestamp', () => {
    const result = normalizeManualBallKeyframes([
      { t: 12, x: 80, y: -2, z: -80, confidence: 0.1 },
      { t: 1.0004, x: 4, y: 2, z: 3, confidence: 0.2 },
      { t: 1, x: 1, y: 1, z: 2, confidence: 0.3 },
    ], bounds)

    expect(result).toHaveLength(2)
    expect(result[0]).toMatchObject({ t: 1, x: 4, y: 2, z: 3, confidence: 1, observed: true })
    expect(result[1]).toMatchObject({ t: 10, x: 50, y: 0.24, z: -30 })
    expect(result.every((frame) => frame.projectionSource === 'manual')).toBe(true)
  })

  it('interpolates between automatic samples before creating a manual keyframe', () => {
    const result = manualBallKeyframeAt([
      { t: 0, x: 0, y: 0.24, z: 0, confidence: 0.8 },
      { t: 10, x: 20, y: 2, z: 10, confidence: 0.8 },
    ], 5, bounds)

    expect(result).toMatchObject({ t: 5, x: 10, y: 1.12, z: 5, confidence: 1 })
  })
})
