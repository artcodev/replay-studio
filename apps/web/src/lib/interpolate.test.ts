import { describe, expect, it } from 'vitest'
import { interpolateKeyframes, upsertKeyframe } from './interpolate'

describe('scene interpolation', () => {
  const frames = [
    { t: 0, x: 0, z: 0, confidence: 1 },
    { t: 2, x: 10, z: -4, confidence: 0.8 },
  ]

  it('interpolates positions and confidence', () => {
    expect(interpolateKeyframes(frames, 1)).toEqual({ t: 1, x: 5, y: 0, z: -2, confidence: 0.9 })
  })

  it('inserts a keyframe in temporal order', () => {
    const result = upsertKeyframe(frames, { t: 1, x: 4, z: -1, confidence: 1 })
    expect(result.map((frame) => frame.t)).toEqual([0, 1, 2])
  })
})

