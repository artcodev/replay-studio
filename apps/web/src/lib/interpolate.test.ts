import { describe, expect, it } from 'vitest'
import { interpolateKeyframes, isInferredAt, upsertKeyframe } from './interpolate'

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


describe('latent position detection', () => {
  const frames = [
    { t: 0, x: 0, z: 0, confidence: 0.2, observed: false },
    { t: 1, x: 5, z: 0, confidence: 0.9, observed: true },
    { t: 2, x: 10, z: 0, confidence: 0.9, observed: true },
    { t: 3, x: 15, z: 0, confidence: 0.2, observed: false },
  ]

  it('marks the inferred prefix, suffix, and bridge intervals as latent', () => {
    expect(isInferredAt(frames, -1)).toBe(true)
    expect(isInferredAt(frames, 0.5)).toBe(true)
    expect(isInferredAt(frames, 1.5)).toBe(false)
    expect(isInferredAt(frames, 2.5)).toBe(true)
    expect(isInferredAt(frames, 9)).toBe(true)
  })

  it('treats missing evidence flags as observed and empty tracks as latent', () => {
    expect(isInferredAt([{ t: 0, x: 0, z: 0, confidence: 1 }], 0)).toBe(false)
    expect(isInferredAt([], 0)).toBe(true)
  })
})
