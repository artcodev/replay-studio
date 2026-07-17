import { describe, expect, it } from 'vitest'
import type { Keyframe } from '../types'
import {
  buildPathTrackingSegments,
  interpolatePathTrackingSegments,
  keyframePathEvidence,
  pathTrackingOptionsForSubject,
  pathTrackingPoints,
} from './pathTracking'

const frame = (t: number, evidence?: 'observed' | 'inferred'): Keyframe => ({
  t,
  x: t,
  z: t * 2,
  confidence: evidence === 'inferred' ? 0.2 : 0.9,
  ...(evidence ? {
    observed: evidence === 'observed',
    state: evidence,
    presenceState: evidence === 'observed' ? 'observed' : 'inferred-gap',
  } : {}),
})

describe('pathTracking', () => {
  it('classifies detector/manual samples and latent samples explicitly', () => {
    expect(keyframePathEvidence(frame(0, 'observed'))).toBe('observed')
    expect(keyframePathEvidence(frame(1, 'inferred'))).toBe('inferred')
    expect(keyframePathEvidence({
      ...frame(2),
      observed: false,
      state: 'occluded',
    })).toBe('inferred')
  })

  it('fails conflicting evidence metadata closed as inferred', () => {
    expect(keyframePathEvidence({
      ...frame(1, 'observed'),
      state: 'inferred',
    })).toBe('inferred')
  })

  it('downgrades geometrically uncertain samples to inferred evidence', () => {
    expect(keyframePathEvidence({
      ...frame(1, 'observed'),
      positionUncertaintyMetres: 4.5,
    })).toBe('inferred')
  })

  it('keeps legacy keyframes visible as observed evidence', () => {
    expect(keyframePathEvidence(frame(0))).toBe('observed')
  })

  it('groups adjacent edges while sharing evidence-change boundaries', () => {
    const segments = buildPathTrackingSegments([
      frame(0, 'observed'),
      frame(1, 'observed'),
      frame(2, 'inferred'),
      frame(3, 'inferred'),
      frame(4, 'observed'),
      frame(5, 'observed'),
    ])

    expect(segments.map((segment) => segment.evidence)).toEqual([
      'observed',
      'inferred',
      'observed',
    ])
    expect(segments.map((segment) => segment.points.map((point) => point.t))).toEqual([
      [0, 1],
      [1, 2, 3, 4],
      [4, 5],
    ])
  })

  it('treats a timestamped non-finite sample as a path barrier', () => {
    const keyframes = [
      frame(2, 'observed'),
      { ...frame(1, 'observed'), x: Number.NaN },
      frame(0, 'observed'),
    ]
    const segments = buildPathTrackingSegments([
      ...keyframes,
    ])

    expect(segments).toEqual([])
    expect(pathTrackingPoints(keyframes).map((point) => point.t)).toEqual([0, 2])
  })

  it('uses the latest sample at a duplicate timestamp without zero-length edges', () => {
    const corrected = { ...frame(1, 'observed'), x: 42 }
    const segments = buildPathTrackingSegments([
      frame(0, 'observed'),
      frame(1, 'inferred'),
      corrected,
      frame(2, 'observed'),
    ])

    expect(segments).toHaveLength(1)
    expect(segments[0].evidence).toBe('observed')
    expect(segments[0].points.map((point) => point.t)).toEqual([0, 1, 2])
    expect(segments[0].points[1].x).toBe(42)
  })

  it('exposes the normalized samples used by renderer and status UI', () => {
    const points = pathTrackingPoints([
      frame(1, 'inferred'),
      { ...frame(1, 'observed'), x: 9 },
      { ...frame(2), z: Number.POSITIVE_INFINITY },
      frame(0, 'observed'),
    ])

    expect(points.map((point) => [point.t, point.x, point.evidence])).toEqual([
      [0, 0, 'observed'],
      [1, 9, 'observed'],
    ])
  })

  it('does not invent a path from fewer than two valid samples', () => {
    expect(buildPathTrackingSegments([])).toEqual([])
    expect(buildPathTrackingSegments([frame(0, 'observed')])).toEqual([])
  })

  it('breaks an impossible player teleport instead of drawing a solid bridge', () => {
    const segments = buildPathTrackingSegments([
      { ...frame(0, 'observed'), x: 0, z: 0 },
      { ...frame(1, 'observed'), x: 30, z: 0 },
      { ...frame(2, 'observed'), x: 31, z: 0 },
    ], pathTrackingOptionsForSubject('player'))

    expect(segments).toHaveLength(1)
    expect(segments[0].points.map((point) => point.t)).toEqual([1, 2])
    expect(interpolatePathTrackingSegments(segments, 0.5)).toBeNull()
    expect(interpolatePathTrackingSegments(segments, 1.5)?.x).toBeCloseTo(30.5)
  })

  it('marks a sparse observed-to-observed edge as inferred', () => {
    const segments = buildPathTrackingSegments([
      { ...frame(0, 'observed'), x: 0, z: 0 },
      { ...frame(2, 'observed'), x: 2, z: 0 },
    ], pathTrackingOptionsForSubject('player'))

    expect(segments).toHaveLength(1)
    expect(segments[0].evidence).toBe('inferred')
  })
})
