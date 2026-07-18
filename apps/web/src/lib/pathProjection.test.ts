import { describe, expect, it } from 'vitest'
import type { CalibrationFrameEvidence } from '../types/calibration'
import {
  clipPathEdgeToFrame,
  interpolatePathProjectionHomography,
  normalisePathProjectionHomography,
  projectPitchEdgeInContext,
  projectPitchPointInContext,
  resolvePathProjectionContext,
} from './pathProjection'

function frame(
  sceneTime: number,
  options: Partial<CalibrationFrameEvidence> = {},
): CalibrationFrameEvidence {
  return {
    sourceFrameIndex: Math.round(sceneTime * 10),
    sampleIndex: Math.round(sceneTime * 10),
    sceneTime,
    sourceTime: sceneTime,
    status: 'accepted',
    source: 'test',
    projectionSource: 'direct',
    frameWidth: 100,
    frameHeight: 60,
    imageToPitch: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ...options,
  }
}

describe('video path projection', () => {
  it('uses an exact QA-accepted frame and fails closed on an exact rejected frame', () => {
    const accepted = resolvePathProjectionContext([frame(1)], 1)
    expect(accepted?.mode).toBe('exact')
    expect(projectPitchPointInContext(accepted!, { x: 12, z: 24 })).toEqual({ x: 12, y: 24 })

    expect(resolvePathProjectionContext([
      frame(1, { status: 'rejected' }),
      frame(1.1),
    ], 1)).toBeNull()
  })

  it('interpolates image points only across adjacent accepted frames with reliable motion', () => {
    const context = resolvePathProjectionContext([
      frame(0),
      frame(0.2, {
        imageToPitch: [[1, 0, -20], [0, 1, 0], [0, 0, 1]],
        cameraMotion: { status: 'estimated' },
      }),
    ], 0.1)

    expect(context?.mode).toBe('interpolated')
    expect(context?.alpha).toBeCloseTo(0.5)
    expect(projectPitchPointInContext(context!, { x: 10, z: 20 })).toEqual({ x: 20, y: 20 })
  })

  it('matches the server by interpolating normalized homographies before inversion', () => {
    const context = resolvePathProjectionContext([
      frame(0),
      frame(0.2, {
        imageToPitch: [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        cameraMotion: { status: 'estimated', confidence: 1 },
      }),
    ], 0.1)

    expect(context?.mode).toBe('interpolated')
    const projected = projectPitchPointInContext(context!, { x: 10, z: 15 })
    expect(projected?.x).toBeCloseTo(10 / 1.5)
    expect(projected?.y).toBeCloseTo(10)
    expect(interpolatePathProjectionHomography(
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
      0.5,
      100,
      60,
    )).not.toBeNull()
  })

  it('does not bridge rejected samples, camera cuts, unreliable motion, or oversized gaps', () => {
    expect(resolvePathProjectionContext([
      frame(0),
      frame(0.1, { status: 'rejected' }),
      frame(0.2, { cameraMotion: { status: 'estimated' } }),
    ], 0.15)).toBeNull()
    expect(resolvePathProjectionContext([
      frame(0),
      frame(0.2, { cameraMotion: { status: 'cut' } }),
    ], 0.1)).toBeNull()
    expect(resolvePathProjectionContext([
      frame(0),
      frame(0.2, { cameraMotion: { status: 'unreliable' } }),
    ], 0.1)).toBeNull()
    expect(resolvePathProjectionContext([
      frame(0),
      frame(1, { cameraMotion: { status: 'estimated' } }),
    ], 0.5)).toBeNull()
  })

  it('allows a bounded nearest endpoint but rejects a stale camera', () => {
    expect(resolvePathProjectionContext([frame(1)], 0.8)?.mode).toBe('nearest')
    expect(resolvePathProjectionContext([frame(1)], 0.7)).toBeNull()
  })

  it('clips path edges to the visible video frame and retains crossing edges', () => {
    expect(clipPathEdgeToFrame({ x: -20, y: 30 }, { x: 120, y: 30 }, 100, 60)).toEqual({
      start: { x: 0, y: 30 },
      end: { x: 100, y: 30 },
    })
    expect(clipPathEdgeToFrame({ x: -20, y: -10 }, { x: -5, y: -2 }, 100, 60)).toBeNull()

    const context = resolvePathProjectionContext([frame(0)], 0)!
    expect(projectPitchEdgeInContext(context, { x: -20, z: 30 }, { x: 120, z: 30 })).toEqual({
      start: { x: 0, y: 30 },
      end: { x: 100, y: 30 },
    })
  })

  it('rejects invalid homographies and mismatched interpolation frame sizes', () => {
    expect(resolvePathProjectionContext([
      frame(0, { imageToPitch: [[1, 0], [0, 1]] }),
    ], 0)).toBeNull()
    expect(resolvePathProjectionContext([
      frame(0),
      frame(0.2, {
        frameWidth: 200,
        cameraMotion: { status: 'estimated' },
      }),
    ], 0.1)).toBeNull()
    expect(normalisePathProjectionHomography(
      [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
      100,
      60,
    )).toBeNull()
  })
})
