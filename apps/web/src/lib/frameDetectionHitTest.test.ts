import { describe, expect, it } from 'vitest'
import {
  orderedFrameDetectionHits,
  selectFrameDetectionHit,
  type FrameDetectionHitTarget,
} from './frameDetectionHitTest'

function detection(
  id: string,
  x: number,
  y: number,
  width: number,
  height: number,
  confidence = 0.5,
): FrameDetectionHitTarget {
  return { id, bbox: { x, y, width, height }, confidence }
}

describe('frame detection hit testing', () => {
  it('expands a distant detection to the requested minimum hit target size', () => {
    const tiny = detection('tiny-player', 99, 99, 2, 4)

    expect(orderedFrameDetectionHits([tiny], { x: 94, y: 100 })).toEqual([])
    expect(orderedFrameDetectionHits([tiny], { x: 94, y: 100 }, {
      minimumTargetSize: 16,
    })).toEqual([tiny])
  })

  it('prioritizes actual containment over an expanded nearby target', () => {
    const actual = detection('actual', 20, 20, 10, 20)
    const expanded = detection('expanded', 30, 20, 2, 2)

    expect(orderedFrameDetectionHits([expanded, actual], { x: 29, y: 25 }, {
      minimumTargetSize: 20,
    }).map((item) => item.id)).toEqual(['actual', 'expanded'])
  })

  it('orders overlapping real boxes by smaller area, then center distance', () => {
    const large = detection('large', 0, 0, 100, 100)
    const close = detection('close', 42, 42, 20, 20)
    const far = detection('far', 30, 30, 20, 20)

    expect(orderedFrameDetectionHits([large, far, close], { x: 50, y: 50 })
      .map((item) => item.id)).toEqual(['close', 'far', 'large'])
  })

  it('uses confidence and id as stable tie breakers', () => {
    const low = detection('low', 10, 10, 20, 20, 0.3)
    const highZ = detection('z-high', 10, 10, 20, 20, 0.9)
    const highA = detection('a-high', 10, 10, 20, 20, 0.9)

    expect(orderedFrameDetectionHits([low, highZ, highA], { x: 20, y: 20 })
      .map((item) => item.id)).toEqual(['a-high', 'z-high', 'low'])
  })

  it('cycles through ranked candidates and wraps to the first one', () => {
    const small = detection('small', 40, 40, 20, 20)
    const large = detection('large', 0, 0, 100, 100)
    const point = { x: 50, y: 50 }

    expect(selectFrameDetectionHit([large, small], point)?.id).toBe('small')
    expect(selectFrameDetectionHit([large, small], point, {
      previousCandidateId: 'small',
    })?.id).toBe('large')
    expect(selectFrameDetectionHit([large, small], point, {
      previousCandidateId: 'large',
    })?.id).toBe('small')
  })

  it('normalizes negative boxes and safely rejects invalid pointer input', () => {
    const reversed = detection('reversed', 20, 20, -10, -10)

    expect(orderedFrameDetectionHits([reversed], { x: 15, y: 15 })).toEqual([reversed])
    expect(orderedFrameDetectionHits([reversed], { x: Number.NaN, y: 15 })).toEqual([])
  })
})
