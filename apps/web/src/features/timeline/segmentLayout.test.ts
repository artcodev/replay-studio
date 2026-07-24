import { describe, expect, it } from 'vitest'
import { canSplitSegmentTail, normalizeSegmentLayout, splitSegmentTail, type SceneVideo } from './segmentLayout'

function video(): SceneVideo {
  return {
    id: 'video-1',
    filename: 'clip.mp4',
    mediaUrl: '/clip.mp4',
    posterUrl: '/poster.jpg',
    fps: 25,
    frameCount: 100,
    processingState: 'ready',
    segments: [
      { id: 'a', label: 'A', start: 0, end: 1, duration: 1, score: 1, layout: { group: 1, variant: 'A', label: '1-A', role: 'original', confidence: 1 } },
      { id: 'b', label: 'B', start: 1, end: 2, duration: 1, score: 1, layout: { group: 1, variant: 'B', label: '1-B', role: 'replay', confidence: 1 } },
      { id: 'c', label: 'C', start: 2, end: 3, duration: 1, score: 1, layout: { group: 1, variant: 'C', label: '1-C', role: 'replay', confidence: 1 } },
      { id: 'd', label: 'D', start: 3, end: 4, duration: 1, score: 1, layout: { group: 2, variant: 'A', label: '2-A', role: 'original', confidence: 1 } },
    ],
    segmentLayout: {
      status: 'proposed',
      method: 'shot-order-fallback',
      confidence: 0.5,
      scoreChangeTimes: [],
      groups: [],
      warnings: [],
    },
  }
}

describe('segment layout edits', () => {
  it('allows splitting any continuous run that leaves a remainder', () => {
    const fixture = video()
    expect(canSplitSegmentTail(fixture, ['b', 'c'])).toBe(true)
    expect(canSplitSegmentTail(fixture, ['a'])).toBe(true)
    expect(canSplitSegmentTail(fixture, ['b'])).toBe(true)
    // The whole event and cross-event picks stay refused.
    expect(canSplitSegmentTail(fixture, ['a', 'b', 'c'])).toBe(false)
    expect(canSplitSegmentTail(fixture, ['a', 'c'])).toBe(false)
    expect(canSplitSegmentTail(fixture, ['c', 'd'])).toBe(false)
  })

  it('creates a new event and shifts later events', () => {
    const fixture = video()
    expect(splitSegmentTail(fixture, ['b', 'c'])).toBe(2)
    expect(fixture.segments?.map((segment) => segment.layout?.label)).toEqual([
      '1-A', '2-A', '2-B', '3-A',
    ])
  })

  it('splits a head run keeping broadcast-time event order', () => {
    const fixture = video()
    expect(splitSegmentTail(fixture, ['a'])).toBe(1)
    expect(fixture.segments?.map((segment) => segment.layout?.label)).toEqual([
      '1-A', '2-A', '2-B', '3-A',
    ])
  })

  it('normalizes the first variant to original', () => {
    const fixture = video()
    if (fixture.segments?.[0]?.layout) fixture.segments[0].layout.role = 'replay'
    normalizeSegmentLayout(fixture)
    expect(fixture.segments?.[0]?.layout?.role).toBe('original')
  })
})
