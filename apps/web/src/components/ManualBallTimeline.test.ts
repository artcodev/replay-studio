import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import ManualBallTimeline, {
  clampManualBallTime,
  manualBallTimelineEvents,
  normalizeManualBallTimes,
} from './ManualBallTimeline.vue'
import type { Keyframe } from '../types'

function keyframe(t: number): Keyframe {
  return { t, x: t, z: t, confidence: 1, observed: true }
}

describe('ManualBallTimeline', () => {
  it('renders a labeled empty timeline and safe controls with no keypoints', async () => {
    const html = await renderToString(createSSRApp(ManualBallTimeline, {
      duration: 5,
      currentTime: 1.25,
      keyframes: [],
      selectedTime: null,
    }))

    expect(html).toContain('Ball keypoints')
    expect(html).toContain('Manual ball keypoint timeline')
    expect(html).toContain('0 keypoints')
    expect(html).toContain('Add the first keypoint at the current frame.')
    expect(html).toContain('Add keypoint')
    expect(html).toContain('Selected ball keypoint time in seconds')
    expect(html).toContain('Delete selected')
  })

  it('deduplicates and clamps markers while rendering interpolation segments', async () => {
    const html = await renderToString(createSSRApp(ManualBallTimeline, {
      duration: 4,
      currentTime: 1,
      keyframes: [keyframe(-2), keyframe(2), keyframe(2), keyframe(8)],
      selectedTime: 2,
    }))

    expect(normalizeManualBallTimes([-2, 2, 2, 8], 4)).toEqual([0, 2, 4])
    expect(html).toContain('3 keypoints')
    expect(html.match(/aria-label="Ball keypoint/g)).toHaveLength(3)
    expect(html.match(/class="interpolation-segment/g)).toHaveLength(2)
    expect(html).toContain('aria-label="Ball keypoint 1 at 00:00.000"')
    expect(html).toContain('aria-label="Ball keypoint 2 at 00:02.000"')
    expect(html).toContain('aria-label="Ball keypoint 3 at 00:04.000"')
    expect(html).toContain('aria-pressed="true"')
  })

  it('emits select and seek from a marker action', () => {
    expect(manualBallTimelineEvents({
      duration: 4,
      currentTime: 0,
      keyframeTimes: [1, 2],
      selectedTime: null,
    }, { type: 'select', time: 2 })).toEqual([
      { type: 'select', time: 2 },
      { type: 'seek', time: 2 },
    ])
  })

  it('adds a clamped playhead time but selects an existing duplicate', () => {
    expect(clampManualBallTime(8, 4)).toBe(4)
    expect(manualBallTimelineEvents({
      duration: 4,
      currentTime: 8,
      keyframeTimes: [1],
      selectedTime: null,
    }, { type: 'add' })).toEqual([{ type: 'add', time: 4 }])

    expect(manualBallTimelineEvents({
      duration: 4,
      currentTime: 2,
      keyframeTimes: [1, 2],
      selectedTime: null,
    }, { type: 'add' })).toEqual([
      { type: 'select', time: 2 },
      { type: 'seek', time: 2 },
    ])
  })

  it('emits deletion only for a selected keypoint', () => {
    expect(manualBallTimelineEvents({
      duration: 4,
      currentTime: 0,
      keyframeTimes: [1, 2],
      selectedTime: 1,
    }, { type: 'remove' })).toEqual([{ type: 'remove', time: 1 }])

    expect(manualBallTimelineEvents({
      duration: 4,
      currentTime: 0,
      keyframeTimes: [1, 2],
      selectedTime: null,
    }, { type: 'remove' })).toEqual([])
  })

  it('clamps edited time before the next keypoint to prevent duplicates', () => {
    expect(manualBallTimelineEvents({
      duration: 4,
      currentTime: 1,
      keyframeTimes: [1, 2, 3],
      selectedTime: 1,
    }, { type: 'update-time', requestedTime: 2 })).toEqual([
      { type: 'updateTime', value: { from: 1, to: 1.999 } },
    ])
  })
})
