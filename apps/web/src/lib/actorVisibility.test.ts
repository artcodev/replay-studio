import { describe, expect, it } from 'vitest'
import type { Track } from '../types/tracking'
import { shouldRenderActor, shouldRenderBall, shouldRenderPlayerVisual } from './actorVisibility'

describe('shouldRenderActor', () => {
  it('keeps an actor renderable for a low-confidence inferred keyframe', () => {
    const track: Pick<Track, 'keyframes'> = {
      keyframes: [{
        t: 0,
        x: 12,
        z: -8,
        confidence: 0,
        observed: false,
        presenceState: 'inferred-before-first',
      }],
    }

    expect(shouldRenderActor(track)).toBe(true)
  })

  it('does not render an actor that has no position keyframes', () => {
    expect(shouldRenderActor({ keyframes: [] })).toBe(false)
  })

  it('keeps labels independent from player model visibility', () => {
    const options = { showModels: false, showLabels: true }
    expect(shouldRenderPlayerVisual('player-model', options)).toBe(false)
    expect(shouldRenderPlayerVisual('player-label', options)).toBe(true)
  })

  it('lets the ball toggle override otherwise valid tracking evidence', () => {
    const keyframes = [{ t: 0 }, { t: 2 }]
    expect(shouldRenderBall(true, keyframes, 1, 0.9)).toBe(true)
    expect(shouldRenderBall(false, keyframes, 1, 0.9)).toBe(false)
    expect(shouldRenderBall(true, keyframes, 3, 0.9)).toBe(false)
  })
})
