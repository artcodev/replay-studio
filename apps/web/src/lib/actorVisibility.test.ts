import { describe, expect, it } from 'vitest'
import type { Track } from '../types/tracking'
import { shouldRenderActor, shouldRenderBall, shouldRenderPlayerVisual } from './actorVisibility'

describe('shouldRenderActor', () => {
  it('renders an actor only inside its evidence-supported time window', () => {
    const track: Pick<Track, 'keyframes'> = {
      keyframes: [{
        t: 0.5,
        x: 12,
        z: -8,
        confidence: 0.9,
        observed: true,
        presenceState: 'observed',
      }],
    }

    expect(shouldRenderActor(track, 0)).toBe(false)
    expect(shouldRenderActor(track, 0.46)).toBe(true)
    expect(shouldRenderActor(track, 0.5)).toBe(true)
    expect(shouldRenderActor(track, 0.56)).toBe(false)
  })

  it('does not render an actor that has no position keyframes', () => {
    expect(shouldRenderActor({ keyframes: [] }, 0)).toBe(false)
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
