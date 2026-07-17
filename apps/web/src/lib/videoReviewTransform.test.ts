import { describe, expect, it } from 'vitest'
import {
  clampVideoReviewTransform,
  clientPointToContainedMedia,
  panVideoReviewTransform,
  zoomVideoReviewTransform,
} from './videoReviewTransform'

describe('video review zoom and pan bounds', () => {
  it('resets translation at the minimum scale', () => {
    expect(clampVideoReviewTransform({ scale: 1, x: 80, y: -40 }, 800, 500)).toEqual({
      scale: 1,
      x: 0,
      y: 0,
    })
  })

  it('keeps the focal point registered while zooming', () => {
    expect(zoomVideoReviewTransform({ scale: 1, x: 0, y: 0 }, 2, 120, -60, 800, 500)).toEqual({
      scale: 2,
      x: -120,
      y: 60,
    })
  })

  it('clamps pointer and keyboard panning to the scaled frame bounds', () => {
    expect(panVideoReviewTransform({ scale: 2, x: 0, y: 0 }, 1000, -1000, 800, 500)).toEqual({
      scale: 2,
      x: 400,
      y: -250,
    })
  })

  it('keeps annotation coordinates registered after the shared layer is zoomed and panned', () => {
    expect(clientPointToContainedMedia(
      500,
      400,
      { left: -300, top: -100, width: 1600, height: 1000 },
      800,
      500,
    )).toEqual({ x: 400, y: 250 })
  })

  it('accounts for contain letterboxing and clamps points outside the image', () => {
    expect(clientPointToContainedMedia(
      500,
      0,
      { left: 0, top: 0, width: 1000, height: 1000 },
      1920,
      1080,
    )).toEqual({ x: 960, y: 0 })
  })
})
