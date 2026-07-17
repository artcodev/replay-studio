import { describe, expect, it } from 'vitest'
import { invertHomography, projectPitchMarkings } from './pitchProjection'

describe('pitch calibration evidence projection', () => {
  it('inverts a valid image-to-pitch homography', () => {
    expect(invertHomography([
      [2, 0, 10],
      [0, 4, 20],
      [0, 0, 1],
    ])).toEqual([
      [0.5, 0, -5],
      [0, 0.25, -5],
      [0, 0, 1],
    ])
  })

  it('refuses singular or malformed calibration evidence', () => {
    expect(invertHomography([[1, 0], [0, 1]])).toBeNull()
    expect(invertHomography([[1, 0, 0], [1, 0, 0], [0, 0, 1]])).toBeNull()
    expect(projectPitchMarkings(null, 960, 540)).toEqual([])
  })

  it('projects canonical pitch lines into the video overlay', () => {
    const markings = projectPitchMarkings([
      [105 / 960, 0, -52.5],
      [0, 68 / 540, -34],
      [0, 0, 1],
    ], 960, 540)
    expect(markings.find((marking) => marking.id === 'halfway')?.points.length).toBeGreaterThan(2)
    expect(markings.find((marking) => marking.id === 'center-circle')?.points.length).toBeGreaterThan(8)
    expect(markings.every((marking) => marking.points.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y)))).toBe(true)
  })
})
