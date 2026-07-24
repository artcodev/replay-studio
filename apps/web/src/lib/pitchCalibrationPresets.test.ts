import { describe, expect, it } from 'vitest'
import {
  projectPitchCalibrationPresetAnchors,
  seedPitchCalibrationAnchors,
} from './pitchCalibrationPresets'

describe('pitch calibration preset anchors', () => {
  it('switches penalty-area anchors to the distinct goal-area geometry', () => {
    const imageToPitch = [
      [105 / 960, 0, -52.5],
      [0, 68 / 540, -34],
      [0, 0, 1],
    ]
    const penalty = projectPitchCalibrationPresetAnchors(
      imageToPitch,
      'penalty-area-right',
      960,
      540,
    )
    const goal = projectPitchCalibrationPresetAnchors(
      imageToPitch,
      'goal-area-right',
      960,
      540,
    )

    expect(penalty.map((anchor) => anchor.pitch)).not.toEqual(
      goal.map((anchor) => anchor.pitch),
    )
    expect(penalty[0].image.x).not.toBe(goal[0].image.x)
    expect(goal.every((anchor) => anchor.source === 'projected')).toBe(true)
  })

  it('provides an explicit manual seed when no homography is available', () => {
    const anchors = seedPitchCalibrationAnchors('center-circle', 1920, 1080)

    expect(anchors).toHaveLength(4)
    expect(anchors.every((anchor) => anchor.source === 'seed')).toBe(true)
    expect(anchors[0].image).toEqual({ x: 652.8, y: 615.6 })
  })
})
