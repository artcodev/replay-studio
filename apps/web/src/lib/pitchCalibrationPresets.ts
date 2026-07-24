import { invertHomography, projectHomographyPoint } from './pitchProjection'
import type {
  PitchCalibrationAnchor,
  PitchCalibrationPreset,
} from '../types/calibration'

type PitchPoint = [number, number]

export const PITCH_CALIBRATION_PRESETS: Array<{
  value: PitchCalibrationPreset
  label: string
}> = [
  { value: 'penalty-area-left', label: 'Left penalty area' },
  { value: 'goal-area-left', label: 'Left goal area' },
  { value: 'center-circle', label: 'Center circle' },
  { value: 'goal-area-right', label: 'Right goal area' },
  { value: 'penalty-area-right', label: 'Right penalty area' },
]

type PresetAnchor = {
  id: string
  label: string
  pitch: PitchPoint
  seed: PitchPoint
}

const PRESET_ANCHORS: Record<PitchCalibrationPreset, PresetAnchor[]> = {
  'penalty-area-right': [
    { id: 'front-far', label: 'Penalty front · far', pitch: [36, -20.16], seed: [.24, .36] },
    { id: 'front-near', label: 'Penalty front · near', pitch: [36, 20.16], seed: [.16, .76] },
    { id: 'goal-far', label: 'Goal line · far', pitch: [52.5, -20.16], seed: [.74, .35] },
    { id: 'goal-near', label: 'Goal line · near', pitch: [52.5, 20.16], seed: [.90, .78] },
  ],
  'goal-area-right': [
    { id: 'front-far', label: 'Goal area front · far', pitch: [47, -9.16], seed: [.38, .43] },
    { id: 'front-near', label: 'Goal area front · near', pitch: [47, 9.16], seed: [.31, .69] },
    { id: 'goal-far', label: 'Goal line · far', pitch: [52.5, -9.16], seed: [.74, .42] },
    { id: 'goal-near', label: 'Goal line · near', pitch: [52.5, 9.16], seed: [.86, .71] },
  ],
  'penalty-area-left': [
    { id: 'goal-far', label: 'Goal line · far', pitch: [-52.5, -20.16], seed: [.10, .35] },
    { id: 'goal-near', label: 'Goal line · near', pitch: [-52.5, 20.16], seed: [.26, .78] },
    { id: 'front-far', label: 'Penalty front · far', pitch: [-36, -20.16], seed: [.76, .36] },
    { id: 'front-near', label: 'Penalty front · near', pitch: [-36, 20.16], seed: [.84, .76] },
  ],
  'goal-area-left': [
    { id: 'goal-far', label: 'Goal line · far', pitch: [-52.5, -9.16], seed: [.14, .42] },
    { id: 'goal-near', label: 'Goal line · near', pitch: [-52.5, 9.16], seed: [.26, .71] },
    { id: 'front-far', label: 'Goal area front · far', pitch: [-47, -9.16], seed: [.62, .43] },
    { id: 'front-near', label: 'Goal area front · near', pitch: [-47, 9.16], seed: [.69, .69] },
  ],
  'center-circle': [
    { id: 'circle-left', label: 'Circle · left', pitch: [-9.15, 0], seed: [.34, .57] },
    { id: 'circle-top', label: 'Circle · far', pitch: [0, -9.15], seed: [.50, .40] },
    { id: 'circle-right', label: 'Circle · right', pitch: [9.15, 0], seed: [.66, .57] },
    { id: 'circle-bottom', label: 'Circle · near', pitch: [0, 9.15], seed: [.50, .75] },
  ],
}

export function seedPitchCalibrationAnchors(
  preset: PitchCalibrationPreset,
  width: number,
  height: number,
): PitchCalibrationAnchor[] {
  return PRESET_ANCHORS[preset].map((anchor) => ({
    id: anchor.id,
    label: anchor.label,
    image: {
      x: Math.round(anchor.seed[0] * width * 100) / 100,
      y: Math.round(anchor.seed[1] * height * 100) / 100,
    },
    pitch: { x: anchor.pitch[0], z: anchor.pitch[1] },
    source: 'seed',
  }))
}

/**
 * Change the semantic anchor group without invoking PnLCalib again.
 *
 * The existing frame homography already maps every canonical pitch point into
 * the image. A preset switch therefore only projects four different pitch
 * points. If that projection is mostly outside the image, an explicit seed is
 * returned for manual placement.
 */
export function projectPitchCalibrationPresetAnchors(
  imageToPitch: number[][] | null | undefined,
  preset: PitchCalibrationPreset,
  width: number,
  height: number,
): PitchCalibrationAnchor[] {
  if (!imageToPitch) return seedPitchCalibrationAnchors(preset, width, height)
  const pitchToImage = invertHomography(imageToPitch)
  if (!pitchToImage) return seedPitchCalibrationAnchors(preset, width, height)
  const projected = PRESET_ANCHORS[preset].map((anchor) => {
    const image = projectHomographyPoint(anchor.pitch, pitchToImage)
    return image
      ? {
          id: anchor.id,
          label: anchor.label,
          image: {
            x: Math.round(image.x * 100) / 100,
            y: Math.round(image.y * 100) / 100,
          },
          pitch: { x: anchor.pitch[0], z: anchor.pitch[1] },
          source: 'projected',
        } satisfies PitchCalibrationAnchor
      : null
  })
  const inside = projected.filter((anchor) => (
    anchor
    && anchor.image.x >= -width * .08
    && anchor.image.x <= width * 1.08
    && anchor.image.y >= -height * .08
    && anchor.image.y <= height * 1.08
  )).length
  return inside >= 3 && projected.every(Boolean)
    ? projected as PitchCalibrationAnchor[]
    : seedPitchCalibrationAnchors(preset, width, height)
}
