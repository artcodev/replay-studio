import { describe, expect, it } from 'vitest'
import type { CalibrationFrameEvidence } from '../types/calibration'
import type { TrackObservation } from '../types/tracking'
import {
  buildTrackProjectionDebugSamples,
  nearestProjectionDebugSample,
  projectImagePoint,
} from './trackProjectionDebug'

const pitch = { length: 105, width: 68 }
const calibration = (frame: number, sceneTime: number, scale = 0.1): CalibrationFrameEvidence => ({
  sourceFrameIndex: frame,
  sampleIndex: frame,
  sceneTime,
  sourceTime: sceneTime,
  status: 'accepted',
  solutionStatus: 'direct-accepted',
  source: 'direct',
  projectionSource: 'direct',
  imageToPitch: [[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
})
const observation = (frame: number, sceneTime: number, x: number): TrackObservation => ({
  frameIndex: frame,
  sceneTime,
  bbox: { x, y: 0, width: 10, height: 10 },
  confidence: 0.9,
  metricStatus: 'accepted',
})

describe('track projection debugger', () => {
  it('projects the exact bbox bottom-centre through the frame homography', () => {
    expect(projectImagePoint(
      [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]],
      { x: 15, y: 20 },
      pitch,
    )).toEqual({ x: 1.5, z: 2 })
  })

  it('replays per-frame matrices and exposes speed and identity boundaries', () => {
    const samples = buildTrackProjectionDebugSamples(
      [observation(10, 0, 0), observation(11, 0.04, 5)],
      [calibration(10, 0), calibration(11, 0.04, 0.2)],
      pitch,
      'bbox-bottom',
    )
    expect(samples[0].matrixPitch).toEqual({ x: 0.5, z: 1 })
    expect(samples[1].matrixPitch).toEqual({ x: 2, z: 2 })
    expect(samples[1].speedMetresPerSecond).toBeCloseTo(45.069, 2)
    expect(samples[1].severity).toBe('identity-split')
    expect(samples[1].imageDeltaPixels).toBe(5)
    expect(samples[1].imageMotionPitchDeltaMetres).toBeCloseTo(0.5, 6)
    expect(samples[1].calibrationMotionPitchDeltaMetres).toBeCloseTo(Math.SQRT2, 6)
  })

  it('isolates a frame-matrix jump from bbox movement', () => {
    const samples = buildTrackProjectionDebugSamples(
      [observation(10, 0, 0), observation(11, 0.04, 0)],
      [
        calibration(10, 0, 0.1),
        {
          ...calibration(11, 0.04, 0.1),
          imageToPitch: [[0.1, 0, 2], [0, 0.1, 0], [0, 0, 1]],
        },
      ],
      pitch,
    )

    expect(samples[1].imageDeltaPixels).toBe(0)
    expect(samples[1].imageMotionPitchDeltaMetres).toBe(0)
    expect(samples[1].calibrationMotionPitchDeltaMetres).toBe(2)
    expect(samples[1].pitchDeltaMetres).toBe(2)
  })

  it('separates optical camera motion from a discontinuous calibration matrix', () => {
    const current = {
      ...calibration(11, 0.04, 0.1),
      cameraMotion: {
        status: 'estimated' as const,
        currentToPrevious: [[1, 0, 5], [0, 1, 0], [0, 0, 1]],
      },
    }
    const continuous = buildTrackProjectionDebugSamples(
      [observation(10, 0, 5), observation(11, 0.04, 0)],
      [calibration(10, 0, 0.1), {
        ...current,
        imageToPitch: [[0.1, 0, 0.5], [0, 0.1, 0], [0, 0, 1]],
      }],
      pitch,
    )[1]

    expect(continuous.cameraCompensatedImageDeltaPixels).toBe(0)
    expect(continuous.calibrationContinuityDeltaMetres).toBe(0)

    const discontinuous = buildTrackProjectionDebugSamples(
      [observation(10, 0, 5), observation(11, 0.04, 0)],
      [calibration(10, 0, 0.1), {
        ...current,
        imageToPitch: [[0.1, 0, 2.5], [0, 0.1, 0], [0, 0, 1]],
      }],
      pitch,
    )[1]
    expect(discontinuous.calibrationContinuityDeltaMetres).toBe(2)
  })

  it('uses stored pose projection when bbox-bottom is not authoritative', () => {
    const first = { ...observation(10, 0, 0), pitch: { x: 1, z: 2 } }
    const samples = buildTrackProjectionDebugSamples(
      [first],
      [calibration(10, 0)],
      pitch,
      'pose-feet',
    )
    expect(samples[0].matrixPitch).toBeNull()
    expect(samples[0].effectivePitch).toEqual({ x: 1, z: 2 })
    expect(samples[0].projectionMethod).toBe('stored-observation')
  })

  it('returns the nearest sample but marks distant playhead positions inactive', () => {
    const samples = buildTrackProjectionDebugSamples(
      [observation(10, 1, 0)],
      [calibration(10, 1)],
      pitch,
    )
    expect(nearestProjectionDebugSample(samples, 1.04)?.active).toBe(true)
    expect(nearestProjectionDebugSample(samples, 2)?.active).toBe(false)
  })
})
