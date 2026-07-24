import { describe, expect, it } from 'vitest'
import {
  calibrationFramesWithExclusions,
  calibrationReviewFromEvidence,
  calibrationReviewTimelineStatus,
  formatCalibrationThreshold,
  nearestCalibrationFrame,
  normalizeCalibrationGates,
} from './calibrationQaPresentation'

describe('calibration QA presentation', () => {
  it('selects the nearest available evidence frame', () => {
    const frames = [
      { sceneTime: 1, sourceFrameIndex: 1 },
      { sceneTime: 4, sourceFrameIndex: 4 },
    ]
    expect(nearestCalibrationFrame(frames as never, 3.2)?.sourceFrameIndex).toBe(4)
  })

  it('normalizes gate maps and threshold units for the view', () => {
    const quality = {
      gates: {
        reprojection: {
          label: 'Reprojection',
          status: 'warning',
          value: 4.25,
          unit: 'pixels',
          threshold: { passAtMost: 3 },
        },
      },
    }
    expect(normalizeCalibrationGates(quality as never)).toEqual([{
      id: 'reprojection',
      label: 'Reprojection',
      status: 'review',
      value: '4.3 px',
      threshold: 'pass ≤ 3.0 px',
      detail: null,
    }])
    expect(formatCalibrationThreshold({ rejectBelow: 0.75 }, 'ratio')).toBe('reject < 75%')
  })

  it('keeps temporal calibration visually distinct from direct calibration', () => {
    const sample = {
      sampleIndex: 1,
      sourceFrameIndex: 25,
      sceneTime: 1,
      solutionStatus: 'temporal-accepted',
      projectionSource: 'temporal-bidirectional',
      resolved: true,
      residualP95: null,
      rejectionReasons: [],
      acceptedByOperator: false,
      manual: false,
    }

    expect(calibrationReviewTimelineStatus(sample)).toBe('temporal')
    expect(calibrationReviewTimelineStatus({
      ...sample,
      solutionStatus: 'direct-accepted',
      projectionSource: 'direct',
    })).toBe('direct')
    expect(calibrationReviewTimelineStatus({ ...sample, resolved: false })).toBe('unresolved')
  })

  it('keeps excluded source frames browsable but outside calibration status', () => {
    const frames = calibrationFramesWithExclusions([
      {
        sampleIndex: 0,
        sourceFrameIndex: 317,
        sceneTime: 6.47,
        solutionStatus: 'direct-accepted',
        projectionSource: 'direct',
        resolved: true,
        residualP95: null,
        rejectionReasons: [],
        acceptedByOperator: false,
        manual: false,
      },
    ], [
      { sourceFrameIndex: 318, sceneTime: 6.5 },
    ])

    expect(frames).toHaveLength(2)
    expect(frames[1]).toMatchObject({
      sourceFrameIndex: 318,
      sceneTime: 6.5,
      solutionStatus: 'excluded',
      excluded: true,
      resolved: false,
    })
    expect(calibrationReviewTimelineStatus(frames[1])).toBe('excluded')
  })

  it('recovers the review timeline from the immutable evidence of an older full run', () => {
    const review = calibrationReviewFromEvidence([
      {
        sampleIndex: 0,
        sourceFrameIndex: 10,
        sceneTime: 0.4,
        sourceTime: 5.4,
        status: 'accepted',
        solutionStatus: 'direct-accepted',
        source: 'pnlcalib',
        projectionSource: 'direct',
        imageToPitch: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        frameWidth: 1920,
        frameHeight: 1080,
        alignmentMetrics: {
          precision: 1,
          recall: 1,
          f1: 1,
          residualP50: 1.5,
          residualP95: 2.5,
        },
      },
      {
        sampleIndex: 1,
        sourceFrameIndex: 20,
        sceneTime: 0.8,
        sourceTime: 5.8,
        status: 'missing',
        solutionStatus: 'unresolved',
        source: 'none',
        projectionSource: 'none',
        rejectionReasons: ['no-automatic-calibration-candidate'],
      },
    ], {
      calibrationInputFingerprint: 'sha256:calibration',
      fallbackConsent: { sampleIndices: [1], confirmedAt: '2026-07-22T12:00:00Z' },
    })

    expect(review).toMatchObject({
      status: 'confirmed',
      totalFrames: 2,
      resolvedFrames: 1,
      unresolvedFrames: 1,
      fallbackSampleIndices: [1],
    })
    expect(review?.frames[0]).toMatchObject({
      resolved: true,
      residualP95: 2.5,
      frameWidth: 1920,
    })
    expect(review?.frames[1]).toMatchObject({ resolved: false, imageToPitch: null })
  })
})
