import { describe, expect, it } from 'vitest'
import {
  calibrationEvidenceAtTime,
  calibrationFrameDiagnostics,
  calibrationLineResidualLabel,
  calibrationPreviewWarnings,
  calibrationRejectionReasonLabel,
} from './calibrationDiagnostics'
import type { CalibrationFrameEvidence, PitchCalibrationDraft } from '../types'

const draft: PitchCalibrationDraft = {
  sceneId: 'shot-01',
  sceneTime: 0,
  frameIndex: 1,
  frameWidth: 960,
  frameHeight: 540,
  source: 'frame-evidence',
  status: 'accepted',
  detectedKeypoints: [],
  detectedKeypointCount: 9,
  inlierCount: 8,
  inlierRatio: 8 / 9,
  rejectionReasons: [],
  visiblePitchSide: 'right',
  preset: 'penalty-area-right',
  confidence: 0.81,
  alignmentError: 3.2,
  alignmentMetrics: {
    precision: 0.8,
    recall: 0.7,
    f1: 0.747,
    residualP50: 3.2,
    residualP95: 6.4,
  },
  quality: 'good',
  anchors: [],
  markings: [],
  imageToPitch: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  warnings: [],
}

const evidence: CalibrationFrameEvidence[] = [{
  sourceFrameIndex: 1,
  sampleIndex: 0,
  sceneTime: 0.01,
  sourceTime: 0.01,
  status: 'rejected',
  source: 'field-keypoints',
  backend: 'field-keypoints',
  projectionSource: 'direct',
  keypointCount: 7,
  inlierCount: 5,
  inlierRatio: 5 / 7,
  rejectionReasons: ['semantic-line-alignment-poor'],
  keypoints: [{
    image: { x: 100, y: 200 },
    projectedImage: { x: 103, y: 198 },
    inlier: true,
  }],
}]

describe('frame calibration diagnostics', () => {
  it('uses only evidence sampled at the requested frame', () => {
    expect(calibrationEvidenceAtTime(evidence, 0)?.sourceFrameIndex).toBe(1)
    expect(calibrationEvidenceAtTime(evidence, 0.5)).toBeNull()
  })

  it('combines preview metrics with source-frame evidence', () => {
    expect(calibrationFrameDiagnostics(draft, evidence)).toMatchObject({
      status: 'accepted',
      sourceStatus: 'rejected',
      method: 'field-keypoints',
      keypointCount: 9,
      inlierCount: 8,
      inlierRatio: 8 / 9,
      residualP50: 3.2,
      residualP95: 6.4,
      points: [],
      lines: [],
      visibleSide: 'right',
      visibleSideTrusted: true,
      rejectionReasons: [],
    })
  })

  it('exposes optional semantic source lines for the overlay', () => {
    const result = calibrationFrameDiagnostics({
      ...draft,
      rawLines: [{
        label: 'Big rect · front edge',
        start: { x: 120, y: 210 },
        end: { x: 410, y: 245 },
        confidence: 0.92,
        residualP50: 1.4,
        residualP95: 3.25,
        residualStatus: 'scored',
        accepted: true,
      }],
    }, [])

    expect(result.lines).toEqual([expect.objectContaining({
      label: 'Big rect · front edge',
      accepted: true,
      residualP95: 3.25,
    })])
  })

  it('formats semantic-line residuals and identifies unscored 3D goal-frame lines', () => {
    expect(calibrationLineResidualLabel({
      name: 'Big rect left main',
      residualStatus: 'scored',
      residualP95: 2.76,
    })).toBe('p95 2.8px')
    expect(calibrationLineResidualLabel({
      name: 'Goal left crossbar',
      groundPlane: false,
      residualStatus: 'not-scored-3d',
    })).toBe('3D')
  })

  it('marks the visible side of a rejected candidate as untrusted', () => {
    const result = calibrationFrameDiagnostics({
      ...draft,
      status: 'rejected',
      quality: 'poor',
      rejectionReasons: ['semantic-line-alignment-poor'],
    }, [])

    expect(result.visibleSide).toBe('right')
    expect(result.visibleSideTrusted).toBe(false)
    expect(calibrationRejectionReasonLabel(result.rejectionReasons[0])).toBe(
      'Projected markings do not align with the observed pitch lines.',
    )
  })

  it('does not reuse an old evidence side for an unsaved manual preview', () => {
    const result = calibrationFrameDiagnostics({
      ...draft,
      source: 'manual',
      status: undefined,
      visiblePitchSide: undefined,
    }, evidence)

    expect(result.visibleSide).toBe('unknown')
    expect(result.visibleSideTrusted).toBe(false)
  })

  it('labels a candidate cap separately from a time deadline', () => {
    const warnings = calibrationPreviewWarnings({
      ...draft,
      warnings: ['The bounded line/curve fallback reached its candidate search limit before the deadline; its best-so-far result was retained when available.'],
      evidence: {
        backendDiagnostics: {
          budgetExhausted: true,
          candidateLimitReached: true,
          deadlineExceeded: false,
          elapsedSeconds: 1.3,
          quadCandidateLimit: 240,
          quadCandidatesGenerated: 2880,
          quadCandidatesEvaluated: 240,
        },
      },
    })

    expect(warnings).toHaveLength(1)
    expect(warnings[0]).toContain('candidate limit')
    expect(warnings[0]).toContain('not a time deadline')
  })

  it('reports a real elapsed-time deadline without calling it a candidate cap', () => {
    const warnings = calibrationPreviewWarnings({
      ...draft,
      warnings: ['The bounded line/curve fallback reached its five-second deadline; its best-so-far result was retained when available.'],
      evidence: {
        backendDiagnostics: {
          budgetExhausted: true,
          candidateLimitReached: false,
          deadlineExceeded: true,
          elapsedSeconds: 5.01,
          quadCandidateLimit: 240,
          quadCandidatesGenerated: 120,
          quadCandidatesEvaluated: 18,
        },
      },
    })

    expect(warnings).toHaveLength(1)
    expect(warnings[0]).toContain('time deadline after 5.0s')
    expect(warnings[0]).not.toContain('candidate cap')
  })

  it('keeps simultaneous candidate and deadline signals distinct', () => {
    const warnings = calibrationPreviewWarnings({
      ...draft,
      warnings: [
        'The bounded line/curve fallback reached its five-second deadline.',
        'The bounded line/curve fallback reached its candidate search limit before the deadline.',
      ],
      evidence: {
        backendDiagnostics: {
          budgetExhausted: true,
          candidateLimitReached: true,
          deadlineExceeded: true,
          elapsedSeconds: 5.02,
        },
      },
    })

    expect(warnings).toEqual([
      expect.stringContaining('both its candidate cap and its time deadline'),
    ])
  })
})
