import type { CalibrationEvidenceLine, CalibrationEvidencePoint, CalibrationFrameEvidence, PitchCalibrationDraft } from '../types/calibration'

export type CalibrationFrameDiagnostics = {
  evidence: CalibrationFrameEvidence | null
  points: CalibrationEvidencePoint[]
  lines: CalibrationEvidenceLine[]
  status: 'accepted' | 'review' | 'rejected' | 'missing'
  sourceStatus: CalibrationFrameEvidence['status'] | null
  method: string
  keypointCount: number | null
  inlierCount: number | null
  inlierRatio: number | null
  residualP50: number | null
  residualP95: number | null
  groundResidualP50: number | null
  groundResidualP95: number | null
  precision: number | null
  recall: number | null
  f1: number | null
  visibleSide: 'left' | 'right' | 'unknown'
  visibleSideTrusted: boolean
  rejectionReasons: string[]
}

const rejectionReasonLabels: Record<string, string> = {
  'semantic-line-alignment-poor': 'Projected markings do not align with the observed pitch lines.',
  'semantic-line-alignment-unscored': 'Projected-markings alignment could not be scored on this frame.',
  'temporal-semantic-line-alignment-poor': 'Temporally projected markings do not align with the observed pitch lines.',
  'semantic-keypoint-ground-error-too-high': 'Semantic keypoints have a ground-plane residual tail above 1 metre.',
}

export function calibrationRejectionReasonLabel(reason: string) {
  return rejectionReasonLabels[reason] ?? reason.replaceAll('-', ' ')
}

export function calibrationLineResidualLabel(line: CalibrationEvidenceLine) {
  if (line.residualStatus === 'not-scored-3d' || line.groundPlane === false) return '3D'
  if (line.residualP95 !== null && line.residualP95 !== undefined) {
    return `p95 ${line.residualP95.toFixed(1)}px`
  }
  if (line.residualStatus === 'not-scored') return 'not scored'
  return null
}

type CalibrationSearchStop = 'candidate-limit' | 'deadline' | 'candidate-limit-and-deadline' | 'unknown' | null

function calibrationSearchStop(draft: PitchCalibrationDraft): CalibrationSearchStop {
  const diagnostics = draft.evidence?.backendDiagnostics
  if (!diagnostics) return null
  const terminationReason = diagnostics.terminationReason?.toLowerCase() ?? ''
  const explicitCandidateLimit = Boolean(
    diagnostics.candidateLimitReached
    || diagnostics.candidatePoolLimitReached
    || diagnostics.candidateEvaluationLimitReached
    || terminationReason.includes('candidate')
  )
  const explicitDeadline = Boolean(
    diagnostics.deadlineExceeded
    || terminationReason.includes('deadline')
    || terminationReason.includes('timeout')
  )
  if (explicitCandidateLimit && explicitDeadline) return 'candidate-limit-and-deadline'
  if (explicitCandidateLimit) return 'candidate-limit'
  if (explicitDeadline) return 'deadline'

  const limit = diagnostics.quadCandidateLimit ?? 0
  const generated = diagnostics.quadCandidatesGenerated ?? 0
  const evaluated = diagnostics.quadCandidatesEvaluated ?? 0
  // The line fallback intentionally caps its generated pool at 12x the
  // evaluation limit. Reaching either cap is not evidence of a time deadline.
  const inferredCandidateLimit = limit > 0 && (generated >= limit * 12 || evaluated >= limit)
  const inferredDeadline = Boolean(
    diagnostics.budgetExhausted && (diagnostics.elapsedSeconds ?? 0) >= 4.75,
  )
  if (inferredCandidateLimit && inferredDeadline) return 'candidate-limit-and-deadline'
  if (inferredCandidateLimit) return 'candidate-limit'
  if (inferredDeadline) return 'deadline'
  return diagnostics.budgetExhausted ? 'unknown' : null
}

function searchStopWarning(draft: PitchCalibrationDraft, stop: Exclude<CalibrationSearchStop, null>) {
  const diagnostics = draft.evidence?.backendDiagnostics
  const generated = diagnostics?.quadCandidatesGenerated
  const evaluated = diagnostics?.quadCandidatesEvaluated
  if (stop === 'candidate-limit') {
    const counts = evaluated !== undefined || generated !== undefined
      ? ` (${evaluated ?? '—'} evaluated${generated !== undefined ? `, ${generated} generated` : ''})`
      : ''
    return `Line/curve fallback reached its candidate limit${counts}. This is a bounded candidate cap, not a time deadline.`
  }
  if (stop === 'candidate-limit-and-deadline') {
    const elapsed = diagnostics?.elapsedSeconds
    return `Line/curve fallback reached both its candidate cap and its time deadline${elapsed !== undefined ? ` after ${elapsed.toFixed(1)}s` : ''}; these are separate limits and the best-so-far result is shown.`
  }
  if (stop === 'deadline') {
    const elapsed = diagnostics?.elapsedSeconds
    return `Line/curve fallback reached its time deadline${elapsed !== undefined ? ` after ${elapsed.toFixed(1)}s` : ''}; its best-so-far result is shown.`
  }
  return 'Line/curve fallback stopped at a bounded search limit; the backend did not report whether the candidate cap or time deadline ended the search.'
}

export function calibrationPreviewWarnings(draft: PitchCalibrationDraft) {
  const stop = calibrationSearchStop(draft)
  const ambiguousBudgetWarning = (warning: string) => (
    /line\/curve fallback/i.test(warning)
    && /(search budget|budget exhausted|exhausted its|five-second deadline|candidate search limit|configured search budget)/i.test(warning)
  )
  const warnings = draft.warnings.filter((warning) => !ambiguousBudgetWarning(warning))
  if (stop) warnings.push(searchStopWarning(draft, stop))
  else if (draft.warnings.some(ambiguousBudgetWarning)) {
    warnings.push(searchStopWarning(draft, 'unknown'))
  }
  return warnings.filter((warning, index, all) => all.indexOf(warning) === index)
}

export function calibrationEvidenceAtTime(
  frames: CalibrationFrameEvidence[],
  sceneTime: number,
  toleranceSeconds = 0.12,
) {
  if (!frames.length) return null
  const nearest = frames.reduce((best, frame) => (
    Math.abs(frame.sceneTime - sceneTime) < Math.abs(best.sceneTime - sceneTime)
      ? frame
      : best
  ))
  return Math.abs(nearest.sceneTime - sceneTime) <= toleranceSeconds ? nearest : null
}

export function calibrationFrameDiagnostics(
  draft: PitchCalibrationDraft,
  frames: CalibrationFrameEvidence[],
): CalibrationFrameDiagnostics {
  const evidence = calibrationEvidenceAtTime(frames, draft.sceneTime)
  const points = draft.detectedKeypoints !== undefined
    ? draft.detectedKeypoints
    : draft.keypoints !== undefined
      ? draft.keypoints
      : evidence?.keypoints ?? []
  const metrics = draft.alignmentMetrics
  const lines = draft.rawLines !== undefined
    ? draft.rawLines
    : draft.evidence?.rawLines !== undefined
      ? draft.evidence.rawLines
      : evidence?.rawLines ?? []
  const qualityStatus = draft.quality === 'good'
    ? 'accepted'
    : draft.quality === 'poor'
      ? 'rejected'
      : 'review'
  const rejectionReasons = (
    draft.rejectionReasons !== undefined
      ? draft.rejectionReasons
      : evidence?.rejectionReasons ?? []
  ).filter((reason, index, all) => all.indexOf(reason) === index)
  const visibleSide = draft.visiblePitchSide
    ?? draft.evidence?.visiblePitchSide
    ?? (draft.source === 'frame-evidence' ? evidence?.visiblePitchSide : null)
    ?? 'unknown'
  const resolvedStatus = draft.status ?? qualityStatus

  return {
    evidence,
    points,
    lines,
    status: resolvedStatus,
    sourceStatus: evidence?.status ?? null,
    method: draft.method
      ?? draft.backend
      ?? (draft.source === 'frame-evidence' ? evidence?.backend ?? evidence?.source : null)
      ?? draft.source,
    keypointCount: draft.detectedKeypointCount
      ?? draft.keypointCount
      ?? evidence?.detectedKeypointCount
      ?? evidence?.keypointCount
      ?? (points.length || null),
    inlierCount: draft.inlierCount ?? evidence?.inlierCount ?? null,
    inlierRatio: draft.inlierRatio ?? evidence?.inlierRatio ?? null,
    residualP50: metrics?.residualP50 ?? draft.alignmentError ?? evidence?.reprojectionError ?? null,
    residualP95: metrics?.residualP95 ?? draft.reprojectionP95 ?? evidence?.reprojectionP95 ?? null,
    groundResidualP50: draft.groundErrorP50Metres
      ?? evidence?.groundErrorP50Metres
      ?? null,
    groundResidualP95: draft.groundErrorP95Metres
      ?? evidence?.groundErrorP95Metres
      ?? null,
    precision: metrics?.precision ?? null,
    recall: metrics?.recall ?? null,
    f1: metrics?.f1 ?? null,
    visibleSide,
    visibleSideTrusted: visibleSide !== 'unknown'
      && resolvedStatus === 'accepted'
      && draft.source !== 'manual'
      && draft.source !== 'manual-seed',
    rejectionReasons,
  }
}
