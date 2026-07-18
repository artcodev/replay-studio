import type { CalibrationFrameEvidence } from '../../types/calibration'
import type {
  ProcessingStatus,
  QualityGateStatus,
  QualityVerdict,
  ReconstructionQuality,
} from '../../types/reconstruction'

export type CalibrationGateView = {
  id: string
  label: string
  status: QualityGateStatus
  value: string | null
  threshold: string | null
  detail: string | null
}

type CalibrationHypothesis = NonNullable<CalibrationFrameEvidence['hypotheses']>[number]

export function nearestCalibrationFrame(
  frames: readonly CalibrationFrameEvidence[],
  sceneTime: number,
): CalibrationFrameEvidence | null {
  if (!frames.length) return null
  return frames.reduce((nearest, frame) => (
    Math.abs(frame.sceneTime - sceneTime) < Math.abs(nearest.sceneTime - sceneTime)
      ? frame
      : nearest
  ))
}

export function calibrationBallBackendSummary(counts: Record<string, number> | null | undefined) {
  return Object.entries(counts ?? {})
    .map(([backend, count]) => `${backend} × ${count}`)
    .join(' · ') || 'not recorded'
}

export function selectedCalibrationHypothesis(
  frame: CalibrationFrameEvidence | null,
): CalibrationHypothesis | null {
  if (!frame?.hypotheses?.length) return null
  return frame.hypotheses.find((hypothesis) => (
    hypothesis.selected || hypothesis.id === frame.selectedHypothesisId
  )) ?? null
}

export function calibrationReportReasons(quality: ReconstructionQuality | null | undefined): string[] {
  const summary = quality?.summary
  const reasons = Array.isArray(summary)
    ? summary
    : typeof summary === 'string' && summary.trim()
      ? [summary]
      : summary && typeof summary === 'object'
        ? [
            ...(Array.isArray(summary.reasons) ? summary.reasons.filter((item): item is string => typeof item === 'string') : []),
            ...(typeof summary.message === 'string' ? [summary.message] : []),
          ]
        : []
  const limitations = (quality?.limitations ?? []).map((item) => (
    typeof item === 'string' ? item : `${item.code}: ${item.message}`
  ))
  return [...reasons, ...limitations]
}

export function formatCalibrationValue(value: unknown, unit?: unknown): string | null {
  if (value === null || value === undefined || value === '') return null
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>
    if ('value' in record) return formatCalibrationValue(record.value, record.unit ?? unit)
    return JSON.stringify(value)
  }
  if (typeof value === 'number' && unit === 'ratio') return `${Math.round(value * 100)}%`
  const unitLabel = unit === 'pixels'
    ? 'px'
    : unit === 'seconds'
      ? 's'
      : unit === 'metres' || unit === 'meters'
        ? 'm'
        : unit
  const formatted = typeof value === 'number'
    ? Math.abs(value) < 1 ? value.toFixed(3) : value.toFixed(1)
    : String(value)
  return `${formatted}${typeof unitLabel === 'string' && unitLabel ? ` ${unitLabel}` : ''}`
}

export function formatCalibrationThreshold(value: unknown, unit?: unknown): string | null {
  if (!value || typeof value !== 'object') return formatCalibrationValue(value, unit)
  const labels: Record<string, string> = {
    passAtLeast: 'pass ≥',
    passAtMost: 'pass ≤',
    reviewAtLeast: 'review ≥',
    reviewAtMost: 'review ≤',
    rejectBelow: 'reject <',
    rejectAbove: 'reject >',
  }
  const parts = Object.entries(value as Record<string, unknown>)
    .filter(([, threshold]) => threshold !== null && threshold !== undefined)
    .map(([key, threshold]) => `${labels[key] ?? key} ${formatCalibrationValue(threshold, unit)}`)
  return parts.length ? parts.join(' · ') : null
}

export function normalizeCalibrationGates(
  quality: ReconstructionQuality | null | undefined,
): CalibrationGateView[] {
  const rawGates = quality?.gates ?? []
  const gates = Array.isArray(rawGates)
    ? rawGates
    : Object.entries(rawGates).map(([id, gate]) => ({ id, ...gate }))
  return gates.map((gate, index) => {
    const record = gate as Record<string, unknown>
    const rawStatus = String(record.status ?? 'not-available').toLowerCase()
    const status: QualityGateStatus = rawStatus === 'pass' || rawStatus === 'passed'
      ? 'pass'
      : rawStatus === 'review' || rawStatus === 'warning' || rawStatus === 'warn'
        ? 'review'
        : rawStatus === 'fail' || rawStatus === 'failed' || rawStatus === 'reject'
          ? 'fail'
          : 'not-available'
    return {
      id: String(record.id ?? record.name ?? `gate-${index}`),
      label: String(record.label ?? record.name ?? record.id ?? `Quality gate ${index + 1}`),
      status,
      value: formatCalibrationValue(record.value ?? record.metricValue, record.unit),
      threshold: formatCalibrationThreshold(record.threshold ?? record.thresholds, record.unit),
      detail: typeof record.reason === 'string'
        ? record.reason
        : typeof record.detail === 'string'
          ? record.detail
          : typeof record.note === 'string'
            ? record.note
            : null,
    }
  })
}

export function formatCalibrationPercent(value: number | null | undefined) {
  return value === null || value === undefined || !Number.isFinite(value)
    ? '—'
    : `${Math.round(value * 100)}%`
}

export function formatCalibrationNumber(value: number | null | undefined, suffix = '') {
  return value === null || value === undefined || !Number.isFinite(value)
    ? '—'
    : `${value.toFixed(1)}${suffix}`
}

export function calibrationPersonSupport(value: CalibrationFrameEvidence['personSupport']) {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return String(value)
  return `${value.supported} / ${value.total} · ${formatCalibrationPercent(value.ratio)}`
}

export function isTemporalCalibrationProjection(source: CalibrationFrameEvidence['projectionSource']) {
  return source === 'temporal-forward'
    || source === 'temporal-backward'
    || source === 'temporal-bidirectional'
}

export function isAmbiguousCalibrationFrame(frame: CalibrationFrameEvidence) {
  return frame.solutionStatus === 'ambiguous'
    || (Boolean(frame.hypotheses?.length) && !frame.hypotheses?.some((hypothesis) => hypothesis.selected))
      && Boolean(frame.rejectionReasons?.some((reason) => reason.includes('conflict')))
}

export function calibrationObservationLabel(frame: CalibrationFrameEvidence) {
  if (frame.observationStatus) return frame.observationStatus.replaceAll('-', ' ')
  if (frame.projectionSource === 'direct' && frame.status === 'accepted') return 'direct accepted'
  return frame.source !== 'none' ? 'direct rejected / unavailable' : 'missing'
}

export function calibrationSolutionLabel(frame: CalibrationFrameEvidence) {
  if (isAmbiguousCalibrationFrame(frame)) return 'ambiguous — no projection selected'
  if (frame.solutionStatus) return frame.solutionStatus.replaceAll('-', ' ')
  if (isTemporalCalibrationProjection(frame.projectionSource) && frame.status === 'accepted') return 'recovered'
  return frame.status
}

export function calibrationTemporalDirection(frame: CalibrationFrameEvidence) {
  if (frame.temporal?.direction) return frame.temporal.direction
  if (frame.projectionSource.startsWith('temporal-')) return frame.projectionSource.replace('temporal-', '')
  return selectedCalibrationHypothesis(frame)?.origin.replace('temporal-', '') ?? '—'
}

export function calibrationTemporalAnchors(frame: CalibrationFrameEvidence) {
  const indices = frame.temporal?.anchorFrameIndices
    ?? selectedCalibrationHypothesis(frame)?.anchorFrameIndices
    ?? []
  return indices.length ? indices.map((index) => `#${index}`).join(' + ') : '—'
}

export function calibrationTemporalGap(frame: CalibrationFrameEvidence) {
  return frame.temporal?.temporalDistanceSeconds
    ?? selectedCalibrationHypothesis(frame)?.temporalDistanceSeconds
    ?? null
}

export function calibrationTemporalMotionConfidence(frame: CalibrationFrameEvidence) {
  return frame.temporal?.motionConfidence
    ?? selectedCalibrationHypothesis(frame)?.motionConfidence
    ?? null
}

export function calibrationFrameUncertainty(frame: CalibrationFrameEvidence) {
  return frame.uncertainty?.p95Metres
    ?? selectedCalibrationHypothesis(frame)?.uncertaintyP95Metres
    ?? frame.positionUncertaintyMetres
    ?? null
}

export function calibrationMotionLabel(frame: CalibrationFrameEvidence) {
  const motion = frame.cameraMotion
  if (!motion) return 'not recorded'
  return motion.confidence === null || motion.confidence === undefined
    ? motion.status
    : `${motion.status} · ${formatCalibrationPercent(motion.confidence)}`
}

export function calibrationFrameStatus(frame: CalibrationFrameEvidence) {
  if (isAmbiguousCalibrationFrame(frame)) return 'ambiguous'
  if (frame.status === 'accepted' && frame.projectionSource === 'direct') return 'direct'
  if (frame.status === 'accepted' && isTemporalCalibrationProjection(frame.projectionSource)) return 'recovered'
  if (frame.status === 'accepted') return 'propagated'
  return frame.status
}

export function calibrationFrameStatusLabel(frame: CalibrationFrameEvidence) {
  if (isAmbiguousCalibrationFrame(frame)) return 'AMBIGUOUS · NOT SELECTED'
  if (frame.status === 'accepted' && frame.projectionSource === 'direct') return 'DIRECT · ACCEPTED'
  if (frame.status === 'accepted' && isTemporalCalibrationProjection(frame.projectionSource)) {
    return `${frame.projectionSource.toUpperCase()} · RECOVERED`
  }
  if (frame.status === 'accepted') return `${frame.projectionSource.toUpperCase()} · ACCEPTED`
  return frame.status.toUpperCase()
}

export function calibrationProcessingLabel(status: ProcessingStatus) {
  if (status === 'completed') return 'COMPUTE COMPLETE'
  if (status === 'processing') return 'COMPUTING'
  if (status === 'queued') return 'QUEUED'
  return 'COMPUTE FAILED'
}

export function calibrationVerdictLabel(verdict: QualityVerdict) {
  if (verdict === 'pending') return 'QUALITY PENDING'
  return verdict === 'unknown' ? 'NOT EVALUATED' : `QUALITY ${verdict.toUpperCase()}`
}
