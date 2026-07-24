import type { ReconstructionProgress } from '../../types/reconstruction'

export type CalibrationWorkflowStep = {
  id: string
  label: string
  purpose: string
  state: 'done' | 'current' | 'upcoming'
}

const FULL_STEPS = [
  ['prepare', 'Prepare immutable inputs', 'Load source-resolution frames and the existing calibration artifact.'],
  ['direct', 'Run direct PnLCalib', 'Infer field points and lines on every selected direct frame.'],
  ['local-retry', 'Retry local rejects', 'Retry rejected frames up to two times with frame-local candidate changes.'],
  ['p95-retry', 'Recheck p95 outliers', 'Retry frames rejected only after shot-wide residual p95 quality checks.'],
  ['temporal', 'Resolve temporal gaps', 'Use accepted direct anchors and measured camera motion for remaining frames.'],
  ['publish', 'Validate and publish', 'Build the immutable calibration artifact and stop for operator review.'],
] as const

const INCREMENTAL_STEPS = [
  ['prepare', 'Load published calibration', 'Verify the draft was made against the current immutable artifact.'],
  ['apply', 'Apply staged corrections', 'Replace evidence only for the manually edited frames.'],
  ['affected', 'Recompute affected dependants', 'Resolve only frames whose temporal solution can depend on an edited frame.'],
  ['publish', 'Validate and publish', 'Publish a new immutable artifact and clear the staged edit session.'],
] as const

function currentStepIndex(
  progress: ReconstructionProgress | null,
  incremental: boolean,
): number {
  const label = `${progress?.label ?? ''} ${progress?.detail ?? ''}`.toLowerCase()
  if (label.includes('publish') || label.includes('finaliz') || label.includes('review gate')) {
    return incremental ? 3 : 5
  }
  if (incremental) {
    if (label.includes('affected') || label.includes('depend')) return 2
    if (label.includes('apply') || label.includes('manual') || label.includes('draft')) return 1
    return 0
  }
  if (label.includes('temporal') || label.includes('gap')) return 4
  if (label.includes('p95') || label.includes('shot-wide')) return 3
  if (label.includes('retry') || label.includes('reject')) return 2
  if (label.includes('pnlcalib') || label.includes('direct')) return 1
  return 0
}

export function calibrationWorkflowSteps(
  progress: ReconstructionProgress | null,
  calibrationTrigger: 'full-request' | 'manual-draft-finalize' | null,
): CalibrationWorkflowStep[] {
  const incremental = calibrationTrigger === 'manual-draft-finalize'
  const definitions = incremental ? INCREMENTAL_STEPS : FULL_STEPS
  const current = currentStepIndex(progress, incremental)
  return definitions.map(([id, label, purpose], index) => ({
    id,
    label,
    purpose,
    state: index < current ? 'done' : index === current ? 'current' : 'upcoming',
  }))
}
