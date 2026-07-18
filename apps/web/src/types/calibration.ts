export type PitchCalibrationPreset =
  | 'penalty-area-left'
  | 'goal-area-left'
  | 'center-circle'
  | 'goal-area-right'
  | 'penalty-area-right'

export type PitchCalibrationAnchor = {
  id: string
  label: string
  image: { x: number; y: number }
  pitch: { x: number; z: number }
}

export type PitchCalibrationAlignmentMetrics = {
  precision: number
  recall: number
  f1: number
  residualP50: number
  residualP95: number
  modelSampleCount?: number
  observedSampleCount?: number
  tolerancePixels?: number
}

export type PitchCalibrationBackendDiagnostics = {
  inputWidth?: number
  inputHeight?: number
  elapsedSeconds?: number
  budgetExhausted?: boolean
  quadCandidateLimit?: number
  candidatePoolLimit?: number
  quadCandidatesGenerated?: number
  quadCandidatesEvaluated?: number
  terminationReason?: 'candidate-limit' | 'candidate-pool-limit' | 'deadline' | 'completed' | string
  candidateLimitReached?: boolean
  candidatePoolLimitReached?: boolean
  candidateEvaluationLimitReached?: boolean
  deadlineExceeded?: boolean
}

export type CalibrationEvidenceLine = {
  id?: string | number | null
  name?: string | null
  label?: string | null
  family?: string | null
  start?: { x: number; y: number } | null
  end?: { x: number; y: number } | null
  points?: Array<{ x: number; y: number }>
  confidence?: number | null
  groundPlane?: boolean | null
  residualP50?: number | null
  residualP95?: number | null
  residualStatus?: 'scored' | 'not-scored' | 'not-scored-3d' | string | null
  accepted?: boolean | null
  inlier?: boolean | null
}

export type PitchCalibrationDraft = {
  sceneId: string
  sceneTime: number
  frameIndex: number
  frameWidth: number
  frameHeight: number
  source: 'reconstruction' | 'frame-evidence' | 'saved' | 'approximate-seed' | 'manual-seed' | 'manual'
  status?: 'accepted' | 'review' | 'rejected' | 'missing'
  method?: string | null
  backend?: string | null
  confidenceKind?: string | null
  visiblePitchSide?: 'left' | 'right' | 'unknown' | null
  preset: PitchCalibrationPreset
  confidence: number
  alignmentError: number | null
  alignmentMetrics?: PitchCalibrationAlignmentMetrics | null
  horizon?: {
    start: { x: number; y: number }
    end: { x: number; y: number }
  } | null
  quality: 'good' | 'review' | 'poor'
  anchors: PitchCalibrationAnchor[]
  markings: Array<{
    id: string
    kind: 'line' | 'curve'
    points: Array<{ x: number; y: number }>
  }>
  imageToPitch: number[][]
  keypoints?: CalibrationEvidencePoint[]
  detectedKeypoints?: CalibrationEvidencePoint[]
  rawLines?: CalibrationEvidenceLine[]
  detectedKeypointCount?: number | null
  inlierCount?: number | null
  inlierRatio?: number | null
  keypointCount?: number | null
  reprojectionP95?: number | null
  rejectionReasons?: string[]
  evidence?: {
    backendDiagnostics?: PitchCalibrationBackendDiagnostics | null
    visiblePitchSide?: 'left' | 'right' | 'unknown' | null
    rawLines?: CalibrationEvidenceLine[]
  } | null
  warnings: string[]
}

export type CalibrationFrameStatus = 'pass' | 'review' | 'reject' | 'missing'

export type ProjectionSource =
  | 'direct'
  | 'temporal-forward'
  | 'temporal-backward'
  | 'temporal-bidirectional'
  | 'temporal-interpolation'
  | 'dense-bracket-interpolated'
  | 'manual-propagated'
  | 'screen-approximate'
  | 'propagated'
  | 'presence-inferred'
  | 'manual'
  | 'manual-pitch-coordinate'
  | 'approximate'
  | 'screen'
  | 'none'

export type CameraMotionStatus =
  | 'first-frame'
  | 'reference'
  | 'estimated'
  | 'unreliable'
  | 'cut'
  | 'unestimated'
  | 'missing'

export type CameraMotionEvidence = {
  status: CameraMotionStatus
  model?: string | null
  confidence?: number | null
  currentToPrevious?: number[][] | null
  currentToReference?: number[][] | null
  metrics?: {
    trackedCount?: number | null
    inlierCount?: number | null
    inlierRatio?: number | null
    residualP50Px?: number | null
    residualP95Px?: number | null
    forwardBackwardP95Px?: number | null
    coverageRatio?: number | null
    sceneChangeScore?: number | null
  } | null
  rejectionReasons?: string[]
}

export type CalibrationHypothesisEvidence = {
  id: string
  rank: number
  selected: boolean
  origin: 'direct' | 'direct-rejected' | 'temporal-direct' | 'temporal-forward' | 'temporal-backward'
  eligibility?: 'rejected-observation' | null
  score: number
  scoreKind?: string | null
  visiblePitchSide?: 'left' | 'right' | 'unknown' | null
  anchorFrameIndices: number[]
  anchorSampleIndices?: number[]
  motionEdgeIndices?: number[]
  temporalDistanceSeconds?: number | null
  motionConfidence?: number | null
  uncertaintyP95Metres?: number | null
  disagreementMetres?: number | null
  imageToPitch?: number[][] | null
  rejectionReasons?: string[]
}

export type TemporalCalibrationEvidence = {
  direction: 'forward' | 'backward' | 'bidirectional'
  anchorFrameIndices: number[]
  anchorSampleIndices?: number[]
  anchorSceneTimes?: number[]
  motionEdgeIndices?: number[]
  temporalDistanceSeconds?: number | null
  motionConfidence?: number | null
}

export type CalibrationUncertaintyEvidence = {
  kind?: string | null
  p50Metres?: number | null
  p95Metres?: number | null
  temporalDistanceSeconds?: number | null
  motionConfidence?: number | null
  source?: string | null
  reasons?: string[]
}

export type CalibrationEvidencePoint = {
  id?: string | null
  label?: string | null
  image: { x: number; y: number }
  projected?: { x: number; y: number } | null
  projectedImage?: { x: number; y: number } | null
  residualPx?: number | null
  residualVector?: { dx: number; dy: number; magnitude: number } | null
  inlier?: boolean | null
}

export type CalibrationEvidenceMarking = {
  id: string
  kind: 'line' | 'curve'
  label?: string | null
  points: Array<{ x: number; y: number }>
  accepted?: boolean | null
}

export type CalibrationFrameQuality = {
  frameIndex: number
  sourceFrameIndex?: number | null
  sceneTime: number
  sourceTime?: number | null
  status: CalibrationFrameStatus
  projectionSource: ProjectionSource
  calibrationId?: string | null
  model?: string | null
  confidence?: number | null
  frameWidth?: number | null
  frameHeight?: number | null
  keypointCount?: number | null
  inlierCount?: number | null
  inlierRatio?: number | null
  reprojectionErrorP50Px?: number | null
  reprojectionErrorP95Px?: number | null
  temporalJitterMetres?: number | null
  visibleSide?: 'left' | 'right' | 'unknown' | null
  imageToPitch?: number[][] | null
  keypoints?: CalibrationEvidencePoint[]
  markings?: CalibrationEvidenceMarking[]
  reasons?: string[]
  warnings?: string[]
}

export type CalibrationFrameEvidence = {
  sourceFrameIndex: number
  sampleIndex: number
  sceneTime: number
  sourceTime: number
  status: 'accepted' | 'rejected' | 'missing'
  observationStatus?: 'direct-accepted' | 'direct-rejected' | 'manual-propagated' | 'missing'
  solutionStatus?:
    | 'direct-accepted'
    | 'temporal-accepted'
    | 'temporal-rejected'
    | 'manual-accepted'
    | 'ambiguous'
    | 'unresolved'
  source: string
  projectionSource: ProjectionSource
  backend?: string | null
  confidence?: number | null
  imageToPitch?: number[][] | null
  keypointCount?: number | null
  detectedKeypointCount?: number | null
  inlierCount?: number | null
  inlierRatio?: number | null
  reprojectionError?: number | null
  reprojectionP95?: number | null
  visiblePitchSide?: 'left' | 'right' | 'unknown' | null
  rejectionReasons?: string[]
  personSupport?: number | { supported: number; total: number; ratio: number } | null
  cameraMotion?: CameraMotionEvidence | null
  temporal?: TemporalCalibrationEvidence | null
  uncertainty?: CalibrationUncertaintyEvidence | null
  positionUncertaintyMetres?: number | null
  selectedHypothesisId?: string | null
  ambiguityMargin?: number | null
  hypotheses?: CalibrationHypothesisEvidence[]
  frameWidth?: number | null
  frameHeight?: number | null
  keypoints?: CalibrationEvidencePoint[]
  rawLines?: CalibrationEvidenceLine[]
  markings?: CalibrationEvidenceMarking[]
}

export type CalibrationEvidence = {
  schemaVersion: 1
  summary: {
    sampledFrameCount: number
    acceptedFrameCount: number
    rejectedFrameCount: number
    missingFrameCount: number
    directCoverage: number
    usableCoverage: number
    directFrameCount?: number
    temporalRecoveredFrameCount?: number
    temporalAmbiguousFrameCount?: number
    temporalUncertaintyP95Metres?: number | null
    cameraMotionReliability?: number | null
    cameraMotionEstimatedEdgeCount?: number
    cameraMotionUnreliableEdgeCount?: number
    cameraMotionCutCount?: number
    maxGapSeconds: number | null
    sideAgreement: number | null
    reprojectionP50?: number | null
    reprojectionP95?: number | null
    alignmentF1P10?: number | null
    visiblePitchSide?: 'left' | 'right' | null
  }
  frameEvidence: CalibrationFrameEvidence[]
}

export type PitchOrientation = {
  visiblePitchSide?: 'left' | 'right' | 'unknown'
  visiblePitchSideSource?: 'manual' | 'manual-calibration' | 'calibration' | 'unknown'
  visiblePitchSideAgreement?: number | null
  attackingGoal: 'left' | 'right' | 'unknown'
  attackingGoalSource?: 'manual' | 'unknown'
  updatedAt: string
}
