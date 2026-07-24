import type { CalibrationFrameEvidence, CalibrationFrameQuality } from './calibration'
import type { Keyframe, TrackObservation } from './tracking'

export type ReconstructionModel =
  | 'yolo26n.pt'
  | 'yolo26s.pt'
  | 'yolo26m.pt'
  | 'yolo26l.pt'
  | 'yolo26x.pt'
  | 'football.pt'

export type BallDetectionBackend =
  | 'generic-ultralytics'
  | 'dedicated-ultralytics'
  | 'wasb-service'

export type BallTrajectoryMode = 'automatic' | 'manual'

/** Queued immutable run input: skip is valid only while the manual ball trajectory is authoritative. */
export type BallDetectionProfile = 'automatic' | 'skip-manual-authoritative'

/** Queued immutable run input: 'off' trades automatic shirt-number merge evidence for a cheaper run. */
export type JerseyOcrProfile = 'automatic' | 'off'
export type ContactPointProfile = 'bbox-bottom' | 'pose-feet'

/**
 * A calibrate run computes and publishes an immutable calibration artifact.
 * A full run may only consume that artifact; it has no calibration authority.
 */
export type ReconstructionMode = 'calibrate' | 'full'

/** Which stage the last run produced. Absent on pre-two-stage scenes. */
export type ReconstructionStage = 'calibration' | 'reconstruction'
export type ReconstructionResultState =
  | 'calibration-only'
  | 'current'
  | 'stale'
  | 'unknown'
  | 'unavailable'
export type TrackingCoordinatePolicy = 'metric-required' | 'explicit-image-fallback'

export type CalibrationProvenance = {
  schemaVersion: 1
  runId?: string | null
  producedAt?: string | null
  calibrationInputFingerprint: string
  dataFingerprint: string
  artifact: ReconstructionArtifactReference
  samplingFrameRate?: number | null
  directCalibrationMaxGapSeconds?: number | null
  totalFrames: number
  resolvedFrames: number
  unresolvedFrames: number
}

export type CalibrationArtifactInput = {
  schemaVersion: 1
  producerRunId?: string | null
  producedAt?: string | null
  calibrationInputFingerprint: string
  dataFingerprint: string
  artifact: ReconstructionArtifactReference
  samplingFrameRate?: number | null
  directCalibrationMaxGapSeconds?: number | null
  totalFrames: number
  resolvedFrames: number
  unresolvedFrames: number
  coordinateSpace?: string | null
}

/** review: unresolved frames remain; ready: 100% resolved; confirmed: operator accepted the gap. */
export type CalibrationReviewStatus = 'review' | 'ready' | 'confirmed'

export type CalibrationReviewSample = {
  sampleIndex: number
  sourceFrameIndex: number | null
  sceneTime: number | null
  solutionStatus: string
  projectionSource: string | null
  /** Present on entries from `frames`; the failing subset omits it (always false). */
  resolved?: boolean
  /** The operator removed this exact source frame from every analysis pipeline. */
  excluded?: boolean
  residualP95: number | null
  rejectionReasons: string[]
  acceptedByOperator: boolean
  manual: boolean
  /** Frame-local homography of the resolved calibration, for the inspection overlay. */
  imageToPitch?: number[][] | null
  frameWidth?: number | null
  frameHeight?: number | null
}

export type CalibrationReview = {
  status: CalibrationReviewStatus
  inputFingerprint: string | null
  calibrationInputFingerprint?: string | null
  totalFrames: number
  resolvedFrames: number
  unresolvedFrames: number
  resolvedRatio: number
  /** Every sampled frame (resolved and unresolved), in order, for browsing. */
  frames: CalibrationReviewSample[]
  unresolvedSamples: CalibrationReviewSample[]
  warnings: string[]
  confirmedAt?: string | null
  fallbackPolicy?: 'explicit-image-fallback'
  fallbackSampleIndices?: number[]
}

export type ReconstructionPhase = {
  id: string
  label: string
  status: 'completed' | 'current' | 'pending'
}

export type ReconstructionProgress = {
  phase: string
  phaseIndex: number
  phaseCount: number
  label: string
  detail: string | null
  completed: number
  total: number
  phasePercent: number
  overallPercent: number
  elapsedSeconds: number
  etaSeconds: number | null
  updatedAt: string
  phases: ReconstructionPhase[]
}

export type ProcessingStatus = 'queued' | 'processing' | 'completed' | 'cancelled' | 'failed'

export type QualityVerdict = 'pass' | 'review' | 'reject' | 'pending' | 'unknown'

export type QualityGateStatus = 'pass' | 'review' | 'fail' | 'not-available'

export type ReconstructionQualityGate = {
  id: string
  label: string
  status: QualityGateStatus
  value?: number | string | null
  unit?: string | null
  threshold?: string | null
  detail?: string | null
}

export type ReconstructionQualityReport = {
  verdict: QualityVerdict
  score?: number | null
  reasons?: string[]
  gates: ReconstructionQualityGate[]
  calibration: {
    sampledFrameCount: number
    directFrameCount: number
    propagatedFrameCount?: number
    rejectedFrameCount: number
    missingFrameCount?: number
    directCoverage: number
    metricCoverage?: number | null
    fallbackFrameCount: number
    reprojectionErrorP50Px?: number | null
    reprojectionErrorP95Px?: number | null
    temporalJitterP95Metres?: number | null
    sideFlipCount?: number
    frames: CalibrationFrameQuality[]
  }
}

export type ReconstructionQuality = {
  verdict: QualityVerdict
  summary?: string | string[] | Record<string, unknown> | null
  metrics?: Record<string, number | string | boolean | null | {
    value: number | string | boolean | null
    unit?: string | null
    source?: string | null
    [key: string]: unknown
  }>
  gates?: Array<ReconstructionQualityGate | Record<string, unknown>> | Record<string, Record<string, unknown>>
  limitations?: Array<string | { code: string; message: string }>
  identityValidation?: {
    groundTruthAvailable: boolean
    status: 'evaluated' | 'unavailable' | 'invalid'
    sampleCount?: number
    idf1: number | null
    idPrecision?: number | null
    idRecall?: number | null
    idSwitchCount?: number | null
    fragmentCount?: number | null
    duplicateAssignmentFrameCount?: number | null
    duplicateOverlapSeconds?: number | null
    duplicateOverlapTimebase?: 'seconds' | 'frame-index+explicit-fps' | 'frame-index-without-fps' | 'mixed'
    identityAssignmentFrameRate?: number | null
    hota?: number | null
    gsHota?: number | null
    reason?: string
  }
}

export type ReconstructionArtifactReference = {
  id: `sha256:${string}`
  kind: string
  schemaVersion: number
  uri: `artifact://sha256/${string}`
  sha256: string
  byteSize: number
  contentType: 'application/json'
}

export type ReconstructionArtifactManifest = {
  schemaVersion: 1
  artifacts: {
    identityDiagnostics?: ReconstructionArtifactReference
    identityTimeline?: ReconstructionArtifactReference
    ballTrajectory?: ReconstructionArtifactReference
    calibrationFrames?: ReconstructionArtifactReference
  }
}

export type ReconstructionSeriesWindow = {
  schemaVersion: 1
  sceneId: string
  window: {
    start: number
    end: number
    frameStart: number | null
    frameEnd: number | null
  }
  tracks: Array<{
    id: string
    keyframes: Keyframe[]
    observations: TrackObservation[]
  }>
  canonicalPeople: Array<{
    canonicalPersonId: string
    observations: TrackObservation[]
  }>
  ball: {
    keyframes: Keyframe[]
    automaticKeyframes: Keyframe[]
    manualKeyframes: Keyframe[]
  }
  calibration: { frameEvidence: CalibrationFrameEvidence[] }
  ballDetection: { frames: Array<Record<string, unknown>> }
}
