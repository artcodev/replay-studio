import type { CalibrationFrameEvidence, CalibrationFrameQuality } from './calibration'
import type { Keyframe, TrackObservation } from './tracking'

export type ReconstructionModel =
  | 'yolo26n.pt'
  | 'yolo26s.pt'
  | 'yolo26m.pt'
  | 'yolo26l.pt'
  | 'yolo26x.pt'

export type BallDetectionBackend =
  | 'generic-ultralytics'
  | 'dedicated-ultralytics'
  | 'wasb-service'

export type BallTrajectoryMode = 'automatic' | 'manual'

/** Queued immutable run input: skip is valid only while the manual ball trajectory is authoritative. */
export type BallDetectionProfile = 'automatic' | 'skip-manual-authoritative'

/** Queued immutable run input: 'off' trades automatic shirt-number merge evidence for a cheaper run. */
export type JerseyOcrProfile = 'automatic' | 'off'

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
