import type { BallDetectionBackend } from './reconstruction'
import type { CanonicalPersonStatus } from './identity'
import type { MetricProjectionStatus, ObservationPositionSource, TrackObservation } from './tracking'

export type FrameAnnotationKind =
  | 'home-player'
  | 'away-player'
  | 'home-goalkeeper'
  | 'away-goalkeeper'
  | 'referee'
  | 'other'
  | 'ignore'

export type FrameIdentityAction = 'confirm' | 'exclude' | 'merge' | 'split'

export type FrameIdentityScope = 'observation' | 'range' | 'identity'

export type FrameIdentityPreviewState = 'uncorrected' | 'confirmed' | 'excluded' | 'merged' | 'split'

export type FrameAnnotation = {
  id: string
  sceneTime: number
  sourceTime: number
  frameIndex: number
  bbox: { x: number; y: number; width: number; height: number }
  kind: FrameAnnotationKind
  label: string | null
  externalPlayerId: string | null
  action?: FrameIdentityAction
  scope?: FrameIdentityScope
  mergeTargetId?: string | null
  sourceTrackId?: string | null
  canonicalPersonId?: string | null
  targetObservationId?: string | null
  targetObservation?: TrackObservation | null
  rangeStart?: number | null
  rangeEnd?: number | null
  splitCanonicalPersonId?: string | null
  affectedPreview?: {
    canonicalPersonId: string
    splitCanonicalPersonId: string
    rangeStart: number
    rangeEnd: number
    affectedObservationCount: number
    remainingObservationCount: number
  } | null
  previewState?: Exclude<FrameIdentityPreviewState, 'uncorrected'>
  correctionKind?: 'canonical-roster-binding-v1' | string
  rosterBindingState?: 'bound' | 'unbound'
  updatedAt: string
}

export type FrameAnalysis = {
  sceneId: string
  requestedTime: number
  sceneTime: number
  ballSceneTime?: number
  ballFrameIndex?: number
  sourceTime: number
  frameIndex: number
  frameWidth: number
  frameHeight: number
  model: string
  ballBackend?: BallDetectionBackend
  projectionMode: string
  calibrationStatus: 'ready' | 'review' | 'approximate' | 'fallback' | 'rejected'
  matchedTracks: number
  people: Array<{
    id: string
    confidence: number
    bbox: { x: number; y: number; width: number; height: number }
    // null: the person is real but has no honest pitch position
    // (off-pitch, or the published track was dropped by QA).
    pitch: { x: number; z: number } | null
    jerseyColor: string
    annotationId: string | null
    annotationIds?: string[]
    annotationLabel: string | null
    kind: FrameAnnotationKind | null
    source: 'automatic' | 'manual' | 'tracked-observation'
    matchedTrackId: string | null
    matchedTrackLabel: string | null
    canonicalPersonId?: string | null
    identityStatus?: CanonicalPersonStatus | null
    identityConfidence?: number | null
    identitySource?: string | null
    displayName?: string | null
    jerseyNumber?: string | null
    teamId: string | null
    matchDistance: number | null
    observationId?: string | null
    matchSource?: 'persisted-observation' | 'manual-identity' | null
    matchIou?: number | null
    matchConfidence?: number | null
    matchMargin?: number | null
    metricStatus: MetricProjectionStatus | null
    metricReason: string | null
    rawPitch?: { x: number; z: number } | null
    positionSource: ObservationPositionSource | null
    correctionAction: FrameIdentityAction | null
    correctionScope: FrameIdentityScope | null
    mergeTargetId: string | null
    sourceTrackId: string | null
    targetObservationId?: string | null
    rangeStart?: number | null
    rangeEnd?: number | null
    splitCanonicalPersonId?: string | null
    affectedPreview?: FrameAnnotation['affectedPreview']
    previewState: FrameIdentityPreviewState
  }>
  annotations: FrameAnnotation[]
  correctionSummary: { confirmed: number; excluded: number; merged: number; split?: number }
  ballCandidates: Array<{
    id: string
    confidence: number
    image: { x: number; y: number }
    pitch: { x: number; z: number }
    primary: boolean
    backend?: string
  }>
  warnings: string[]
  reconstruction?: {
    status: 'queued' | 'processing' | 'ready' | 'cancelled' | 'failed'
    model?: string
    runId?: string
    runRevision?: number
    inputFingerprint?: string
  }
}

export type ModelComparisonRun = {
  model: string
  frameCount: number
  totalDetections: number
  meanDetectionsPerFrame: number
  minimumDetectionsInFrame: number
  maximumDetectionsInFrame: number
  inPitchDetections: number
  outsidePitchDetections: number
  wouldClampDetections: number
  lowConfidenceDetections: number
  rawTrackCount: number
  stableTrackCount: number
  acceptedTrackCount: number
  boundaryRiskTrackCount: number
  inferenceSeconds: number
  meanInferenceMilliseconds: number
}

export type ModelComparisonReport = {
  sceneId: string
  completedAt: string
  frameCount: number
  settings: {
    imageSize: number
    confidence: number
    device: string
    outsidePitchMarginMetres: number
    metricProjection: boolean
  }
  baseline: ModelComparisonRun
  candidate: ModelComparisonRun
  comparison: {
    sharedDetections: number
    baselineOnlyDetections: number
    candidateOnlyDetections: number
    baselineOnlyInPitchDetections: number
    candidateOnlyInPitchDetections: number
    inPitchObservationGain: number
    outsidePitchDetectionDelta: number
    stableTrackDelta: number
    acceptedTrackDelta: number
    verdict: 'candidate' | 'baseline' | 'review'
    rationale: string[]
  }
  warnings: string[]
}

export type ModelComparisonQueue = {
  runId: string
  sceneId: string
  kind: 'model-comparison'
  status: 'queued' | 'waiting' | 'processing'
}
