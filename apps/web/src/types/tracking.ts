import type { ProjectionSource } from './calibration'

export type Keyframe = {
  id?: string
  t: number
  x: number
  y?: number
  z: number
  confidence: number
  observed?: boolean
  state?: 'observed' | 'inferred' | 'occluded'
  detectionConfidence?: number | null
  trajectoryConfidence?: number | null
  confidenceKind?: string | null
  positionSource?: string | null
  heightSource?: string | null
  provenance?: {
    source: string
    method?: string
    [key: string]: unknown
  }
  presenceState?: 'observed' | 'inferred-gap'
  support?: number
  metricSupport?: number
  projection?: {
    source: ProjectionSource
    calibrationFrameIndex?: number | null
    uncertaintyMetres?: number | null
  }
  projectionSource?: ProjectionSource
  positionUncertaintyMetres?: number | null
}

export type MetricProjectionStatus = 'accepted' | 'rejected' | 'unprojected'

export type ObservationPositionSource = 'observation' | 'track-inferred'

export type CanonicalPersonRole = 'player' | 'goalkeeper' | 'referee' | 'other'

export type TrackObservation = {
  id?: string
  observationId?: string
  frameIndex: number
  sourceFrameIndex?: number
  sceneTime: number
  sourceTime?: number
  bbox: { x: number; y: number; width: number; height: number }
  pitch?: { x: number; z: number }
  rawPitch?: { x: number; z: number } | null
  confidence: number
  annotationId?: string | null
  annotationIds?: string[]
  metricStatus?: MetricProjectionStatus | null
  metricReason?: string | null
  associationCost?: number | null
  associationMargin?: number | null
  associationDiagnostics?: {
    model?: string
    coordinateMode?: 'metric' | 'explicit-image-fallback'
    rejectionReason?: string | null
    [key: string]: unknown
  } | null
  trackingDecision?: string | null
  trajectoryRejection?: {
    startTime: number
    endTime: number
    startFrameIndex?: number | null
    endFrameIndex?: number | null
    observationCount: number
    reason: string
  } | null
  positionSource?: ObservationPositionSource | null
  canonicalPersonId?: string | null
  projectionSource?: ProjectionSource | null
  calibrationFrameIndex?: number | null
  positionUncertaintyMetres?: number | null
  sourceTrackletId?: string | null
}

export type Track = {
  id: string
  label: string
  teamId: string
  color: string
  number: number
  role?: CanonicalPersonRole
  canonicalPersonId?: string | null
  source?: 'automatic' | 'manual-anchor' | 'provisional'
  /** The person exists in video evidence, but its team/roster identity is unresolved. */
  provisional?: boolean
  annotationIds?: string[]
  identityCorrection?: {
    status: 'merged'
    targetId: string
    annotationIds: string[]
    mergedTrackIds: string[]
  }
  externalPlayerId: string | null
  presence?: {
    policy: 'observed-window-with-latent-gaps'
    coverage: number
    observationCount: number
    inferredKeyframeCount: number
    observedStart?: number
    observedEnd?: number
    observedSpanRatio?: number
    sampleCadenceSeconds?: number
  }
  observations?: TrackObservation[]
  keyframeCount?: number
  observationCount?: number
  keyframes: Keyframe[]
}
