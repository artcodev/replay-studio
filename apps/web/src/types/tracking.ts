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
  presenceState?: 'observed' | 'inferred-before-first' | 'inferred-gap' | 'inferred-after-last'
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
  positionSource?: ObservationPositionSource | null
  canonicalPersonId?: string | null
}

export type Track = {
  id: string
  label: string
  teamId: string
  color: string
  number: number
  role?: CanonicalPersonRole
  canonicalPersonId?: string | null
  source?: 'automatic' | 'manual-anchor'
  annotationIds?: string[]
  identityCorrection?: {
    status: 'merged'
    targetId: string
    annotationIds: string[]
    mergedTrackIds: string[]
  }
  externalPlayerId: string | null
  presence?: {
    policy: 'continuous-latent'
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
