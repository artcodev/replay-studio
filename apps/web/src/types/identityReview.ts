import type { CanonicalIdentityConflict, CanonicalIdentityEvidence, CanonicalPersonStatus, CanonicalRosterCandidate } from './identity'
import type { CanonicalPersonRole, TrackObservation } from './tracking'

export type IdentityReviewCropDiagnostic = {
  status?: string | null
  usable?: boolean | null
  rejectionReasons?: string[]
  number?: string | null
  confidence?: number | null
}

export type IdentityReviewRepresentativeObservation = {
  observationId: string
  frameIndex: number
  sourceFrameIndex?: number | null
  sceneTime: number
  sourceTime?: number | null
  bbox: TrackObservation['bbox']
  confidence?: number | null
  reviewQuality?: number | null
  cropUrl?: string | null
  reid?: IdentityReviewCropDiagnostic | null
  jerseyOcr?: IdentityReviewCropDiagnostic | null
}

export type IdentityReviewResolutionState = 'conflict' | 'suggested' | 'anonymous' | 'bound' | 'excluded'

export type IdentityReviewItem = {
  canonicalPersonId: string
  displayName: string
  identityStatus: CanonicalPersonStatus
  identityConfidence?: number | null
  identitySource?: string | null
  teamId?: string | null
  role?: CanonicalPersonRole | null
  jerseyNumber?: string | null
  candidateNumber?: string | null
  externalPlayerId?: string | null
  renderTrackId?: string | null
  observationCount: number
  resolutionState: IdentityReviewResolutionState
  priority: number
  representativeObservations: IdentityReviewRepresentativeObservation[]
  evidence: CanonicalIdentityEvidence[]
  rosterCandidates: CanonicalRosterCandidate[]
  conflicts: CanonicalIdentityConflict[]
}

export type IdentityReviewWorkerHealth = {
  configured?: boolean | null
  status: string
  backend?: string | null
  modelVersion?: string | null
  providerVersion?: string | null
  detail?: string | null
  requestedObservationCount?: number | null
  submittedCropCount?: number | null
  selectedCropCount?: number | null
  usableObservationCount?: number | null
  recognizedCropCount?: number | null
  rawUsableObservationCount?: number | null
  rejectedObservationCount?: number | null
  rejectedCropCount?: number | null
  rejectionReasons?: string[]
}

export type IdentityReviewResponse = {
  sceneId: string
  revision: number
  matchSnapshot: {
    id?: string | null
    contentHash?: string | null
    matchId?: string | null
    roster: {
      status: 'ready' | 'incomplete' | 'review' | 'unavailable'
      playerCount: number
      complete: boolean
      automaticIdentityEligible: boolean
      manualIdentityEligible: boolean
      reasons: string[]
      warnings: string[]
    }
  }
  workers: {
    identity?: IdentityReviewWorkerHealth
    reid?: IdentityReviewWorkerHealth
    jerseyOcr?: IdentityReviewWorkerHealth
  }
  summary: {
    canonicalPersonCount: number
    boundCount: number
    suggestedCount: number
    conflictCount: number
    anonymousCount: number
    excludedCount: number
  }
  items: IdentityReviewItem[]
}
