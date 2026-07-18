import type { CanonicalPersonRole, TrackObservation } from './tracking'

export type CanonicalPersonStatus = 'resolved' | 'provisional' | 'excluded'

export type CanonicalIdentityEvidenceKind =
  | 'manual'
  | 'reid'
  | 'jersey-ocr'
  | 'team'
  | 'role'
  | 'trajectory'
  | 'multi-pass'
  | 'multi-angle-identity'
  | 'roster-prior'

/**
 * Human-readable evidence published by the identity resolver. Model embeddings
 * stay in backend artifacts; the editor only receives compact provenance.
 */

export type CanonicalIdentityEvidence = {
  id: string
  kind: CanonicalIdentityEvidenceKind
  label: string
  value?: string | number | null
  confidence?: number | null
  supportCount?: number
  sampleCount?: number
  source?: string | null
  model?: string | null
  frameIndices?: number[]
  manual?: boolean
  /** Cross-view provenance published when an independently aligned replay supports this identity. */
  sourceSceneId?: string | null
  sourceCanonicalPersonId?: string | null
  signals?: string[]
  alignmentConfidence?: number | null
  alignmentMethod?: string | null
  observationCount?: number
}

export type CanonicalMultiAngleIdentityEvidence = CanonicalIdentityEvidence & {
  kind: 'multi-angle-identity'
  sourceSceneId: string
  sourceCanonicalPersonId: string
  signals: string[]
  observations: TrackObservation[]
  sourceTrackletIds: string[]
}

/** A hypothesis is not a binding until CanonicalPerson.externalPlayerId is set. */

export type CanonicalRosterCandidate = {
  externalPlayerId: string
  rank?: number | null
  score?: number | null
  identitySignalScore?: number | null
  name?: string | null
  number?: string | null
  position?: string | null
  teamId?: string | null
  reasons?: string[]
  conflicts?: string[]
  eligible?: boolean
  proposalStatus?: 'selected' | 'alternative' | 'ambiguous' | 'rejected' | string
  requiresManualConfirmation?: boolean
  evidence?: CanonicalIdentityEvidence[]
}

export type CanonicalIdentityConflict = {
  id: string
  code: string
  message: string
  severity: 'review' | 'blocking'
  relatedCanonicalPersonIds?: string[]
  relatedTrackletIds?: string[]
}

export type CanonicalPerson = {
  canonicalPersonId: string
  displayName: string
  identityStatus: CanonicalPersonStatus
  identityConfidence: number | null
  identitySource: string | null
  teamId: string | null
  role: CanonicalPersonRole | null
  jerseyNumber: string | null
  /** Accepted automatic or manual roster binding. Candidate IDs live elsewhere. */
  externalPlayerId: string | null
  memberTrackletIds: string[]
  annotationIds?: string[]
  observationCount?: number
  observations?: TrackObservation[]
  evidence: CanonicalIdentityEvidence[]
  multiAngleEvidence?: CanonicalMultiAngleIdentityEvidence[]
  sourcePassIds?: string[]
  rosterCandidates: CanonicalRosterCandidate[]
  conflicts: CanonicalIdentityConflict[]
  provenance?: 'automatic' | 'manual' | 'mixed'
}

export type CanonicalIdentityDiagnostics = {
  sourceTrackletCount: number
  canonicalPersonCount: number
  resolvedPersonCount: number
  provisionalPersonCount: number
  excludedPersonCount: number
  conflictPersonCount: number
  manualDecisionCount: number
  estimatedIdSwitchCount?: number | null
  duplicateOverlapSeconds?: number | null
  jerseyReadableCoverage?: number | null
  associationConfidenceP10?: number | null
  associationConfidenceP50?: number | null
  acceptedAssociationConfidenceP10?: number | null
  reviewAssociationConfidenceP50?: number | null
  strongReidBidirectionalEdgeCount?: number
  reidUsableObservationCount?: number
  reidSelectedIndependentSampleCount?: number
  reidCropCoverage?: number
  jerseyReliablePersonCount?: number
  jerseyProvisionalPersonCount?: number
  jerseyConflictPersonCount?: number
  rosterCandidateCount?: number
  groundTruthAvailable?: boolean
}
