import type {
  CanonicalPerson,
  ExternalPlayer,
  IdentityReviewItem,
  IdentityReviewResponse,
  IdentityReviewWorkerHealth,
  TrackObservation,
} from '../types'

export type IdentityReviewBox = TrackObservation['bbox']

/**
 * Lightweight presentation contract for a review thumbnail. The reconstruction
 * may only have a detector box, while a later artifact endpoint can add a full
 * frame or crop URL without changing the canonical-person document.
 */
export type IdentityReviewObservation = {
  id: string
  observationId?: string | null
  frameIndex: number
  sceneTime: number
  bbox?: IdentityReviewBox | null
  frameWidth?: number | null
  frameHeight?: number | null
  previewUrl?: string | null
  cropUrl?: string | null
  confidence?: number | null
  quality?: number | null
  source?: string | null
  rejectionReasons?: string[]
}

export type IdentityReviewWorkerStatus =
  | 'ready'
  | 'processing'
  | 'degraded'
  | 'no-observations'
  | 'disabled'
  | 'unavailable'
  | 'failed'
  | 'invalid-response'
  | 'unknown'

export type IdentityReviewWorkerState = {
  id: string
  label: string
  status: IdentityReviewWorkerStatus
  configured?: boolean | null
  backend?: string | null
  modelVersion?: string | null
  detail?: string | null
  requestedCount?: number | null
  usableCount?: number | null
  rejectedCount?: number | null
  rejectionReasons?: string[]
}

export type IdentityReviewLinkCandidate = {
  id: string
  targetCanonicalPersonId: string
  targetLabel?: string | null
  confidence?: number | null
  source?: string | null
  status: 'review' | 'rejected'
  reasons?: string[]
  predecessorTrackletId?: string | null
  successorTrackletId?: string | null
}

export type IdentityReviewCandidateDecision =
  | {
      canonicalPersonId: string
      kind: 'roster'
      candidateId: string
      externalPlayerId: string
    }
  | {
      canonicalPersonId: string
      kind: 'identity-link'
      candidateId: string
      targetCanonicalPersonId: string
    }

export type IdentityReviewCannotLinkDecision = {
  canonicalPersonId: string
  candidateId: string
  targetCanonicalPersonId: string
}

export type IdentityReviewInspectFrame = {
  canonicalPersonId: string
  observationId: string | null
  frameIndex: number
  sceneTime: number
  bbox: IdentityReviewBox | null
}

export type ManualRosterBindingDecision = {
  canonicalPersonId: string
  externalPlayerId: string
}

/**
 * Build the same explicit roster-binding command used by ranked candidates,
 * but fail closed for a missing selection, a no-op, or a duplicated roster ID.
 */
export function manualRosterBindingDecision(
  canonicalPersonId: string,
  currentExternalPlayerId: string | null | undefined,
  selectedExternalPlayerId: string,
  rosterPlayers: readonly Pick<ExternalPlayer, 'id'>[],
): ManualRosterBindingDecision | null {
  if (!canonicalPersonId || !selectedExternalPlayerId.trim()) return null
  if (selectedExternalPlayerId === currentExternalPlayerId) return null
  if (rosterPlayers.filter((player) => player.id === selectedExternalPlayerId).length !== 1) return null
  return { canonicalPersonId, externalPlayerId: selectedExternalPlayerId }
}

export function rosterReviewDecision(
  canonicalPersonId: string,
  externalPlayerId: string,
): Extract<IdentityReviewCandidateDecision, { kind: 'roster' }> {
  return {
    canonicalPersonId,
    kind: 'roster',
    candidateId: externalPlayerId,
    externalPlayerId,
  }
}

export function linkReviewDecision(
  canonicalPersonId: string,
  candidate: Pick<IdentityReviewLinkCandidate, 'id' | 'targetCanonicalPersonId'>,
): Extract<IdentityReviewCandidateDecision, { kind: 'identity-link' }> {
  return {
    canonicalPersonId,
    kind: 'identity-link',
    candidateId: candidate.id,
    targetCanonicalPersonId: candidate.targetCanonicalPersonId,
  }
}

export function cannotLinkReviewDecision(
  canonicalPersonId: string,
  candidate: Pick<IdentityReviewLinkCandidate, 'id' | 'targetCanonicalPersonId'>,
): IdentityReviewCannotLinkDecision {
  return {
    canonicalPersonId,
    candidateId: candidate.id,
    targetCanonicalPersonId: candidate.targetCanonicalPersonId,
  }
}

export function inspectIdentityObservationDecision(
  canonicalPersonId: string,
  observation: IdentityReviewObservation,
): IdentityReviewInspectFrame {
  return {
    canonicalPersonId,
    observationId: observation.observationId || observation.id || null,
    frameIndex: observation.frameIndex,
    sceneTime: observation.sceneTime,
    bbox: observation.bbox ?? null,
  }
}

function finiteProbability(value: number | null | undefined): number {
  if (value === null || value === undefined || !Number.isFinite(value)) return -1
  return Math.min(1, Math.max(0, value))
}

function finiteQuality(value: number | null | undefined): number {
  if (value === null || value === undefined || !Number.isFinite(value)) return -1
  return Math.max(0, value)
}

function usableBox(box: IdentityReviewBox | null | undefined): box is IdentityReviewBox {
  return Boolean(
    box
    && [box.x, box.y, box.width, box.height].every(Number.isFinite)
    && box.width > 0
    && box.height > 0,
  )
}

export function observationHasReviewEvidence(observation: IdentityReviewObservation): boolean {
  return Boolean(
    observation.cropUrl?.trim()
    || observation.previewUrl?.trim()
    || usableBox(observation.bbox),
  )
}

/** Rank quality-selected crops deterministically and never invent a preview. */
export function topIdentityReviewObservations(
  observations: readonly IdentityReviewObservation[],
  limit = 8,
): IdentityReviewObservation[] {
  if (!Number.isFinite(limit) || limit <= 0) return []
  return observations
    .filter(observationHasReviewEvidence)
    .map((observation, index) => ({ observation, index }))
    .sort((left, right) => (
      finiteQuality(right.observation.quality)
      - finiteQuality(left.observation.quality)
      || Number(Boolean(right.observation.cropUrl?.trim()))
      - Number(Boolean(left.observation.cropUrl?.trim()))
      || Number(Boolean(right.observation.previewUrl?.trim()))
      - Number(Boolean(left.observation.previewUrl?.trim()))
      || finiteProbability(right.observation.confidence)
      - finiteProbability(left.observation.confidence)
      || left.observation.sceneTime - right.observation.sceneTime
      || left.observation.frameIndex - right.observation.frameIndex
      || left.index - right.index
    ))
    .slice(0, Math.floor(limit))
    .map(({ observation }) => observation)
}

export function canonicalReviewObservations(
  identity: Pick<CanonicalPerson, 'observations'>,
): IdentityReviewObservation[] {
  return (identity.observations ?? []).map((observation, index) => ({
    id: observation.id
      || observation.observationId
      || `frame-${observation.frameIndex}-observation-${index + 1}`,
    observationId: observation.observationId || observation.id || null,
    frameIndex: observation.frameIndex,
    sceneTime: observation.sceneTime,
    bbox: observation.bbox,
    confidence: observation.confidence,
    source: observation.positionSource || null,
    rejectionReasons: observation.metricReason ? [observation.metricReason] : [],
  }))
}

const WORKER_STATUS_LABELS: Record<IdentityReviewWorkerStatus, string> = {
  ready: 'Ready',
  processing: 'Processing',
  degraded: 'Degraded',
  'no-observations': 'No observations',
  disabled: 'Disabled',
  unavailable: 'Unavailable',
  failed: 'Failed',
  'invalid-response': 'Invalid response',
  unknown: 'Unknown',
}

export function identityReviewWorkerStatusLabel(status: IdentityReviewWorkerStatus): string {
  return WORKER_STATUS_LABELS[status]
}

function reviewWorkerStatus(value: unknown): IdentityReviewWorkerStatus {
  const status = String(value || 'unknown') as IdentityReviewWorkerStatus
  return status in WORKER_STATUS_LABELS ? status : 'unknown'
}

function optionalCount(health: IdentityReviewWorkerHealth, keys: string[]): number | null {
  for (const key of keys) {
    const value = health[key]
    if (typeof value === 'number' && Number.isFinite(value) && value >= 0) return value
  }
  return null
}

export function identityReviewWorkerStates(
  review: Pick<IdentityReviewResponse, 'workers'> | null,
): IdentityReviewWorkerState[] | null {
  if (!review) return null
  const reid = review.workers.identity ?? review.workers.reid
  const jersey = review.workers.jerseyOcr
  return [
    ...(reid ? [{ id: 'reid', label: 'Player ReID', health: reid }] : []),
    ...(jersey ? [{ id: 'jersey-ocr', label: 'Jersey OCR', health: jersey }] : []),
  ].map(({ id, label, health }) => ({
    id,
    label,
    status: reviewWorkerStatus(health.status),
    configured: health.configured,
    backend: health.backend ?? null,
    modelVersion: health.modelVersion ?? health.providerVersion ?? null,
    detail: health.detail ?? null,
    requestedCount: optionalCount(health, [
      'requestedObservationCount',
      'submittedCropCount',
      'selectedCropCount',
    ]),
    usableCount: optionalCount(health, [
      'usableObservationCount',
      'recognizedCropCount',
      'rawUsableObservationCount',
    ]),
    rejectedCount: optionalCount(health, [
      'rejectedObservationCount',
      'rejectedCropCount',
    ]),
    rejectionReasons: Array.isArray(health.rejectionReasons)
      ? health.rejectionReasons.filter((reason): reason is string => typeof reason === 'string')
      : [],
  }))
}

export function identityReviewItemObservations(
  item: Pick<IdentityReviewItem, 'representativeObservations'> | null,
): IdentityReviewObservation[] | null {
  if (!item) return null
  return item.representativeObservations
    .filter((observation) => (
      Number.isFinite(observation.frameIndex)
      && Number.isFinite(observation.sceneTime)
      && Boolean(observation.observationId)
    ))
    .map((observation) => {
      const rejectionReasons = [
        ...(observation.reid?.rejectionReasons ?? []),
        ...(observation.jerseyOcr?.rejectionReasons ?? []),
      ].filter((reason): reason is string => typeof reason === 'string' && Boolean(reason.trim()))
      return {
        id: observation.observationId,
        observationId: observation.observationId,
        frameIndex: observation.frameIndex,
        sceneTime: observation.sceneTime,
        bbox: observation.bbox,
        cropUrl: observation.cropUrl ?? null,
        confidence: observation.confidence ?? null,
        quality: observation.reviewQuality ?? null,
        rejectionReasons: [...new Set(rejectionReasons)],
      }
    })
}
