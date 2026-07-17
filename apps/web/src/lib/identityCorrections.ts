import type { FrameAnnotation, FrameIdentityAction, Track, TrackObservation } from '../types'

export type IdentityMergeTarget = {
  id: string
  label: string
  type: 'track' | 'annotation'
}

export type PublishedIdentityOwnership = {
  canonicalPeople?: Array<{
    canonicalPersonId: string
    observations?: TrackObservation[]
    annotationIds?: string[]
  }>
  tracks?: Array<Pick<Track, 'id' | 'canonicalPersonId' | 'observations' | 'annotationIds'>>
}

const DEDICATED_ROSTER_CORRECTION = 'canonical-roster-binding-v1'

export function semanticAnnotationForEdit(
  person: { annotationId?: string | null; annotationIds?: string[] },
  annotations: FrameAnnotation[],
): FrameAnnotation | null {
  const candidateIds = [...new Set([
    ...(person.annotationId ? [person.annotationId] : []),
    ...(person.annotationIds ?? []),
  ])]
  for (const annotationId of candidateIds) {
    const annotation = annotations.find((item) => item.id === annotationId)
    if (annotation && annotation.correctionKind !== DEDICATED_ROSTER_CORRECTION) {
      return annotation
    }
  }
  // Unknown or dedicated-only IDs are never updated in place. A semantic
  // correction gets its own generic annotation instead.
  return null
}

export function semanticAnnotationIdForEdit(
  person: { annotationId?: string | null; annotationIds?: string[] },
  annotations: FrameAnnotation[],
): string | null {
  return semanticAnnotationForEdit(person, annotations)?.id ?? null
}

function dedicatedRosterDecision(
  annotations: FrameAnnotation[],
  ownerIds: ReadonlySet<string>,
  ownership?: PublishedIdentityOwnership,
): string | null | 'conflict' {
  const decisions = new Set<string>()
  for (const annotation of annotations) {
    if (annotation.correctionKind !== DEDICATED_ROSTER_CORRECTION) continue
    const resolvedOwners = currentRosterCorrectionOwners(annotation, ownership)
    if (!resolvedOwners.ids.some((ownerId) => ownerIds.has(ownerId))) continue
    if (resolvedOwners.ambiguous) return 'conflict'
    const state = annotation.rosterBindingState
      ?? (annotation.externalPlayerId ? 'bound' : 'unbound')
    decisions.add(state === 'bound'
      ? `bound:${annotation.externalPlayerId ?? ''}`
      : 'unbound')
  }
  if (decisions.size > 1) return 'conflict'
  return decisions.values().next().value ?? null
}

function currentRosterCorrectionOwners(
  annotation: FrameAnnotation,
  ownership?: PublishedIdentityOwnership,
): { ids: string[]; ambiguous: boolean } {
  const observationMatches = (observation: TrackObservation, observationId: string) => (
    observation.observationId === observationId || observation.id === observationId
  )
  const publishedOwners = (predicate: (observation: TrackObservation) => boolean) => {
    const ids = new Set<string>()
    for (const person of ownership?.canonicalPeople ?? []) {
      if (person.observations?.some(predicate)) ids.add(person.canonicalPersonId)
    }
    for (const track of ownership?.tracks ?? []) {
      if (track.observations?.some(predicate)) {
        ids.add(track.canonicalPersonId || track.id)
      }
    }
    return [...ids]
  }

  // Published observation anchors and annotation references are one strong
  // ownership tier. Conflicting strong owners fail closed instead of letting
  // a stale persisted canonicalPersonId choose a destructive action.
  const strongOwners = new Set<string>()
  if (annotation.targetObservationId) {
    for (const ownerId of publishedOwners(
      (observation) => observationMatches(observation, annotation.targetObservationId as string),
    )) strongOwners.add(ownerId)
  }

  for (const ownerId of publishedOwners(
    (observation) => (
      observation.annotationId === annotation.id
      || observation.annotationIds?.includes(annotation.id) === true
    ),
  )) strongOwners.add(ownerId)
  for (const person of ownership?.canonicalPeople ?? []) {
    if (person.annotationIds?.includes(annotation.id)) {
      strongOwners.add(person.canonicalPersonId)
    }
  }
  for (const track of ownership?.tracks ?? []) {
    if (track.annotationIds?.includes(annotation.id)) {
      strongOwners.add(track.canonicalPersonId || track.id)
    }
  }
  if (strongOwners.size) {
    return {
      ids: [...strongOwners],
      ambiguous: strongOwners.size > 1,
    }
  }

  const snapshot = annotation.targetObservation
  if (snapshot?.bbox && snapshot.frameIndex !== undefined) {
    const intersectionOverUnion = (
      left: TrackObservation['bbox'],
      right: TrackObservation['bbox'],
    ) => {
      const leftRight = left.x + left.width
      const leftBottom = left.y + left.height
      const rightRight = right.x + right.width
      const rightBottom = right.y + right.height
      const width = Math.max(0, Math.min(leftRight, rightRight) - Math.max(left.x, right.x))
      const height = Math.max(0, Math.min(leftBottom, rightBottom) - Math.max(left.y, right.y))
      const intersection = width * height
      const union = left.width * left.height + right.width * right.height - intersection
      return union > 0 ? intersection / union : 0
    }
    const geometricOwners = publishedOwners((observation) => (
      observation.frameIndex === snapshot.frameIndex
      && Math.abs(observation.sceneTime - snapshot.sceneTime) <= 0.08
      && intersectionOverUnion(observation.bbox, snapshot.bbox) >= 0.75
    ))
    if (geometricOwners.length) {
      return {
        ids: geometricOwners,
        ambiguous: geometricOwners.length > 1,
      }
    }
  }

  const fallback = [annotation.canonicalPersonId]
    .filter((id): id is string => Boolean(id))
  return { ids: [...new Set(fallback)], ambiguous: false }
}

export function dedicatedRosterMergeCompatible(
  annotations: FrameAnnotation[],
  sourceOwnerIds: readonly (string | null | undefined)[],
  targetOwnerIds: readonly (string | null | undefined)[],
  ownership?: PublishedIdentityOwnership,
): boolean {
  const source = dedicatedRosterDecision(
    annotations,
    new Set(sourceOwnerIds.filter((id): id is string => Boolean(id))),
    ownership,
  )
  const target = dedicatedRosterDecision(
    annotations,
    new Set(targetOwnerIds.filter((id): id is string => Boolean(id))),
    ownership,
  )
  if (source === 'conflict' || target === 'conflict') return false
  return source === null || target === null || source === target
}

export type DedicatedRosterBindingState = 'bound' | 'unbound' | 'conflict' | null

export function dedicatedRosterBindingStateForOwner(
  annotations: FrameAnnotation[],
  ownerIds: readonly (string | null | undefined)[],
  ownership?: PublishedIdentityOwnership,
): DedicatedRosterBindingState {
  const decision = dedicatedRosterDecision(
    annotations,
    new Set(ownerIds.filter((id): id is string => Boolean(id))),
    ownership,
  )
  if (decision === null || decision === 'unbound' || decision === 'conflict') return decision
  return 'bound'
}

export function hasActiveDedicatedUnbindForOwner(
  annotations: FrameAnnotation[],
  ownerIds: readonly (string | null | undefined)[],
  ownership?: PublishedIdentityOwnership,
): boolean {
  const requestedOwners = new Set(
    ownerIds.filter((id): id is string => Boolean(id)),
  )
  const owned = annotations.filter((annotation) => {
    if (annotation.correctionKind !== DEDICATED_ROSTER_CORRECTION) return false
    const resolved = currentRosterCorrectionOwners(annotation, ownership)
    return !resolved.ambiguous
      && resolved.ids.some((ownerId) => requestedOwners.has(ownerId))
  })
  return owned.length === 1 && owned[0].rosterBindingState === 'unbound'
}

export function confirmedRosterBindingsConflict(
  sourceExternalPlayerId: string | null | undefined,
  targetExternalPlayerId: string | null | undefined,
): boolean {
  const source = sourceExternalPlayerId?.trim() || null
  const target = targetExternalPlayerId?.trim() || null
  return source !== null && target !== null && source !== target
}

export function annotationIdentityAction(annotation: Pick<FrameAnnotation, 'kind' | 'action'>): FrameIdentityAction {
  if (annotation.kind === 'ignore') return 'exclude'
  return annotation.action ?? 'confirm'
}

export function identitySplitRangeIsValid(input: {
  duration: number
  canonicalPersonId: string | null
  targetObservationId: string | null
  rangeStart: number | null
  rangeEnd: number | null
  targetTime?: number | null
}) {
  const { duration, canonicalPersonId, targetObservationId, rangeStart, rangeEnd, targetTime } = input
  if (
    !canonicalPersonId
    || !targetObservationId
    || rangeStart === null
    || rangeEnd === null
    || !Number.isFinite(rangeStart)
    || !Number.isFinite(rangeEnd)
    || rangeStart < 0
    || rangeEnd > duration
    || rangeEnd <= rangeStart
  ) return false
  return targetTime === null
    || targetTime === undefined
    || (rangeStart <= targetTime && targetTime < rangeEnd)
}

export function identitySplitObservationCounts(
  observations: TrackObservation[],
  rangeStart: number,
  rangeEnd: number,
  fallback?: { affectedObservationCount: number; remainingObservationCount: number } | null,
) {
  if (!observations.length) {
    return fallback
      ? {
          affected: fallback.affectedObservationCount,
          remaining: fallback.remainingObservationCount,
        }
      : { affected: null, remaining: null }
  }
  const affected = observations.filter((observation) => (
    observation.sceneTime >= rangeStart && observation.sceneTime < rangeEnd
  )).length
  return { affected, remaining: observations.length - affected }
}

export function wouldCreateIdentityMergeCycle(
  annotations: FrameAnnotation[],
  sourceAnnotationId: string | null,
  targetId: string,
) {
  if (!sourceAnnotationId) return false
  const byId = new Map(annotations.map((annotation) => [annotation.id, annotation]))
  const visited = new Set<string>()
  let current: string | null = sourceAnnotationId
  while (current) {
    if (visited.has(current)) return true
    visited.add(current)
    const annotation: FrameAnnotation | undefined = byId.get(current)
    if (current === sourceAnnotationId) {
      current = targetId
    } else if (annotation && annotationIdentityAction(annotation) === 'merge') {
      current = annotation.mergeTargetId ?? null
    } else {
      current = null
    }
  }
  return false
}

export function buildIdentityMergeTargets(
  tracks: Track[],
  annotations: FrameAnnotation[],
  sourceAnnotationId: string | null,
  sourceTrackId: string | null,
): IdentityMergeTarget[] {
  const result: IdentityMergeTarget[] = []
  const ids = new Set<string>()
  const identityExcludedTrackIds = new Set(
    annotations
      .filter((annotation) => (
        annotation.id !== sourceAnnotationId
        && annotationIdentityAction(annotation) === 'exclude'
        && annotation.scope === 'identity'
        && annotation.sourceTrackId
      ))
      .map((annotation) => annotation.sourceTrackId as string),
  )
  for (const track of tracks) {
    if (
      !track.id
      || track.id === sourceTrackId
      || identityExcludedTrackIds.has(track.id)
      || ids.has(track.id)
    ) continue
    ids.add(track.id)
    result.push({ id: track.id, label: track.label || track.id, type: 'track' })
  }
  for (const annotation of annotations) {
    if (
      !annotation.id
      || annotation.id === sourceAnnotationId
      || annotation.correctionKind === DEDICATED_ROSTER_CORRECTION
      || ['exclude', 'split'].includes(annotationIdentityAction(annotation))
      || ids.has(annotation.id)
      || wouldCreateIdentityMergeCycle(annotations, sourceAnnotationId, annotation.id)
    ) continue
    ids.add(annotation.id)
    result.push({
      id: annotation.id,
      label: annotation.label || `Person at ${annotation.sceneTime.toFixed(2)}s`,
      type: 'annotation',
    })
  }
  return result
}
