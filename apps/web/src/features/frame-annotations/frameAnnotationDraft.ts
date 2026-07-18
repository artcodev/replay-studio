import { annotationIdentityAction, semanticAnnotationForEdit } from '../../lib/identityCorrections'
import type { FrameAnnotationWrite } from '../../lib/api/frameAnalysis'
import type {
  FrameAnalysis,
  FrameAnnotation,
  FrameAnnotationKind,
  FrameIdentityAction,
  FrameIdentityScope,
} from '../../types/analysis'
import type { Track } from '../../types/tracking'

export type FrameAnnotationDraft = {
  annotationId: string | null
  bbox: { x: number; y: number; width: number; height: number }
  kind: FrameAnnotationKind
  label: string
  externalPlayerId: string | null
  action: FrameIdentityAction
  scope: FrameIdentityScope
  mergeTargetId: string | null
  sourceTrackId: string | null
  canonicalPersonId: string | null
  targetObservationId: string | null
  rangeStart: number | null
  rangeEnd: number | null
  affectedPreview: FrameAnnotation['affectedPreview']
}

export type DetectedPersonDraftContext = {
  annotations: readonly FrameAnnotation[]
  linkedTrackId: string | null
  canonicalPersonId: string | null
  sceneTime: number
  duration: number
}

function defaultKind(person: FrameAnalysis['people'][number]): FrameAnnotationKind {
  if (person.kind) return person.kind
  if (person.teamId === 'home') return 'home-player'
  if (person.teamId === 'away') return 'away-player'
  if (person.teamId === 'officials') return 'referee'
  return 'other'
}

function hasDedicatedRosterCorrection(
  person: FrameAnalysis['people'][number],
  annotations: readonly FrameAnnotation[],
  canonicalPersonId: string | null,
) {
  const personAnnotationIds = new Set([
    ...(person.annotationIds ?? []),
    ...(person.annotationId ? [person.annotationId] : []),
  ])
  return annotations.some((annotation) => (
    annotation.correctionKind === 'canonical-roster-binding-v1'
    && (
      personAnnotationIds.has(annotation.id)
      || annotation.canonicalPersonId === canonicalPersonId
    )
  ))
}

/** Project one detector observation plus saved corrections into an editable draft. */
export function frameAnnotationDraftFromPerson(
  person: FrameAnalysis['people'][number],
  context: DetectedPersonDraftContext,
): { draft: FrameAnnotationDraft; source: 'saved' | 'dedicated-roster' | 'new' } {
  const persisted = semanticAnnotationForEdit(person, [...context.annotations])
  const dedicatedRosterCorrection = hasDedicatedRosterCorrection(
    person,
    context.annotations,
    context.canonicalPersonId,
  )
  return { draft: {
    annotationId: persisted?.id ?? null,
    bbox: { ...(persisted?.bbox ?? person.bbox) },
    kind: persisted?.kind ?? defaultKind(person),
    label: persisted
      ? persisted.label ?? ''
      : dedicatedRosterCorrection
        ? ''
        : person.annotationLabel || person.displayName || person.matchedTrackLabel || '',
    externalPlayerId: null,
    action: persisted
      ? annotationIdentityAction(persisted)
      : dedicatedRosterCorrection
        ? 'confirm'
        : person.correctionAction ?? 'confirm',
    scope: persisted?.scope
      ?? (dedicatedRosterCorrection
        ? (context.linkedTrackId || context.canonicalPersonId ? 'identity' : 'observation')
        : person.correctionScope ?? (context.linkedTrackId ? 'identity' : 'observation')),
    mergeTargetId: persisted?.mergeTargetId
      ?? (dedicatedRosterCorrection ? null : person.mergeTargetId),
    sourceTrackId: persisted?.sourceTrackId ?? person.sourceTrackId ?? context.linkedTrackId,
    canonicalPersonId: persisted?.canonicalPersonId ?? context.canonicalPersonId,
    targetObservationId: persisted?.targetObservationId
      ?? person.targetObservationId
      ?? person.observationId
      ?? null,
    rangeStart: persisted?.rangeStart ?? person.rangeStart ?? context.sceneTime,
    rangeEnd: persisted?.rangeEnd ?? person.rangeEnd ?? context.duration,
    affectedPreview: persisted?.affectedPreview ?? person.affectedPreview ?? null,
  }, source: persisted ? 'saved' : dedicatedRosterCorrection ? 'dedicated-roster' : 'new' }
}

export function frameAnnotationDraftFromAnnotation(
  annotation: FrameAnnotation,
  tracks: readonly Track[],
  duration: number,
): FrameAnnotationDraft {
  const dedicatedRosterCorrection = annotation.correctionKind === 'canonical-roster-binding-v1'
  return {
    annotationId: dedicatedRosterCorrection ? null : annotation.id,
    bbox: { ...annotation.bbox },
    kind: annotation.kind,
    label: dedicatedRosterCorrection ? '' : annotation.label || '',
    externalPlayerId: null,
    action: dedicatedRosterCorrection ? 'confirm' : annotationIdentityAction(annotation),
    scope: annotation.scope ?? 'observation',
    mergeTargetId: dedicatedRosterCorrection ? null : annotation.mergeTargetId ?? null,
    sourceTrackId: annotation.sourceTrackId ?? null,
    canonicalPersonId: annotation.canonicalPersonId
      ?? tracks.find((track) => track.id === annotation.sourceTrackId)?.canonicalPersonId
      ?? null,
    targetObservationId: annotation.targetObservationId ?? null,
    rangeStart: annotation.rangeStart ?? annotation.sceneTime,
    rangeEnd: annotation.rangeEnd ?? duration,
    affectedPreview: annotation.affectedPreview ?? null,
  }
}

export function newManualFrameAnnotationDraft(
  point: { x: number; y: number },
): FrameAnnotationDraft {
  return {
    annotationId: null,
    bbox: { x: point.x, y: point.y, width: 4, height: 4 },
    kind: 'home-player',
    label: '',
    externalPlayerId: null,
    action: 'confirm',
    scope: 'identity',
    mergeTargetId: null,
    sourceTrackId: null,
    canonicalPersonId: null,
    targetObservationId: null,
    rangeStart: null,
    rangeEnd: null,
    affectedPreview: null,
  }
}

/** Normalize dependent fields after the user changes correction semantics. */
export function normalizeFrameAnnotationAction(
  draft: FrameAnnotationDraft,
  sceneTime: number | null,
  duration: number | null,
): FrameAnnotationDraft {
  const next = { ...draft }
  if (next.action !== 'exclude' && next.kind === 'ignore') {
    next.kind = 'other'
    next.label = ''
    next.externalPlayerId = null
  }
  if (next.action === 'merge') next.scope = 'identity'
  if (next.action === 'split') {
    next.scope = 'range'
    next.externalPlayerId = null
    next.rangeStart ??= sceneTime
    next.rangeEnd ??= duration
  }
  if (next.action !== 'merge') next.mergeTargetId = null
  return next
}

export function frameAnnotationWrite(
  draft: FrameAnnotationDraft,
  sceneTime: number,
): FrameAnnotationWrite {
  return {
    annotationId: draft.annotationId,
    sceneTime,
    bbox: draft.bbox,
    kind: draft.action === 'exclude' ? 'ignore' : draft.kind,
    label: ['exclude', 'split'].includes(draft.action) ? null : draft.label.trim() || null,
    externalPlayerId: null,
    action: draft.action,
    scope: draft.action === 'merge' ? 'identity' : draft.action === 'split' ? 'range' : draft.scope,
    mergeTargetId: draft.action === 'merge' ? draft.mergeTargetId : null,
    sourceTrackId: draft.sourceTrackId,
    canonicalPersonId: draft.canonicalPersonId,
    targetObservationId: draft.action === 'split' ? draft.targetObservationId : null,
    rangeStart: draft.action === 'split' ? draft.rangeStart : null,
    rangeEnd: draft.action === 'split' ? draft.rangeEnd : null,
  }
}
