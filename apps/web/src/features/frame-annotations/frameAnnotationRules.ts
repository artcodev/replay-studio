import {
  buildIdentityMergeTargets,
  confirmedRosterBindingsConflict,
  dedicatedRosterMergeCompatible,
  identitySplitObservationCounts,
  identitySplitRangeIsValid,
} from '../../lib/identityCorrections'
import type { CanonicalPerson } from '../../types/identity'
import type { SceneDocument } from '../../types/scene'
import type { FrameAnnotationDraft } from './frameAnnotationDraft'

type CanonicalLookup = (id: string | null | undefined) => CanonicalPerson | null

export function buildFrameIdentityMergeTargets(
  scene: SceneDocument,
  draft: FrameAnnotationDraft,
  canonicalPersonById: CanonicalLookup,
) {
  const annotations = scene.payload.videoAsset?.reconstruction?.frameAnnotations ?? []
  const publishedOwnership = {
    canonicalPeople: scene.payload.canonicalPeople,
    tracks: scene.payload.tracks,
  }
  const sourceTrack = scene.payload.tracks.find((track) => track.id === draft.sourceTrackId)
  const sourceOwnerIds = [draft.canonicalPersonId, draft.sourceTrackId, sourceTrack?.canonicalPersonId]
  const sourceExternalPlayerId = draft.externalPlayerId
    ?? canonicalPersonById(draft.canonicalPersonId)?.externalPlayerId
    ?? sourceTrack?.externalPlayerId
    ?? null
  const canonicalTargets = (scene.payload.canonicalPeople ?? [])
    .filter((person) => (
      person.identityStatus !== 'excluded'
      && person.canonicalPersonId !== draft.canonicalPersonId
      && dedicatedRosterMergeCompatible(
        annotations,
        sourceOwnerIds,
        [person.canonicalPersonId],
        publishedOwnership,
      )
      && !confirmedRosterBindingsConflict(sourceExternalPlayerId, person.externalPlayerId)
    ))
    .map((person) => ({
      id: person.canonicalPersonId,
      label: person.displayName || person.canonicalPersonId,
      type: 'canonical' as const,
    }))
  const canonicalIds = new Set(canonicalTargets.map((target) => target.id))
  const otherTargets = buildIdentityMergeTargets(
    scene.payload.tracks,
    annotations,
    draft.annotationId,
    draft.sourceTrackId,
  ).filter((target) => {
    if (target.type === 'track') {
      const track = scene.payload.tracks.find((item) => item.id === target.id)
      return (
        (!track?.canonicalPersonId || !canonicalIds.has(track.canonicalPersonId))
        && dedicatedRosterMergeCompatible(
          annotations,
          sourceOwnerIds,
          [track?.id, track?.canonicalPersonId],
          publishedOwnership,
        )
        && !confirmedRosterBindingsConflict(sourceExternalPlayerId, track?.externalPlayerId)
      )
    }
    const annotation = annotations.find((item) => item.id === target.id)
    const annotationTrack = scene.payload.tracks.find(
      (track) => track.id === annotation?.sourceTrackId,
    )
    return dedicatedRosterMergeCompatible(
      annotations,
      sourceOwnerIds,
      [
        annotation?.id,
        annotation?.canonicalPersonId,
        annotation?.sourceTrackId,
        annotationTrack?.canonicalPersonId,
      ],
      publishedOwnership,
    ) && !confirmedRosterBindingsConflict(
      sourceExternalPlayerId,
      annotation?.externalPlayerId,
    )
  })
  return [...canonicalTargets, ...otherTargets]
}

export function frameIdentitySaveIsDisabled(
  scene: SceneDocument | null,
  draft: FrameAnnotationDraft | null,
  activeSceneTime: number | null,
  mergeTargetIds: ReadonlySet<string>,
) {
  const splitRangeInvalid = draft?.action === 'split' && !identitySplitRangeIsValid({
    duration: scene?.duration ?? 0,
    canonicalPersonId: draft.canonicalPersonId,
    targetObservationId: draft.targetObservationId,
    rangeStart: draft.rangeStart,
    rangeEnd: draft.rangeEnd,
    targetTime: activeSceneTime,
  })
  return !draft
    || splitRangeInvalid
    || (draft.action === 'merge' && !mergeTargetIds.has(draft.mergeTargetId ?? ''))
    || (
      draft.action === 'exclude'
      && draft.scope === 'identity'
      && !draft.canonicalPersonId
      && !draft.sourceTrackId
    )
    || (draft.action !== 'exclude' && draft.kind === 'ignore')
}

export function buildFrameIdentitySplitPreview(
  draft: FrameAnnotationDraft | null,
  activeSceneTime: number | null,
  canonicalPersonById: CanonicalLookup,
) {
  if (draft?.action !== 'split' || draft.rangeStart === null || draft.rangeEnd === null) return null
  const identity = canonicalPersonById(draft.canonicalPersonId)
  const observations = identity?.observations ?? []
  const target = observations.find((observation) => (
    (observation.observationId ?? observation.id) === draft.targetObservationId
  ))
  const counts = identitySplitObservationCounts(
    observations,
    draft.rangeStart,
    draft.rangeEnd,
    draft.affectedPreview,
  )
  return {
    identityLabel: identity?.displayName || draft.canonicalPersonId || 'Unknown identity',
    targetTime: target?.sceneTime ?? activeSceneTime,
    affected: counts.affected,
    remaining: counts.remaining,
  }
}
