import type { SceneDocument } from '../../../types/scene'

export type SceneActorSelection = {
  trackId: string | null
  canonicalPersonId: string | null
}

/** Preserve only an actor the user had already selected and the new Scene still owns. */
export function preserveSceneActorSelection(
  scene: SceneDocument,
  selection: SceneActorSelection,
): SceneActorSelection {
  const tracks = scene.payload.tracks
  const canonicalPeople = scene.payload.canonicalPeople ?? []
  const canonicalPerson = selection.canonicalPersonId
    ? canonicalPeople.find(
        (person) => person.canonicalPersonId === selection.canonicalPersonId,
      ) ?? null
    : null

  if (canonicalPerson) {
    const selectedTrack = selection.trackId
      ? tracks.find((track) => track.id === selection.trackId) ?? null
      : null
    const renderTrack = selectedTrack?.canonicalPersonId === canonicalPerson.canonicalPersonId
      ? selectedTrack
      : tracks.find(
          (track) => track.canonicalPersonId === canonicalPerson.canonicalPersonId,
        ) ?? null
    return {
      trackId: renderTrack?.id ?? null,
      canonicalPersonId: canonicalPerson.canonicalPersonId,
    }
  }

  const selectedTrack = selection.trackId
    ? tracks.find((track) => track.id === selection.trackId) ?? null
    : null
  if (!selectedTrack) return { trackId: null, canonicalPersonId: null }

  const trackCanonicalPersonId = selectedTrack.canonicalPersonId
  const trackCanonicalPerson = trackCanonicalPersonId
    ? canonicalPeople.find(
        (person) => person.canonicalPersonId === trackCanonicalPersonId,
      ) ?? null
    : null
  return {
    trackId: selectedTrack.id,
    canonicalPersonId: trackCanonicalPerson?.canonicalPersonId ?? null,
  }
}
