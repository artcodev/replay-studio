import { computed, type Ref, type ShallowRef } from 'vue'
import { annotationIdentityAction } from '../lib/identityCorrections'
import {
  buildPathTrackingSegments,
  pathTrackingOptionsForSubject,
  pathTrackingPoints,
} from '../lib/pathTracking'
import type { CanonicalPerson } from '../types/identity'
import type { Keyframe, Track } from '../types/tracking'
import type { SceneDocument } from '../types/scene'

type EditorObjectSelectionOptions = {
  scene: ShallowRef<SceneDocument | null>
  selectedTrackId: Ref<string | null>
  selectedCanonicalPersonId: Ref<string | null>
  trackQuery: Ref<string>
  ballSelected: Ref<boolean>
}

/** Read-only editor selection projections shared by sidebar, stage, and inspector. */
export function useEditorObjectSelection(options: EditorObjectSelectionOptions) {
  const selectedTrack = computed<Track | null>(() => {
    if (!options.scene.value || !options.selectedTrackId.value) return null
    return options.scene.value.payload.tracks.find(
      (track) => track.id === options.selectedTrackId.value,
    ) ?? null
  })

  const selectedCanonicalPerson = computed<CanonicalPerson | null>(() => {
    const canonicalPersonId = options.selectedCanonicalPersonId.value
      ?? selectedTrack.value?.canonicalPersonId
    if (!options.scene.value || !canonicalPersonId) return null
    return options.scene.value.payload.canonicalPeople?.find(
      (person) => person.canonicalPersonId === canonicalPersonId,
    ) ?? null
  })

  const selectedActionActorId = computed<string | null>(() => {
    if (options.ballSelected.value) return null
    return selectedCanonicalPerson.value?.canonicalPersonId
      ?? selectedTrack.value?.canonicalPersonId
      ?? null
  })
  const selectedActionActorLabel = computed(() => (
    selectedCanonicalPerson.value?.displayName
      ?? selectedTrack.value?.label
      ?? 'Selected player'
  ))

  const selectedPathSubject = computed(() => {
    if (options.ballSelected.value) {
      const keyframes = options.scene.value?.payload.ball.keyframes ?? []
      return {
        kind: 'ball' as const,
        label: 'Match ball',
        color: '#5ee7ff',
        sampleCount: pathTrackingPoints(keyframes).length,
      }
    }
    const track = selectedTrack.value
    if (!track) return null
    return {
      kind: 'player' as const,
      label: track.label,
      color: track.color,
      sampleCount: pathTrackingPoints(track.keyframes).length,
    }
  })
  const selectedPathKeyframes = computed<Keyframe[]>(() => (
    options.ballSelected.value
      ? options.scene.value?.payload.ball.keyframes ?? []
      : selectedTrack.value?.keyframes ?? []
  ))
  const selectedPathSegments = computed(() => {
    const kind = selectedPathSubject.value?.kind
    return kind
      ? buildPathTrackingSegments(
        selectedPathKeyframes.value,
        pathTrackingOptionsForSubject(kind),
      )
      : []
  })
  const unavailablePathSubjectLabel = computed(() => (
    !options.ballSelected.value && !selectedTrack.value
      ? selectedCanonicalPerson.value?.displayName ?? null
      : null
  ))

  const filteredTracks = computed(() => {
    const tracks = options.scene.value?.payload.tracks ?? []
    const query = options.trackQuery.value.trim().toLowerCase()
    if (!query) return tracks
    return tracks.filter((track) => [
      track.label,
      track.id,
      track.number,
      track.teamId,
      track.externalPlayerId,
    ].some((value) => String(value ?? '').toLowerCase().includes(query)))
  })
  const canonicalPeopleWithoutRender = computed(() => {
    const rendered = new Set(
      (options.scene.value?.payload.tracks ?? [])
        .map((track) => track.canonicalPersonId)
        .filter((id): id is string => Boolean(id)),
    )
    return (options.scene.value?.payload.canonicalPeople ?? []).filter(
      (person) => !rendered.has(person.canonicalPersonId) && person.identityStatus !== 'excluded',
    )
  })
  const filteredCanonicalPeopleWithoutRender = computed(() => {
    const query = options.trackQuery.value.trim().toLowerCase()
    if (!query) return canonicalPeopleWithoutRender.value
    return canonicalPeopleWithoutRender.value.filter((person) => [
      person.displayName,
      person.canonicalPersonId,
      person.jerseyNumber,
      person.teamId,
      person.externalPlayerId,
    ].some((value) => String(value ?? '').toLowerCase().includes(query)))
  })
  const ballMatchesTrackQuery = computed(() => {
    const query = options.trackQuery.value.trim().toLowerCase()
    return !query || 'match ball'.includes(query)
  })

  const reconstructionPreviewScene = computed<SceneDocument | null>(() => {
    const current = options.scene.value
    if (!current) return null
    const hiddenTrackIds = new Set(
      (current.payload.videoAsset?.reconstruction?.frameAnnotations ?? [])
        .filter((annotation) => (
          annotation.scope === 'identity'
          && ['exclude', 'merge'].includes(annotationIdentityAction(annotation))
          && annotation.sourceTrackId
        ))
        .map((annotation) => annotation.sourceTrackId as string),
    )
    if (!hiddenTrackIds.size) return current
    return {
      ...current,
      payload: {
        ...current.payload,
        tracks: current.payload.tracks.filter((track) => !hiddenTrackIds.has(track.id)),
      },
    }
  })

  return {
    selectedTrack,
    selectedCanonicalPerson,
    selectedActionActorId,
    selectedActionActorLabel,
    selectedPathSubject,
    selectedPathKeyframes,
    selectedPathSegments,
    unavailablePathSubjectLabel,
    filteredTracks,
    canonicalPeopleWithoutRender,
    filteredCanonicalPeopleWithoutRender,
    ballMatchesTrackQuery,
    reconstructionPreviewScene,
  }
}
