import {
  linkedFrameMetricSelectionStatus,
  renderTrackForFramePerson,
  selectedFramePeople,
  videoTrackSelectionStatus,
  type VideoTrackSelectionStatus,
} from '../../lib/videoTrackSelection'
import type { FrameAnalysis } from '../../types/analysis'
import type { CanonicalPerson } from '../../types/identity'
import type { Track } from '../../types/tracking'

type FramePerson = FrameAnalysis['people'][number]

export function canonicalPersonForId(
  people: readonly CanonicalPerson[] | null | undefined,
  canonicalPersonId: string | null | undefined,
) {
  if (!canonicalPersonId) return null
  return people?.find((person) => person.canonicalPersonId === canonicalPersonId) ?? null
}

export function renderTrackForCanonicalPerson(
  tracks: readonly Track[],
  canonicalPersonId: string | null | undefined,
) {
  if (!canonicalPersonId) return null
  return tracks.find((track) => track.canonicalPersonId === canonicalPersonId) ?? null
}

export function validFrameMatchedTrackId(person: FramePerson, tracks: readonly Track[]) {
  const trackId = person.matchedTrackId
  return trackId && tracks.some((track) => track.id === trackId) ? trackId : null
}

export function canonicalIdForFramePerson(person: FramePerson, tracks: readonly Track[]) {
  const matchedTrackId = validFrameMatchedTrackId(person, tracks)
  return person.canonicalPersonId
    ?? tracks.find((track) => track.id === matchedTrackId)?.canonicalPersonId
    ?? null
}

export function framePersonDisplayLabel(
  person: FramePerson,
  tracks: readonly Track[],
  canonicalPeople: readonly CanonicalPerson[] | null | undefined,
) {
  const identity = canonicalPersonForId(canonicalPeople, canonicalIdForFramePerson(person, tracks))
  return person.annotationLabel
    || person.displayName
    || identity?.displayName
    || person.matchedTrackLabel
    || person.id
}

/** Pick the strongest visible detector observation for one canonical identity. */
export function bestFramePersonForCanonicalIdentity(
  source: FrameAnalysis | null,
  canonicalPersonId: string | null,
) {
  if (!source || !canonicalPersonId) return null
  const sourcePriority: Record<NonNullable<FramePerson['matchSource']>, number> = {
    'persisted-observation': 0,
    'manual-identity': 1,
  }
  return [...selectedFramePeople(source, canonicalPersonId)].sort((left, right) => {
    const sourceDelta = (left.matchSource ? sourcePriority[left.matchSource] : 9)
      - (right.matchSource ? sourcePriority[right.matchSource] : 9)
    if (sourceDelta) return sourceDelta
    const distanceDelta = (left.matchDistance ?? Number.POSITIVE_INFINITY)
      - (right.matchDistance ?? Number.POSITIVE_INFINITY)
    if (distanceDelta) return distanceDelta
    return right.confidence - left.confidence || left.id.localeCompare(right.id)
  })[0] ?? null
}

export function frameAnalysisSelectionStatus(options: {
  analysis: FrameAnalysis | null
  person: FramePerson | null
  tracks: readonly Track[]
  canonicalPeople: readonly CanonicalPerson[] | null | undefined
  selectedTrack: Track | null
  selectedCanonicalPersonId: string | null
  analyzing: boolean
  selectedTrackObserved: boolean | null
}): VideoTrackSelectionStatus | null {
  const canonicalId = options.person
    ? canonicalIdForFramePerson(options.person, options.tracks)
    : options.selectedCanonicalPersonId
  const identity = canonicalPersonForId(options.canonicalPeople, canonicalId)
  const linkedTrack = options.person
    ? renderTrackForFramePerson(options.person, options.tracks)
    : renderTrackForCanonicalPerson(options.tracks, canonicalId) ?? options.selectedTrack
  if (identity && !linkedTrack) {
    const visibleMatches = selectedFramePeople(options.analysis, identity.canonicalPersonId).length
    return {
      state: 'identity-only',
      label: visibleMatches ? 'Identity matched' : 'Canonical identity selected',
      detail: `${identity.displayName || identity.canonicalPersonId} · not projected in 3D`,
      matchCount: visibleMatches,
    }
  }
  if (options.person && !identity && !linkedTrack) {
    return {
      state: 'unlinked',
      label: 'Video detection selected',
      detail: `${framePersonDisplayLabel(options.person, options.tracks, options.canonicalPeople)} · identity is not resolved yet`,
      matchCount: 0,
    }
  }
  if (options.person && linkedTrack) {
    const metricStatus = linkedFrameMetricSelectionStatus(
      options.person,
      linkedTrack.label ?? options.person.matchedTrackLabel,
    )
    if (metricStatus) return metricStatus
  }
  return videoTrackSelectionStatus(
    options.analysis,
    options.selectedTrack?.id ?? null,
    options.selectedTrack?.label ?? identity?.displayName ?? null,
    {
      analyzing: options.analyzing,
      observedAtCurrentTime: options.selectedTrackObserved,
      canonicalPersonId: options.selectedCanonicalPersonId,
    },
  )
}
