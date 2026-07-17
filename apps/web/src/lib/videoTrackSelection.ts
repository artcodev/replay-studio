export type FrameTrackMatch = {
  matchedTrackId: string | null
  canonicalPersonId?: string | null
}

export type FrameMetricMatch = FrameTrackMatch & {
  metricStatus?: 'accepted' | 'rejected' | 'unprojected' | null
  metricReason?: string | null
  positionSource?: 'observation' | 'track-inferred' | null
}

export type VideoTrackSelectionState =
  | 'checking'
  | 'visible'
  | 'uncertain'
  | 'inferred'
  | 'missing'
  | 'identity-only'
  | 'unlinked'
  | 'unchecked'

export type VideoTrackSelectionStatus = {
  state: VideoTrackSelectionState
  label: string
  detail: string
  matchCount: number
}

export type CanonicalRenderTrack = {
  id: string
  canonicalPersonId?: string | null
}

/** Resolves video → 3D using canonical identity, with legacy track id fallback. */
export function renderTrackForFramePerson<T extends CanonicalRenderTrack>(
  person: FrameTrackMatch,
  tracks: readonly T[],
): T | null {
  if (person.canonicalPersonId) {
    const canonicalMatch = tracks.find(
      (track) => track.canonicalPersonId === person.canonicalPersonId,
    )
    if (canonicalMatch) return canonicalMatch
  }
  if (!person.matchedTrackId) return null
  return tracks.find((track) => track.id === person.matchedTrackId) ?? null
}

export function selectedFramePeople<T extends FrameTrackMatch>(
  analysis: { people: readonly T[] } | null,
  trackId: string | null,
  canonicalPersonId: string | null = null,
): T[] {
  if (!analysis) return []

  // Canonical identity is authoritative. matchedTrackId only names an
  // optional render actor and can legitimately be missing (or stale) when
  // metric projection was rejected.
  if (canonicalPersonId) {
    const canonicalMatches = analysis.people.filter(
      (person) => person.canonicalPersonId === canonicalPersonId,
    )
    if (canonicalMatches.length) return canonicalMatches
  }

  if (!trackId) return []
  return analysis.people.filter((person) => person.matchedTrackId === trackId)
}

/** Distinguishes an accepted metric projection from identity-only evidence. */
export function frameMetricBadge(person: FrameMetricMatch): 'METRIC' | 'UNCERTAIN' | null {
  if (
    person.metricStatus === 'rejected'
    || person.metricStatus === 'unprojected'
    || person.positionSource === 'track-inferred'
  ) return 'UNCERTAIN'
  if (person.metricStatus === 'accepted') return 'METRIC'
  return null
}

/**
 * Returns a dedicated state only when a visible, linked observation does not
 * provide a trustworthy metric position. Identity linkage remains valid.
 */
export function linkedFrameMetricSelectionStatus(
  person: FrameMetricMatch,
  trackLabel: string | null,
): VideoTrackSelectionStatus | null {
  if ((!person.canonicalPersonId && !person.matchedTrackId) || frameMetricBadge(person) !== 'UNCERTAIN') return null
  const detail = trackLabel || person.canonicalPersonId || person.matchedTrackId || 'Selected person'
  const reason = person.metricReason
    || (person.positionSource === 'track-inferred'
      ? 'using a track-inferred 3D position'
      : 'no accepted metric projection')
  return {
    state: 'uncertain',
    label: 'Visible · 3D position uncertain',
    detail: `${detail} · ${reason}`,
    matchCount: 1,
  }
}

export function selectionAfterFrameAnalysis(
  selectionAtRequestStart: string | null,
  currentSelection: string | null,
  firstMatchedTrackId: string | null,
  preserveTrackId?: string,
): string | null {
  // Never let a late detector response undo a newer user click.
  if (currentSelection !== selectionAtRequestStart) return currentSelection
  // Analyze Frame inspects the current selection; it must not jump to the
  // detector's first result merely because that result is ordered first.
  return preserveTrackId ?? selectionAtRequestStart ?? firstMatchedTrackId ?? currentSelection
}

export function canonicalSelectionAfterFrameAnalysis(
  selectionAtRequestStart: string | null,
  currentSelection: string | null,
  currentTrackId: string | null,
  selectedTrackCanonicalPersonId: string | null | undefined,
  firstMatchedCanonicalPersonId: string | null,
): string | null {
  // A newer canonical-identity click is authoritative even when that identity
  // has no render track and both old/new track selections are null.
  if (currentSelection !== selectionAtRequestStart) return currentSelection
  if (selectedTrackCanonicalPersonId) return selectedTrackCanonicalPersonId
  if (selectionAtRequestStart) return selectionAtRequestStart
  // A selected legacy track deliberately has no canonical identity. Do not
  // attach the first unrelated detector result to it.
  return currentTrackId ? null : firstMatchedCanonicalPersonId
}

export function videoTrackSelectionStatus(
  analysis: { people: readonly FrameTrackMatch[] } | null,
  trackId: string | null,
  trackLabel: string | null,
  options: {
    analyzing: boolean
    observedAtCurrentTime: boolean | null
    canonicalPersonId?: string | null
  },
): VideoTrackSelectionStatus | null {
  if (!trackId && !options.canonicalPersonId) return null

  const detail = trackLabel || options.canonicalPersonId || trackId || 'Selected person'
  if (options.analyzing) {
    return {
      state: 'checking',
      label: 'Matching source frame',
      detail,
      matchCount: 0,
    }
  }

  if (!analysis) {
    return {
      state: 'unchecked',
      label: 'Frame not checked',
      detail,
      matchCount: 0,
    }
  }

  const matches = selectedFramePeople(analysis, trackId, options.canonicalPersonId)
  if (matches.length) {
    return {
      state: 'visible',
      label: 'Visible in video',
      detail: matches.length > 1 ? `${detail} · ${matches.length} detections` : detail,
      matchCount: matches.length,
    }
  }

  if (options.observedAtCurrentTime === true) {
    return {
      state: 'missing',
      label: 'No matched detection',
      detail,
      matchCount: 0,
    }
  }

  return {
    state: 'inferred',
    label: 'Inferred in 3D',
    detail: `${detail} · not visible in this frame`,
    matchCount: 0,
  }
}
