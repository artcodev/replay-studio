import type { Keyframe, Track } from '../types/tracking'

export type TrackPresenceState = NonNullable<Keyframe['presenceState']>

export type TrackPresenceSnapshot = {
  state: TrackPresenceState
  label: string
  detail: string
  observed: boolean
  uncertaintyMetres: number | null
}

export type TrackPresenceSummary = {
  timelineCoverage: number | null
  observedSampleRatio: number | null
  inferredSampleRatio: number | null
  observedSpanRatio: number | null
  observationCount: number
  inferredKeyframeCount: number
}

// Reconstruction timestamps are serialised to millisecond precision. This
// tolerance only treats numerically equivalent timestamps as the same sampled
// frame; it deliberately does not turn the interval around an observation into
// detector-backed evidence.
const OBSERVED_TIMESTAMP_TOLERANCE_SECONDS = 0.001

const statePresentation: Record<TrackPresenceState, Pick<TrackPresenceSnapshot, 'label' | 'detail' | 'observed'>> = {
  observed: {
    label: 'OBSERVED',
    detail: 'Supported by a detector/tracker observation at this timestamp.',
    observed: true,
  },
  'inferred-before-first': {
    label: 'INFERRED · BEFORE',
    detail: 'Latent position before the first confirmed observation.',
    observed: false,
  },
  'inferred-gap': {
    label: 'INFERRED · GAP',
    detail: 'Latent position through a gap between confirmed observations.',
    observed: false,
  },
  'inferred-after-last': {
    label: 'INFERRED · AFTER',
    detail: 'Latent position after the last confirmed observation.',
    observed: false,
  },
}

function finiteNumber(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function uncertaintyFor(keyframe: Keyframe): number | null {
  return finiteNumber(
    keyframe.positionUncertaintyMetres
      ?? keyframe.projection?.uncertaintyMetres,
  )
}

function nearestKeyframe(keyframes: Keyframe[], time: number): Keyframe | null {
  if (!keyframes.length) return null
  return keyframes.reduce((nearest, candidate) => (
    Math.abs(candidate.t - time) < Math.abs(nearest.t - time) ? candidate : nearest
  ))
}

function observedKeyframes(keyframes: Keyframe[]): Keyframe[] {
  const hasPresenceSemantics = keyframes.some((keyframe) => (
    keyframe.observed !== undefined || keyframe.presenceState !== undefined
  ))
  if (!hasPresenceSemantics) return keyframes

  return keyframes.filter((keyframe) => (
    keyframe.observed === true
    || (keyframe.observed !== false && keyframe.presenceState === 'observed')
  ))
}

function uncertaintyAtTime(keyframes: Keyframe[], time: number): number | null {
  if (!keyframes.length) return null
  const ordered = [...keyframes].sort((left, right) => left.t - right.t)
  const first = ordered[0]
  const last = ordered[ordered.length - 1]
  if (time <= first.t) return uncertaintyFor(first)
  if (time >= last.t) return uncertaintyFor(last)

  const rightIndex = ordered.findIndex((keyframe) => keyframe.t >= time)
  const left = ordered[Math.max(0, rightIndex - 1)]
  const right = ordered[Math.max(0, rightIndex)]
  const leftUncertainty = uncertaintyFor(left)
  const rightUncertainty = uncertaintyFor(right)
  if (leftUncertainty === null) return rightUncertainty
  if (rightUncertainty === null) return leftUncertainty
  const mix = (time - left.t) / Math.max(0.0001, right.t - left.t)
  return leftUncertainty + (rightUncertainty - leftUncertainty) * mix
}

function presenceStateAtTime(track: Track, time: number): TrackPresenceState {
  const observed = observedKeyframes(track.keyframes).sort((left, right) => left.t - right.t)
  const exactObservation = observed.find((keyframe) => (
    Math.abs(keyframe.t - time) <= OBSERVED_TIMESTAMP_TOLERANCE_SECONDS
  ))
  if (exactObservation) return 'observed'

  const metadataStart = finiteNumber(track.presence?.observedStart)
  const metadataEnd = finiteNumber(track.presence?.observedEnd)
  const observedStart = observed.length ? observed[0].t : metadataStart
  const observedEnd = observed.length ? observed[observed.length - 1].t : metadataEnd
  if (observedStart !== null && observedStart !== undefined && time < observedStart) {
    return 'inferred-before-first'
  }
  if (observedEnd !== null && observedEnd !== undefined && time > observedEnd) {
    return 'inferred-after-last'
  }
  if (observedStart !== null && observedStart !== undefined) return 'inferred-gap'

  const keyframe = nearestKeyframe(track.keyframes, time)
  return keyframe?.presenceState === 'observed'
    ? 'inferred-gap'
    : keyframe?.presenceState ?? 'inferred-gap'
}

export function trackPresenceAtTime(track: Track, time: number): TrackPresenceSnapshot {
  const state = presenceStateAtTime(track, time)
  return {
    state,
    ...statePresentation[state],
    uncertaintyMetres: uncertaintyAtTime(track.keyframes, time),
  }
}

export function trackPresenceSummary(track: Track): TrackPresenceSummary {
  const detectorBackedSamples = observedKeyframes(track.keyframes).length
  const observationCount = track.presence?.observationCount
    ?? detectorBackedSamples
  const inferredKeyframeCount = track.presence?.inferredKeyframeCount
    ?? Math.max(0, track.keyframes.length - detectorBackedSamples)
  const sampleCount = observationCount + inferredKeyframeCount
  const observedSampleRatio = sampleCount > 0 ? observationCount / sampleCount : null

  return {
    timelineCoverage: finiteNumber(track.presence?.coverage),
    observedSampleRatio,
    inferredSampleRatio: observedSampleRatio === null ? null : 1 - observedSampleRatio,
    observedSpanRatio: finiteNumber(track.presence?.observedSpanRatio),
    observationCount,
    inferredKeyframeCount,
  }
}
