import type { Keyframe } from '../types'
import { interpolateKeyframes } from './interpolate'

export type PathEvidence = 'observed' | 'inferred'
export type PathTrackingSubjectKind = 'player' | 'ball'

export const PLAYER_PATH_MAX_SPEED_METRES_PER_SECOND = 14
export const BALL_PATH_MAX_SPEED_METRES_PER_SECOND = 60
export const PATH_MAX_OBSERVED_GAP_SECONDS = 1

export type PathTrackingOptions = {
  maxSpeedMetresPerSecond?: number
  maxObservedGapSeconds?: number
}

export function pathTrackingOptionsForSubject(
  kind: PathTrackingSubjectKind,
): Required<PathTrackingOptions> {
  return {
    maxSpeedMetresPerSecond: kind === 'ball'
      ? BALL_PATH_MAX_SPEED_METRES_PER_SECOND
      : PLAYER_PATH_MAX_SPEED_METRES_PER_SECOND,
    maxObservedGapSeconds: PATH_MAX_OBSERVED_GAP_SECONDS,
  }
}

export type PathTrackingPoint = {
  t: number
  x: number
  y?: number
  z: number
  evidence: PathEvidence
  keyframe: Keyframe
}

/**
 * A continuous run of path edges with the same evidence class.
 * Boundary points are intentionally shared by adjacent segments so the
 * rendered path never develops a visual gap when evidence changes.
 */
export type PathTrackingSegment = {
  evidence: PathEvidence
  points: PathTrackingPoint[]
}

export function keyframePathEvidence(keyframe: Keyframe): PathEvidence {
  // Conflicting metadata is a QA failure, not permission to draw a solid
  // measured path. Any explicit latent marker therefore wins conservatively.
  if (
    keyframe.observed === false
    || keyframe.state === 'inferred'
    || keyframe.state === 'occluded'
    || (keyframe.presenceState !== undefined && keyframe.presenceState !== 'observed')
  ) return 'inferred'

  const uncertainty = keyframe.positionUncertaintyMetres
    ?? keyframe.projection?.uncertaintyMetres
  if (typeof uncertainty === 'number' && Number.isFinite(uncertainty) && uncertainty > 3) {
    return 'inferred'
  }

  if (
    keyframe.observed === true
    || keyframe.state === 'observed'
    || keyframe.presenceState === 'observed'
  ) return 'observed'

  // Older scenes predate explicit evidence metadata. Treating their supplied
  // keyframes as observed preserves the historical rendering contract while
  // every newly reconstructed scene remains explicitly classified.
  return 'observed'
}

function finiteKeyframe(keyframe: Keyframe) {
  return Number.isFinite(keyframe.t)
    && Number.isFinite(keyframe.x)
    && Number.isFinite(keyframe.z)
    && (keyframe.y === undefined || Number.isFinite(keyframe.y))
}

function orderedKeyframeSequence(keyframes: Keyframe[]): Array<PathTrackingPoint | null> {
  const keyframesByTime = new Map<number, Keyframe>()
  keyframes.forEach((keyframe) => {
    if (Number.isFinite(keyframe.t)) keyframesByTime.set(keyframe.t, keyframe)
  })
  return [...keyframesByTime.values()]
    .sort((left, right) => left.t - right.t)
    .map((keyframe) => finiteKeyframe(keyframe) ? {
      t: keyframe.t,
      x: keyframe.x,
      y: keyframe.y,
      z: keyframe.z,
      evidence: keyframePathEvidence(keyframe),
      keyframe,
    } : null)
}

/** Return finite, time-ordered, duplicate-free samples used by every surface. */
export function pathTrackingPoints(keyframes: Keyframe[]): PathTrackingPoint[] {
  return orderedKeyframeSequence(keyframes)
    .filter((point): point is PathTrackingPoint => point !== null)
}

function edgeIsContinuous(
  left: PathTrackingPoint,
  right: PathTrackingPoint,
  maxSpeedMetresPerSecond: number,
) {
  const duration = right.t - left.t
  if (!Number.isFinite(duration) || duration <= 0) return false
  const dx = right.x - left.x
  const dy = (right.y ?? 0) - (left.y ?? 0)
  const dz = right.z - left.z
  const speed = Math.hypot(dx, dy, dz) / duration
  return Number.isFinite(speed) && speed <= maxSpeedMetresPerSecond
}

/** Build whole-highlight world-space sections for a player or the ball. */
export function buildPathTrackingSegments(
  keyframes: Keyframe[],
  options: PathTrackingOptions = {},
): PathTrackingSegment[] {
  const sequence = orderedKeyframeSequence(keyframes)
  const maxSpeed = options.maxSpeedMetresPerSecond
    ?? BALL_PATH_MAX_SPEED_METRES_PER_SECOND
  const maxObservedGap = options.maxObservedGapSeconds
    ?? PATH_MAX_OBSERVED_GAP_SECONDS
  const result: PathTrackingSegment[] = []
  let left: PathTrackingPoint | null = null
  let activeIndex = -1
  for (const right of sequence) {
    if (!right) {
      // A timestamped corrupt sample is a real evidence barrier. Silently
      // deleting it and joining its neighbours would invent a measured edge.
      left = null
      activeIndex = -1
      continue
    }
    if (!left) {
      left = right
      continue
    }
    if (!edgeIsContinuous(left, right, maxSpeed)) {
      // Teleports are normally calibration/identity failures. Start a new run
      // at the destination instead of drawing a false bridge.
      left = right
      activeIndex = -1
      continue
    }
    const evidence: PathEvidence = left.evidence === 'observed'
      && right.evidence === 'observed'
      && right.t - left.t <= maxObservedGap
      ? 'observed'
      : 'inferred'
    const active = activeIndex >= 0 ? result[activeIndex] : undefined
    if (active?.evidence === evidence) {
      active.points.push(right)
    } else {
      activeIndex = result.push({ evidence, points: [left, right] }) - 1
    }
    left = right
  }
  return result
}

/** Interpolate only inside a validated continuous run; never cross a barrier. */
export function interpolatePathTrackingSegments(
  segments: PathTrackingSegment[],
  time: number,
): Keyframe | null {
  if (!Number.isFinite(time)) return null
  const segment = segments.find((candidate) => {
    const start = candidate.points[0]?.t
    const end = candidate.points[candidate.points.length - 1]?.t
    return start !== undefined && end !== undefined && time >= start && time <= end
  })
  return segment
    ? interpolateKeyframes(segment.points.map((point) => point.keyframe), time)
    : null
}
