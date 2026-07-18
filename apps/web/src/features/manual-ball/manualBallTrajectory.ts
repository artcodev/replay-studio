import { interpolateKeyframes } from '../../lib/interpolate'
import type { Keyframe } from '../../types/tracking'

export const MANUAL_BALL_KEYFRAME_TOLERANCE = 0.0005

export type ManualBallPitchBounds = {
  duration: number
  pitchLength: number
  pitchWidth: number
}

function finite(value: number, fallback = 0) {
  return Number.isFinite(value) ? value : fallback
}

/** Canonicalize manually-authored samples before optimistic or remote writes. */
export function normalizeManualBallKeyframes(
  keyframes: readonly Keyframe[],
  bounds: ManualBallPitchBounds,
): Keyframe[] {
  const duration = Math.max(0, finite(bounds.duration))
  const halfLength = Math.max(0, finite(bounds.pitchLength, 105)) / 2
  const halfWidth = Math.max(0, finite(bounds.pitchWidth, 68)) / 2
  const normalized: Keyframe[] = []

  for (const frame of [...keyframes].sort((left, right) => left.t - right.t)) {
    const next: Keyframe = {
      ...frame,
      t: Number(Math.max(0, Math.min(duration, finite(frame.t))).toFixed(3)),
      x: Number(Math.max(-halfLength, Math.min(halfLength, finite(frame.x))).toFixed(2)),
      y: Number(Math.max(0.24, finite(frame.y ?? 0.24, 0.24)).toFixed(2)),
      z: Number(Math.max(-halfWidth, Math.min(halfWidth, finite(frame.z))).toFixed(2)),
      confidence: 1,
      observed: true,
      state: 'observed',
      projectionSource: 'manual',
      projection: { source: 'manual', uncertaintyMetres: 0 },
      positionUncertaintyMetres: 0,
    }
    const duplicate = normalized.findIndex(
      (item) => Math.abs(item.t - next.t) < MANUAL_BALL_KEYFRAME_TOLERANCE,
    )
    if (duplicate >= 0) normalized[duplicate] = next
    else normalized.push(next)
  }
  return normalized.sort((left, right) => left.t - right.t)
}

/** Materialize a manual sample from an explicit position or interpolated path. */
export function manualBallKeyframeAt(
  source: readonly Keyframe[],
  time: number,
  bounds: ManualBallPitchBounds,
  position?: { x: number; z: number },
): Keyframe {
  const interpolated = interpolateKeyframes([...source], time)
  return normalizeManualBallKeyframes([{
    t: time,
    x: position?.x ?? interpolated.x,
    y: position ? 0.24 : interpolated.y ?? 0.24,
    z: position?.z ?? interpolated.z,
    confidence: 1,
  }], bounds)[0]
}
