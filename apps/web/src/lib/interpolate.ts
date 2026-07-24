import type { Keyframe } from '../types/tracking'

export function interpolateKeyframes(keyframes: Keyframe[], time: number): Keyframe {
  if (keyframes.length === 0) return { t: time, x: 0, y: 0, z: 0, confidence: 0 }
  if (time <= keyframes[0].t) return keyframes[0]
  if (time >= keyframes[keyframes.length - 1].t) return keyframes[keyframes.length - 1]

  let rightIndex = keyframes.findIndex((frame) => frame.t >= time)
  if (rightIndex < 1) rightIndex = 1
  const left = keyframes[rightIndex - 1]
  const right = keyframes[rightIndex]
  const span = Math.max(0.0001, right.t - left.t)
  const mix = (time - left.t) / span
  return {
    t: time,
    x: left.x + (right.x - left.x) * mix,
    y: (left.y ?? 0) + ((right.y ?? 0) - (left.y ?? 0)) * mix,
    z: left.z + (right.z - left.z) * mix,
    confidence: left.confidence + (right.confidence - left.confidence) * mix,
  }
}

/**
 * A position is latent when the playhead sits on or between keyframes that
 * were not observed — the figure there is an identity-continuity guess, not
 * evidence, and must be rendered as an inferred position.
 */
export function isInferredAt(keyframes: Keyframe[], time: number): boolean {
  if (keyframes.length === 0) return true
  const observed = (frame: Keyframe) =>
    frame.observed !== false && frame.state !== 'inferred' && frame.state !== 'occluded'
  if (time <= keyframes[0].t) return !observed(keyframes[0])
  const last = keyframes[keyframes.length - 1]
  if (time >= last.t) return !observed(last)
  let rightIndex = keyframes.findIndex((frame) => frame.t >= time)
  if (rightIndex < 1) rightIndex = 1
  return !observed(keyframes[rightIndex - 1]) || !observed(keyframes[rightIndex])
}

export function upsertKeyframe(keyframes: Keyframe[], next: Keyframe): Keyframe[] {
  const tolerance = 0.08
  const found = keyframes.findIndex((frame) => Math.abs(frame.t - next.t) < tolerance)
  const result = [...keyframes]
  if (found >= 0) result[found] = next
  else result.push(next)
  return result.sort((a, b) => a.t - b.t)
}
