import type { Keyframe, Track } from '../types'

export type PlayerVisualKind = 'player-model' | 'player-label'

/**
 * Accepted actors remain renderable for the whole scene. Confidence describes
 * evidence quality; it must not make an inferred actor pop in or out.
 */
export function shouldRenderActor(track: Pick<Track, 'keyframes'>): boolean {
  return track.keyframes.length > 0
}

export function shouldRenderPlayerVisual(
  kind: PlayerVisualKind,
  options: { showModels: boolean; showLabels: boolean },
): boolean {
  return kind === 'player-label' ? options.showLabels : options.showModels
}

export function shouldRenderBall(
  enabled: boolean,
  keyframes: Pick<Keyframe, 't'>[],
  currentTime: number,
  confidence: number,
): boolean {
  if (!enabled || keyframes.length === 0 || confidence <= 0.12) return false
  return currentTime >= keyframes[0].t && currentTime <= keyframes[keyframes.length - 1].t
}
