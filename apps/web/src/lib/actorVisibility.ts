import type { Keyframe, Track } from '../types/tracking'

export type PlayerVisualKind = 'player-model' | 'player-label'

/**
 * Render a person only inside the time window supported by its observations.
 *
 * Encoded segment boundaries and frame timestamps can differ by a fraction of
 * one frame, so a small tolerance keeps the first/last real frame visible
 * without inventing a scene-long prefix or suffix.
 */
export function shouldRenderActor(
  track: Pick<Track, 'keyframes'>,
  currentTime: number,
  boundaryTolerance = 0.05,
): boolean {
  if (track.keyframes.length === 0) return false
  const start = track.keyframes[0].t
  const end = track.keyframes[track.keyframes.length - 1].t
  return currentTime >= start - boundaryTolerance && currentTime <= end + boundaryTolerance
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
