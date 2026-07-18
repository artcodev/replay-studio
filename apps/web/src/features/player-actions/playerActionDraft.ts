import {
  defaultPlayerActionDuration,
  defaultPlayerActionKeypointKind,
} from '../../lib/playerActions'
import type { PlayerAction, PlayerActionType } from '../../types/playerActions'

export function roundActionTime(time: number) {
  return Number(time.toFixed(3))
}

export function createPlayerActionId() {
  const uuid = globalThis.crypto?.randomUUID?.()
  return `action-${uuid ?? `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`}`
}

export function buildManualPlayerAction(
  canonicalPersonId: string,
  sceneDuration: number,
  playheadTime: number,
  type: PlayerActionType = 'pass',
): PlayerAction | null {
  if (sceneDuration <= 0.001) return null
  const intervalDuration = Math.min(defaultPlayerActionDuration(type), sceneDuration)
  const keypointTime = Math.max(0, Math.min(sceneDuration, playheadTime))
  let startTime = Math.max(0, keypointTime - intervalDuration * 0.42)
  let endTime = startTime + intervalDuration
  if (endTime > sceneDuration) {
    endTime = sceneDuration
    startTime = Math.max(0, endTime - intervalDuration)
  }
  startTime = roundActionTime(startTime)
  endTime = roundActionTime(endTime)
  if (endTime <= startTime) endTime = Math.min(sceneDuration, startTime + 0.001)
  if (endTime <= startTime) return null
  return {
    id: createPlayerActionId(),
    canonicalPersonId,
    type,
    startTime,
    endTime,
    keypoints: [{
      kind: defaultPlayerActionKeypointKind(type),
      time: roundActionTime(Math.max(startTime, Math.min(endTime, keypointTime))),
    }],
    confidence: 1,
    status: 'confirmed',
    source: 'manual',
  }
}
