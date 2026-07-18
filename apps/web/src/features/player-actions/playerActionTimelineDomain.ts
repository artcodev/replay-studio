import type { PlayerAction, PlayerActionKeypointKind, PlayerActionType } from '../../types/playerActions'

export const PLAYER_ACTION_TIME_STEP = 0.001

export const PLAYER_ACTION_KEYPOINT_KINDS: readonly PlayerActionKeypointKind[] = [
  'wind-up',
  'contact',
  'release',
  'apex',
  'impact',
  'recovery',
]

export type PlayerActionEdit =
  | { type: 'set-type'; value: PlayerActionType }
  | { type: 'set-start'; time: number }
  | { type: 'set-end'; time: number }
  | { type: 'add-keypoint'; kind?: PlayerActionKeypointKind; time?: number }
  | { type: 'update-keypoint'; index: number; kind?: PlayerActionKeypointKind; time?: number }
  | { type: 'remove-keypoint'; index: number }

export type PlayerActionLayoutItem = { action: PlayerAction; lane: number }

function finiteActionNumber(value: number, fallback = 0) {
  return Number.isFinite(value) ? value : fallback
}

export function clampPlayerActionTime(value: number, duration: number) {
  const safeDuration = Math.max(0, finiteActionNumber(duration))
  const safeValue = finiteActionNumber(value)
  const rounded = Math.round(Math.min(safeDuration, Math.max(0, safeValue)) * 1000) / 1000
  return Math.min(safeDuration, Math.max(0, rounded))
}

export function normalizePlayerAction(action: PlayerAction, duration: number): PlayerAction {
  const firstBoundary = clampPlayerActionTime(action.startTime, duration)
  const secondBoundary = clampPlayerActionTime(action.endTime, duration)
  const safeDuration = Math.max(0, finiteActionNumber(duration))
  const minimumInterval = Math.min(PLAYER_ACTION_TIME_STEP, safeDuration)
  let startTime = Math.min(firstBoundary, secondBoundary)
  let endTime = Math.max(firstBoundary, secondBoundary)
  if (endTime - startTime < minimumInterval) {
    if (startTime + minimumInterval <= safeDuration) endTime = startTime + minimumInterval
    else startTime = Math.max(0, endTime - minimumInterval)
  }
  const keypoints = action.keypoints
    .map((keypoint) => ({
      kind: keypoint.kind,
      time: Math.round(Math.min(endTime, Math.max(startTime, finiteActionNumber(keypoint.time, startTime))) * 1000) / 1000,
    }))
    .sort((first, second) => first.time - second.time || first.kind.localeCompare(second.kind))
  return {
    ...action,
    startTime,
    endTime,
    keypoints,
    confidence: Math.min(1, Math.max(0, finiteActionNumber(action.confidence))),
  }
}

export function normalizePlayerActions(actions: PlayerAction[], duration: number) {
  return actions
    .map((action) => normalizePlayerAction(action, duration))
    .sort((first, second) => first.startTime - second.startTime || first.endTime - second.endTime || first.id.localeCompare(second.id))
}

/** Pure reducer: automatic hypotheses remain review-only evidence. */
export function reducePlayerAction(action: PlayerAction, edit: PlayerActionEdit, duration: number): PlayerAction {
  const current = normalizePlayerAction(action, duration)
  if (current.source !== 'manual') return current
  if (edit.type === 'set-type') return { ...current, type: edit.value }
  if (edit.type === 'set-start') {
    const minimumInterval = Math.min(PLAYER_ACTION_TIME_STEP, Math.max(0, finiteActionNumber(duration)))
    return normalizePlayerAction({
      ...current,
      startTime: Math.min(current.endTime - minimumInterval, clampPlayerActionTime(edit.time, duration)),
    }, duration)
  }
  if (edit.type === 'set-end') {
    const minimumInterval = Math.min(PLAYER_ACTION_TIME_STEP, Math.max(0, finiteActionNumber(duration)))
    return normalizePlayerAction({
      ...current,
      endTime: Math.max(current.startTime + minimumInterval, clampPlayerActionTime(edit.time, duration)),
    }, duration)
  }
  if (edit.type === 'add-keypoint') {
    const requestedTime = edit.time ?? (current.startTime + current.endTime) / 2
    const time = Math.min(current.endTime, Math.max(current.startTime, clampPlayerActionTime(requestedTime, duration)))
    return normalizePlayerAction({
      ...current,
      keypoints: [...current.keypoints, { kind: edit.kind ?? 'contact', time }],
    }, duration)
  }
  if (edit.index < 0 || edit.index >= current.keypoints.length) return current
  if (edit.type === 'remove-keypoint') {
    return { ...current, keypoints: current.keypoints.filter((_, index) => index !== edit.index) }
  }
  const existing = current.keypoints[edit.index]
  const requestedTime = edit.time ?? existing.time
  const time = Math.round(Math.min(current.endTime, Math.max(current.startTime, finiteActionNumber(requestedTime, existing.time))) * 1000) / 1000
  return normalizePlayerAction({
    ...current,
    keypoints: current.keypoints.map((keypoint, index) => index === edit.index
      ? { kind: edit.kind ?? keypoint.kind, time }
      : keypoint),
  }, duration)
}

/** Greedy lane assignment keeps overlapping intervals readable. */
export function layoutPlayerActions(actions: PlayerAction[], duration: number): PlayerActionLayoutItem[] {
  const laneEnds: number[] = []
  return normalizePlayerActions(actions, duration).map((action) => {
    let lane = laneEnds.findIndex((endTime) => endTime <= action.startTime)
    if (lane < 0) lane = laneEnds.length
    laneEnds[lane] = action.endTime
    return { action, lane }
  })
}
