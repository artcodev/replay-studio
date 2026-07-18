import type {
  PlayerAction,
  PlayerActionKeypointKind,
  PlayerActionType,
} from '../types/playerActions'

export type PlayerActionCategory = 'locomotion' | 'ball' | 'defensive' | 'skill'

export type PlayerActionTaxonomyItem = {
  label: string
  category: PlayerActionCategory
  color: string
  defaultDurationSeconds: number
  defaultKeypointKind: PlayerActionKeypointKind
}

export const PLAYER_ACTION_CATEGORY_META: Readonly<Record<PlayerActionCategory, {
  label: string
  color: string
}>> = Object.freeze({
  locomotion: { label: 'Locomotion / body', color: '#76a9ff' },
  ball: { label: 'Ball action', color: '#ffd36a' },
  defensive: { label: 'Defensive action', color: '#ff8f63' },
  skill: { label: 'Skill / deception', color: '#b184ff' },
})

const taxonomy = {
  idle: ['Idle', 'locomotion', 1, 'recovery'],
  walk: ['Walk', 'locomotion', 1, 'contact'],
  run: ['Run', 'locomotion', 0.9, 'contact'],
  sprint: ['Sprint', 'locomotion', 0.8, 'contact'],
  turn: ['Turn', 'locomotion', 0.6, 'apex'],
  jump: ['Jump', 'locomotion', 0.8, 'apex'],
  fall: ['Fall', 'locomotion', 0.9, 'impact'],
  'get-up': ['Get up', 'locomotion', 1.2, 'recovery'],
  'first-touch': ['First touch', 'ball', 0.35, 'contact'],
  drive: ['Drive', 'ball', 0.8, 'contact'],
  pass: ['Pass', 'ball', 0.65, 'contact'],
  cross: ['Cross', 'ball', 0.8, 'contact'],
  shot: ['Shot', 'ball', 0.8, 'contact'],
  header: ['Header', 'ball', 0.6, 'contact'],
  'throw-in': ['Throw-in', 'ball', 1.1, 'release'],
  clearance: ['Clearance', 'ball', 0.7, 'contact'],
  tackle: ['Tackle', 'defensive', 0.8, 'impact'],
  'slide-tackle': ['Slide tackle', 'defensive', 1, 'impact'],
  block: ['Block', 'defensive', 0.6, 'impact'],
  interception: ['Interception', 'defensive', 0.6, 'contact'],
  feint: ['Feint', 'skill', 0.8, 'apex'],
} as const satisfies Record<
  PlayerActionType,
  readonly [string, PlayerActionCategory, number, PlayerActionKeypointKind]
>

export const PLAYER_ACTION_TYPES = Object.freeze(
  Object.keys(taxonomy) as PlayerActionType[],
)

export const PLAYER_ACTION_TAXONOMY: Readonly<Record<PlayerActionType, PlayerActionTaxonomyItem>> =
  Object.freeze(Object.fromEntries(
    PLAYER_ACTION_TYPES.map((type) => {
      const [label, category, defaultDurationSeconds, defaultKeypointKind] = taxonomy[type]
      return [type, Object.freeze({
        label,
        category,
        color: PLAYER_ACTION_CATEGORY_META[category].color,
        defaultDurationSeconds,
        defaultKeypointKind,
      })]
    }),
  ) as Record<PlayerActionType, PlayerActionTaxonomyItem>)

export function playerActionMeta(type: PlayerActionType): PlayerActionTaxonomyItem {
  return PLAYER_ACTION_TAXONOMY[type]
}

export function playerActionLabel(type: PlayerActionType): string {
  return playerActionMeta(type).label
}

export function playerActionCategory(type: PlayerActionType): PlayerActionCategory {
  return playerActionMeta(type).category
}

export function playerActionColor(type: PlayerActionType): string {
  return playerActionMeta(type).color
}

export function defaultPlayerActionDuration(type: PlayerActionType): number {
  return playerActionMeta(type).defaultDurationSeconds
}

export function defaultPlayerActionKeypointKind(type: PlayerActionType): PlayerActionKeypointKind {
  return playerActionMeta(type).defaultKeypointKind
}

function finiteOr(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback
}

function actionBounds(action: PlayerAction) {
  const start = Math.min(action.startTime, action.endTime)
  const end = Math.max(action.startTime, action.endTime)
  return { start, end, duration: Math.max(0, end - start) }
}

export function comparePlayerActions(left: PlayerAction, right: PlayerAction): number {
  const leftStart = finiteOr(left.startTime, Number.POSITIVE_INFINITY)
  const rightStart = finiteOr(right.startTime, Number.POSITIVE_INFINITY)
  return leftStart - rightStart
    || finiteOr(left.endTime, Number.POSITIVE_INFINITY) - finiteOr(right.endTime, Number.POSITIVE_INFINITY)
    || PLAYER_ACTION_TYPES.indexOf(left.type) - PLAYER_ACTION_TYPES.indexOf(right.type)
    || left.canonicalPersonId.localeCompare(right.canonicalPersonId)
    || left.id.localeCompare(right.id)
}

/** Null selection is the editor's "all actors" filter. */
export function filterPlayerActionsForActor(
  actions: readonly PlayerAction[],
  canonicalPersonId: string | null | undefined,
): PlayerAction[] {
  return actions
    .filter((action) => !canonicalPersonId || action.canonicalPersonId === canonicalPersonId)
    .slice()
    .sort(comparePlayerActions)
}

function activePriority(action: PlayerAction): number {
  // Human-authored intervals remain authoritative over automatic hypotheses;
  // confirmation is the second priority axis inside each provenance class.
  if (action.source === 'manual' && action.status === 'confirmed') return 4
  if (action.source === 'manual') return 3
  if (action.status === 'confirmed') return 2
  return 1
}

function compareActiveActions(left: PlayerAction, right: PlayerAction): number {
  const priority = activePriority(right) - activePriority(left)
  if (priority) return priority
  const confidence = finiteOr(right.confidence, -1) - finiteOr(left.confidence, -1)
  if (confidence) return confidence
  const duration = actionBounds(left).duration - actionBounds(right).duration
  if (duration) return duration
  const start = finiteOr(right.startTime, Number.NEGATIVE_INFINITY)
    - finiteOr(left.startTime, Number.NEGATIVE_INFINITY)
  if (start) return start
  return left.id.localeCompare(right.id)
}

/** Deterministically choose one non-rejected interval at the shared playhead. */
export function selectActivePlayerAction(
  actions: readonly PlayerAction[],
  playheadTime: number,
  canonicalPersonId: string | null | undefined = null,
): PlayerAction | null {
  if (!Number.isFinite(playheadTime)) return null
  return filterPlayerActionsForActor(actions, canonicalPersonId)
    .filter((action) => {
      if (
        action.status === 'rejected'
        || !Number.isFinite(action.startTime)
        || !Number.isFinite(action.endTime)
      ) return false
      const { start, end } = actionBounds(action)
      return playheadTime >= start && playheadTime <= end
    })
    .sort(compareActiveActions)[0] ?? null
}

const keypointOrder: Readonly<Record<PlayerActionKeypointKind, number>> = Object.freeze({
  'wind-up': 0,
  contact: 1,
  release: 2,
  apex: 3,
  impact: 4,
  recovery: 5,
})

export type PlayerActionPlaybackKeypoint = {
  keypoint: PlayerAction['keypoints'][number]
  kind: PlayerActionKeypointKind
  time: number
  phase: number
  offsetSeconds: number
  distanceSeconds: number
}

export type PlayerActionPlaybackState = {
  action: PlayerAction
  phase: number
  durationSeconds: number
  elapsedSeconds: number
  nearestKeypoint: PlayerActionPlaybackKeypoint | null
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value))
}

/** Produce seek-safe, renderer-neutral timing data; this does not select a clip. */
export function playerActionPlaybackState(
  action: PlayerAction,
  playheadTime: number,
): PlayerActionPlaybackState | null {
  if (
    !Number.isFinite(playheadTime)
    || !Number.isFinite(action.startTime)
    || !Number.isFinite(action.endTime)
  ) return null
  const { start, end, duration } = actionBounds(action)
  const elapsedSeconds = Math.max(0, Math.min(duration, playheadTime - start))
  const phase = duration > 0 ? clamp01(elapsedSeconds / duration) : (playheadTime < start ? 0 : 1)
  const nearest = action.keypoints
    .filter((keypoint) => (
      Number.isFinite(keypoint.time)
      && keypoint.time >= start
      && keypoint.time <= end
    ))
    .map((keypoint): PlayerActionPlaybackKeypoint => ({
      keypoint,
      kind: keypoint.kind,
      time: keypoint.time,
      phase: duration > 0 ? clamp01((keypoint.time - start) / duration) : 1,
      offsetSeconds: playheadTime - keypoint.time,
      distanceSeconds: Math.abs(playheadTime - keypoint.time),
    }))
    .sort((left, right) => (
      left.distanceSeconds - right.distanceSeconds
      || keypointOrder[left.kind] - keypointOrder[right.kind]
      || left.time - right.time
    ))[0] ?? null
  return {
    action,
    phase,
    durationSeconds: duration,
    elapsedSeconds,
    nearestKeypoint: nearest,
  }
}

export function activePlayerActionPlaybackState(
  actions: readonly PlayerAction[],
  playheadTime: number,
  canonicalPersonId: string | null | undefined = null,
): PlayerActionPlaybackState | null {
  const action = selectActivePlayerAction(actions, playheadTime, canonicalPersonId)
  return action ? playerActionPlaybackState(action, playheadTime) : null
}
