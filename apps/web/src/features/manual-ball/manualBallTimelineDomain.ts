export const MANUAL_BALL_TIME_STEP = 0.001

export type ManualBallTimelineState = {
  duration: number
  currentTime: number
  keyframeTimes: number[]
  selectedTime: number | null
}

export type ManualBallTimelineAction =
  | { type: 'select'; time: number }
  | { type: 'add' }
  | { type: 'remove' }
  | { type: 'update-time'; requestedTime: number }

export type ManualBallTimelineEvent =
  | { type: 'seek'; time: number }
  | { type: 'add'; time: number }
  | { type: 'select'; time: number }
  | { type: 'remove'; time: number }
  | { type: 'updateTime'; value: { from: number; to: number } }

function finiteTime(value: number, fallback = 0) {
  return Number.isFinite(value) ? value : fallback
}

export function clampManualBallTime(value: number, duration: number) {
  const safeDuration = Math.max(0, finiteTime(duration))
  return Math.round(Math.min(safeDuration, Math.max(0, finiteTime(value))) * 1000) / 1000
}

export function normalizeManualBallTimes(times: number[], duration: number) {
  return [...new Set(times.map((time) => clampManualBallTime(time, duration)))].sort((first, second) => first - second)
}

function sameTime(first: number, second: number) {
  return Math.abs(first - second) < MANUAL_BALL_TIME_STEP / 2
}

/** Resolve timeline intent without depending on Vue or browser events. */
export function manualBallTimelineEvents(
  state: ManualBallTimelineState,
  action: ManualBallTimelineAction,
): ManualBallTimelineEvent[] {
  const duration = Math.max(0, finiteTime(state.duration))
  const times = normalizeManualBallTimes(state.keyframeTimes, duration)

  if (action.type === 'select') {
    const time = clampManualBallTime(action.time, duration)
    return [{ type: 'select', time }, { type: 'seek', time }]
  }

  if (action.type === 'add') {
    const time = clampManualBallTime(state.currentTime, duration)
    const duplicate = times.find((candidate) => sameTime(candidate, time))
    return duplicate === undefined
      ? [{ type: 'add', time }]
      : [{ type: 'select', time: duplicate }, { type: 'seek', time: duplicate }]
  }

  if (state.selectedTime === null || !Number.isFinite(state.selectedTime)) return []
  const selected = clampManualBallTime(state.selectedTime, duration)
  const selectedIndex = times.findIndex((time) => sameTime(time, selected))
  if (selectedIndex < 0) return []

  if (action.type === 'remove') return [{ type: 'remove', time: times[selectedIndex] }]
  if (!Number.isFinite(action.requestedTime)) return []

  const previous = times[selectedIndex - 1]
  const next = times[selectedIndex + 1]
  const lowerBound = previous === undefined ? 0 : Math.min(duration, previous + MANUAL_BALL_TIME_STEP)
  const upperBound = next === undefined ? duration : Math.max(0, next - MANUAL_BALL_TIME_STEP)
  const to = clampManualBallTime(Math.min(upperBound, Math.max(lowerBound, action.requestedTime)), duration)
  const from = times[selectedIndex]
  return sameTime(from, to) ? [] : [{ type: 'updateTime', value: { from, to } }]
}
