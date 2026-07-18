import { describe, expect, it } from 'vitest'
import type { PlayerAction, PlayerActionType } from '../types/playerActions'
import {
  PLAYER_ACTION_CATEGORY_META,
  PLAYER_ACTION_TAXONOMY,
  PLAYER_ACTION_TYPES,
  activePlayerActionPlaybackState,
  defaultPlayerActionDuration,
  defaultPlayerActionKeypointKind,
  filterPlayerActionsForActor,
  playerActionCategory,
  playerActionColor,
  playerActionLabel,
  playerActionPlaybackState,
  selectActivePlayerAction,
} from './playerActions'

function action(id: string, overrides: Partial<PlayerAction> = {}): PlayerAction {
  return {
    id,
    canonicalPersonId: 'person-a',
    type: 'pass',
    startTime: 1,
    endTime: 2,
    keypoints: [],
    confidence: 0.5,
    status: 'suggested',
    source: 'automatic',
    ...overrides,
  }
}

describe('player action domain utilities', () => {
  it('defines complete labelled taxonomy metadata and editing defaults', () => {
    const expected: PlayerActionType[] = [
      'idle', 'walk', 'run', 'sprint', 'turn', 'jump', 'fall', 'get-up',
      'first-touch', 'drive', 'pass', 'cross', 'shot', 'header', 'throw-in',
      'clearance', 'tackle', 'slide-tackle', 'block', 'interception', 'feint',
    ]
    expect(PLAYER_ACTION_TYPES).toEqual(expected)
    expect(Object.keys(PLAYER_ACTION_TAXONOMY)).toHaveLength(expected.length)
    expect(playerActionLabel('first-touch')).toBe('First touch')
    expect(playerActionCategory('slide-tackle')).toBe('defensive')
    expect(playerActionColor('slide-tackle')).toBe(PLAYER_ACTION_CATEGORY_META.defensive.color)
    expect(defaultPlayerActionDuration('shot')).toBeGreaterThan(0)
    expect(defaultPlayerActionKeypointKind('shot')).toBe('contact')
    expect(defaultPlayerActionKeypointKind('jump')).toBe('apex')
    expect(defaultPlayerActionKeypointKind('throw-in')).toBe('release')
    expect(PLAYER_ACTION_TYPES.every((type) => (
      PLAYER_ACTION_TAXONOMY[type].label.length > 0
      && /^#[0-9a-f]{6}$/i.test(PLAYER_ACTION_TAXONOMY[type].color)
      && PLAYER_ACTION_TAXONOMY[type].defaultDurationSeconds > 0
    ))).toBe(true)
  })

  it('filters the dedicated timeline by canonical actor without mutating its input', () => {
    const input = [
      action('late', { canonicalPersonId: 'person-a', startTime: 4, endTime: 5 }),
      action('other', { canonicalPersonId: 'person-b', startTime: 2, endTime: 3 }),
      action('early', { canonicalPersonId: 'person-a', startTime: 1, endTime: 2 }),
    ]
    expect(filterPlayerActionsForActor(input, 'person-a').map((item) => item.id)).toEqual([
      'early',
      'late',
    ])
    expect(filterPlayerActionsForActor(input, null).map((item) => item.id)).toEqual([
      'early',
      'other',
      'late',
    ])
    expect(input.map((item) => item.id)).toEqual(['late', 'other', 'early'])
  })

  it('selects overlapping actions deterministically with manual and confirmed authority', () => {
    const actions = [
      action('automatic-suggested', { confidence: 0.99 }),
      action('automatic-confirmed', { status: 'confirmed', confidence: 0.95 }),
      action('manual-suggested', { source: 'manual', confidence: 0.2 }),
      action('manual-confirmed-low', { source: 'manual', status: 'confirmed', confidence: 0.4 }),
      action('manual-confirmed-high', { source: 'manual', status: 'confirmed', confidence: 0.8 }),
      action('rejected', { source: 'manual', status: 'rejected', confidence: 1 }),
      action('other-actor', { canonicalPersonId: 'person-b', source: 'manual', status: 'confirmed' }),
    ]

    expect(selectActivePlayerAction(actions, 1.5, 'person-a')?.id).toBe('manual-confirmed-high')
    expect(selectActivePlayerAction([...actions].reverse(), 1.5, 'person-a')?.id).toBe('manual-confirmed-high')
    expect(selectActivePlayerAction(actions, 1.5, 'person-b')?.id).toBe('other-actor')
    expect(selectActivePlayerAction(actions, 3, 'person-a')).toBeNull()
    expect(selectActivePlayerAction(actions, Number.NaN, 'person-a')).toBeNull()
  })

  it('uses stable specificity and id tie-breakers for equally ranked intervals', () => {
    const broad = action('broad', {
      source: 'manual',
      status: 'confirmed',
      confidence: 0.8,
      startTime: 0,
      endTime: 4,
    })
    const narrowB = action('narrow-b', {
      source: 'manual',
      status: 'confirmed',
      confidence: 0.8,
      startTime: 1,
      endTime: 2,
    })
    const narrowA = { ...narrowB, id: 'narrow-a' }
    expect(selectActivePlayerAction([broad, narrowB, narrowA], 1.5)?.id).toBe('narrow-a')
  })

  it('normalizes seek-safe phase and reports the nearest valid significant keypoint', () => {
    const contact = { kind: 'contact' as const, time: 11 }
    const release = { kind: 'release' as const, time: 13 }
    const state = playerActionPlaybackState(action('shot', {
      type: 'shot',
      startTime: 10,
      endTime: 14,
      keypoints: [
        release,
        { kind: 'wind-up', time: Number.NaN },
        { kind: 'recovery', time: 20 },
        contact,
      ],
    }), 12)

    expect(state).toMatchObject({
      phase: 0.5,
      durationSeconds: 4,
      elapsedSeconds: 2,
      nearestKeypoint: {
        kind: 'contact',
        time: 11,
        phase: 0.25,
        offsetSeconds: 1,
        distanceSeconds: 1,
      },
    })
    expect(state?.nearestKeypoint?.keypoint).toBe(contact)
  })

  it('clamps phase outside the interval and handles zero/reversed intervals', () => {
    const normal = action('normal', { startTime: 10, endTime: 12 })
    expect(playerActionPlaybackState(normal, 5)?.phase).toBe(0)
    expect(playerActionPlaybackState(normal, 20)?.phase).toBe(1)

    const reversed = action('reversed', { startTime: 12, endTime: 10 })
    expect(playerActionPlaybackState(reversed, 11)?.phase).toBe(0.5)
    expect(selectActivePlayerAction([reversed], 11)?.id).toBe('reversed')

    const instant = action('instant', { startTime: 5, endTime: 5 })
    expect(playerActionPlaybackState(instant, 4)?.phase).toBe(0)
    expect(playerActionPlaybackState(instant, 5)?.phase).toBe(1)
    expect(playerActionPlaybackState(instant, 6)?.phase).toBe(1)
    expect(playerActionPlaybackState({ ...normal, startTime: Number.NaN }, 11)).toBeNull()
  })

  it('combines actor-aware active selection with normalized playback state', () => {
    const result = activePlayerActionPlaybackState([
      action('person-a', { startTime: 0, endTime: 2 }),
      action('person-b', {
        canonicalPersonId: 'person-b',
        type: 'block',
        startTime: 1,
        endTime: 3,
        source: 'manual',
        status: 'confirmed',
      }),
    ], 2, 'person-b')

    expect(result?.action.id).toBe('person-b')
    expect(result?.phase).toBe(0.5)
  })
})
