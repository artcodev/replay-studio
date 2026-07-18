import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import type { PlayerAction } from '../types/playerActions'
import PlayerActionTimeline from './PlayerActionTimeline.vue'
import {
  clampPlayerActionTime,
  layoutPlayerActions,
  normalizePlayerAction,
  reducePlayerAction,
} from '../features/player-actions/playerActionTimelineDomain'

function playerAction(overrides: Partial<PlayerAction> = {}): PlayerAction {
  return {
    id: 'action-1',
    canonicalPersonId: 'person-1',
    type: 'pass',
    startTime: 1,
    endTime: 2,
    keypoints: [{ kind: 'contact', time: 1.4 }],
    confidence: 1,
    status: 'confirmed',
    source: 'manual',
    ...overrides,
  }
}

describe('PlayerActionTimeline', () => {
  it('publishes the integration event contract', () => {
    const emits = (PlayerActionTimeline as unknown as { emits: string[] }).emits
    expect(emits).toEqual(expect.arrayContaining(['seek', 'add', 'select', 'update', 'remove']))
    expect(emits).toHaveLength(5)
  })

  it('renders an accessible empty state for an unselected person', async () => {
    const html = await renderToString(createSSRApp(PlayerActionTimeline, {
      canonicalPersonId: '',
      duration: 5,
      currentTime: 1.25,
      actions: [],
      selectedActionId: null,
    }))

    expect(html).toContain('Player action timeline')
    expect(html).toContain('Select a player')
    expect(html).toContain('Select an identified player to edit actions.')
    expect(html).toContain('Add action at 00:01.250')
    expect(html).toContain('disabled')
  })

  it('renders only the selected actor actions, interval boundaries, phases and editor', async () => {
    const action = playerAction({
      keypoints: [
        { kind: 'wind-up', time: 1.1 },
        { kind: 'contact', time: 1.4 },
        { kind: 'recovery', time: 1.9 },
      ],
    })
    const html = await renderToString(createSSRApp(PlayerActionTimeline, {
      canonicalPersonId: 'person-1',
      personLabel: 'Away · #8',
      duration: 5,
      currentTime: 1.4,
      actions: [action, playerAction({ id: 'other-action', canonicalPersonId: 'person-2', type: 'shot' })],
      selectedActionId: action.id,
    }))

    expect(html).toContain('Away · #8')
    expect(html).toContain('1 action')
    expect(html).toContain('Pass from 00:01.000 to 00:02.000')
    expect(html).not.toContain('Shot from')
    expect(html.match(/class="[^"]*\baction-keypoint-marker\b[^"]*"/g)).toHaveLength(3)
    expect(html).toContain('Wind Up at 00:01.100 for Pass')
    expect(html).toContain('Contact at 00:01.400 for Pass')
    expect(html).toContain('Recovery at 00:01.900 for Pass')
    expect(html).toContain('data-testid="selected-action-editor"')
    expect(html).toContain('Action start time in seconds')
    expect(html).toContain('Action end time in seconds')
    expect(html).toContain('＋ Add phase')
    expect(html.match(/class="phase-row"/g)).toHaveLength(3)
  })

  it('shows saving state and disables edits while persistence is pending', async () => {
    const html = await renderToString(createSSRApp(PlayerActionTimeline, {
      canonicalPersonId: 'person-1',
      duration: 5,
      currentTime: 1.4,
      actions: [playerAction()],
      selectedActionId: 'action-1',
      saving: true,
    }))

    expect(html).toContain('Saving action…')
    expect(html.match(/disabled/g)?.length).toBeGreaterThan(4)
  })

  it('keeps automatic suggestions selectable but makes their editor review-only', async () => {
    const automatic = playerAction({
      source: 'automatic',
      status: 'suggested',
      confidence: 0.72,
    })
    const html = await renderToString(createSSRApp(PlayerActionTimeline, {
      canonicalPersonId: 'person-1',
      duration: 5,
      currentTime: 1.4,
      actions: [automatic],
      selectedActionId: automatic.id,
    }))

    expect(html).toContain('Automatic suggestion · review-only')
    expect(html).toContain('aria-label="Pass from 00:01.000 to 00:02.000"')
    expect(html).toMatch(/<button[^>]*class="delete-action"[^>]*disabled/)

    const unchanged = reducePlayerAction(automatic, { type: 'set-type', value: 'shot' }, 5)
    expect(unchanged.type).toBe('pass')
    expect(unchanged).not.toBe(automatic)
  })

  it('clamps scene boundaries, orders inverted intervals and keeps phases inside them', () => {
    expect(clampPlayerActionTime(-2, 5)).toBe(0)
    expect(clampPlayerActionTime(8, 5)).toBe(5)
    expect(clampPlayerActionTime(1.23456, 5)).toBe(1.235)

    const normalized = normalizePlayerAction(playerAction({
      startTime: 8,
      endTime: -2,
      confidence: 4,
      keypoints: [
        { kind: 'recovery', time: 9 },
        { kind: 'contact', time: 2 },
        { kind: 'wind-up', time: -1 },
      ],
    }), 5)

    expect(normalized.startTime).toBe(0)
    expect(normalized.endTime).toBe(5)
    expect(normalized.confidence).toBe(1)
    expect(normalized.keypoints).toEqual([
      { kind: 'wind-up', time: 0 },
      { kind: 'contact', time: 2 },
      { kind: 'recovery', time: 5 },
    ])
  })

  it('keeps boundary edits ordered and reclamps phases with the interval', () => {
    const action = playerAction({
      startTime: 1,
      endTime: 4,
      keypoints: [{ kind: 'contact', time: 3 }],
    })

    const crossedStart = reducePlayerAction(action, { type: 'set-start', time: 8 }, 10)
    expect(crossedStart.startTime).toBe(3.999)
    expect(crossedStart.endTime).toBe(4)
    expect(crossedStart.keypoints[0].time).toBe(3.999)

    const crossedEnd = reducePlayerAction(action, { type: 'set-end', time: -3 }, 10)
    expect(crossedEnd.startTime).toBe(1)
    expect(crossedEnd.endTime).toBe(1.001)
    expect(crossedEnd.keypoints[0].time).toBe(1.001)
  })

  it('expands zero-length imported intervals to the minimum supported duration', () => {
    const normalized = normalizePlayerAction(playerAction({ startTime: 5, endTime: 5 }), 10)
    expect(normalized.startTime).toBe(5)
    expect(normalized.endTime).toBe(5.001)
  })

  it('adds, edits, orders and removes multiple significant phases without mutation', () => {
    const original = playerAction({ keypoints: [] })
    const withContact = reducePlayerAction(original, { type: 'add-keypoint', kind: 'contact', time: 1.8 }, 5)
    const withWindUp = reducePlayerAction(withContact, { type: 'add-keypoint', kind: 'wind-up', time: 1.2 }, 5)

    expect(original.keypoints).toEqual([])
    expect(withWindUp.keypoints).toEqual([
      { kind: 'wind-up', time: 1.2 },
      { kind: 'contact', time: 1.8 },
    ])

    const edited = reducePlayerAction(withWindUp, {
      type: 'update-keypoint',
      index: 0,
      kind: 'release',
      time: 9,
    }, 5)
    expect(edited.keypoints).toEqual([
      { kind: 'contact', time: 1.8 },
      { kind: 'release', time: 2 },
    ])

    const removed = reducePlayerAction(edited, { type: 'remove-keypoint', index: 0 }, 5)
    expect(removed.keypoints).toEqual([{ kind: 'release', time: 2 }])
  })

  it('assigns overlapping intervals to separate visual lanes', () => {
    const layout = layoutPlayerActions([
      playerAction({ id: 'a', startTime: 0, endTime: 2 }),
      playerAction({ id: 'b', startTime: 1, endTime: 3 }),
      playerAction({ id: 'c', startTime: 2, endTime: 4 }),
    ], 5)

    expect(layout.map(({ action, lane }) => [action.id, lane])).toEqual([
      ['a', 0],
      ['b', 1],
      ['c', 0],
    ])
  })
})
