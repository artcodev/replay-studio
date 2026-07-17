import { afterEach, describe, expect, it, vi } from 'vitest'
import type { PlayerAction } from '../types'
import { api } from './api'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('canonical roster binding API', () => {
  it('discovers match-data providers without receiving credentials', async () => {
    const response = {
      defaultProvider: 'api-football',
      providers: [{
        id: 'api-football',
        name: 'API-Football',
        configured: true,
        available: true,
        capabilities: ['fixtures', 'lineups'],
      }],
    }
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => response })
    vi.stubGlobal('fetch', fetch)

    expect(await api.matchDataProviders()).toEqual(response)
    expect(fetch).toHaveBeenCalledWith('/api/catalog/providers', expect.any(Object))
  })

  it('keeps provider selection on catalog and project-binding requests', async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => [] })
    vi.stubGlobal('fetch', fetch)

    await api.eventsByDate('2026-07-17', 'api-football')
    await api.searchEvents('Spain & Belgium', 'api-football')
    await api.bindSceneMatch('project/one', 'fixture/42', 'api-football')

    expect(fetch.mock.calls[0][0]).toBe(
      '/api/catalog/events?date=2026-07-17&provider=api-football',
    )
    expect(fetch.mock.calls[1][0]).toBe(
      '/api/catalog/events/search?q=Spain%20%26%20Belgium&provider=api-football',
    )
    const [path, init] = fetch.mock.calls[2] as [string, RequestInit]
    expect(path).toBe('/api/scenes/project%2Fone/match-binding')
    expect(JSON.parse(String(init.body))).toEqual({
      event_id: 'fixture/42',
      provider: 'api-football',
    })
  })

  it.each([
    ['player-8', 'player-8'],
    [null, null],
  ])('uses the durable canonical endpoint for %s', async (_label, externalPlayerId) => {
    const scene = {
      id: 'scene/one',
      title: 'Scene',
      version: 1,
      duration: 4,
      payload: {},
    }
    const fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => scene,
    })
    vi.stubGlobal('fetch', fetch)

    const result = await api.updateCanonicalRosterBinding(
      'scene/one',
      'canonical/person-2',
      externalPlayerId,
    )

    expect(result).toEqual(scene)
    expect(fetch).toHaveBeenCalledOnce()
    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe(
      '/api/scenes/scene%2Fone/canonical-people/canonical%2Fperson-2/roster-binding',
    )
    expect(init.method).toBe('PUT')
    expect(JSON.parse(String(init.body))).toEqual({ external_player_id: externalPlayerId })
  })

  it('deletes the dedicated binding decision without using the generic annotation API', async () => {
    const scene = {
      id: 'scene/one',
      title: 'Scene',
      version: 2,
      duration: 4,
      payload: {},
    }
    const fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => scene,
    })
    vi.stubGlobal('fetch', fetch)

    const result = await api.clearCanonicalRosterBinding(
      'scene/one',
      'canonical/person-2',
    )

    expect(result).toEqual(scene)
    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe(
      '/api/scenes/scene%2Fone/canonical-people/canonical%2Fperson-2/roster-binding',
    )
    expect(init.method).toBe('DELETE')
    expect(init.body).toBeUndefined()
  })

  it('loads the compact identity review projection and persists roster rejection decisions', async () => {
    const review = { sceneId: 'scene/one', revision: 7, items: [] }
    const scene = { id: 'scene/one', revision: 8, payload: {} }
    const fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => review })
      .mockResolvedValueOnce({ ok: true, json: async () => scene })
    vi.stubGlobal('fetch', fetch)

    expect(await api.identityReview('scene/one')).toEqual(review)
    expect(await api.rejectRosterCandidate(
      'scene/one',
      'canonical/person-2',
      'player/10',
    )).toEqual(scene)

    expect(fetch.mock.calls[0][0]).toBe('/api/scenes/scene%2Fone/identity-review')
    const [path, init] = fetch.mock.calls[1] as [string, RequestInit]
    expect(path).toBe(
      '/api/scenes/scene%2Fone/canonical-people/canonical%2Fperson-2/roster-rejections',
    )
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual({ external_player_id: 'player/10' })
  })

  it('refreshes a persisted match snapshot without calling the catalog endpoint', async () => {
    const response = { scene: { id: 'scene/one' }, bundle: { source: 'thesportsdb' } }
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => response })
    vi.stubGlobal('fetch', fetch)

    expect(await api.refreshSceneMatchBinding('scene/one')).toEqual(response)

    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/scenes/scene%2Fone/match-binding/refresh')
    expect(init.method).toBe('POST')
    expect(init.body).toBeUndefined()
  })

  it('posts a user-supplied roster JSON to the guarded import endpoint unchanged', async () => {
    const payload = {
      event: { id: 'match-1', name: 'Home v Away' },
      teams: {
        home: { id: 'home', name: 'Home' },
        away: { id: 'away', name: 'Away' },
      },
      players: [{ id: 'player-1', name: 'Player One', team_id: 'home' }],
    }
    const response = { scene: { id: 'scene/one' }, bundle: { source: 'manual' } }
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => response })
    vi.stubGlobal('fetch', fetch)

    expect(await api.importSceneMatchBinding('scene/one', payload)).toEqual(response)

    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/scenes/scene%2Fone/match-binding/import')
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual(payload)
  })

  it('turns strict roster validation rows into a readable file error', async () => {
    const fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({
        detail: [{
          loc: ['body', 'players', 3, 'number'],
          msg: 'String should have at most 16 characters',
        }],
      }),
    })
    vi.stubGlobal('fetch', fetch)

    await expect(api.importSceneMatchBinding('scene-1', {
      event: { id: 'match-1', name: 'Home v Away' },
      teams: {
        home: { id: 'home', name: 'Home' },
        away: { id: 'away', name: 'Away' },
      },
      players: [{ id: 'p1', name: 'One' }],
    })).rejects.toThrow('players.3.number: String should have at most 16 characters')
  })
})

describe('player action API', () => {
  it('persists the renderer-neutral manual action contract as camelCase', async () => {
    const scene = { id: 'scene/one', payload: { playerActions: [] } }
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => scene })
    vi.stubGlobal('fetch', fetch)

    const fullAction: PlayerAction = {
      id: 'action:1',
      canonicalPersonId: 'canonical/player-7',
      type: 'shot',
      startTime: 1.2,
      endTime: 2.1,
      keypoints: [
        { kind: 'wind-up', time: 1.3 },
        { kind: 'contact', time: 1.75 },
        { kind: 'recovery', time: 2 },
      ],
      confidence: 0.4,
      status: 'suggested',
      source: 'automatic',
    }
    await api.upsertPlayerAction('scene/one', fullAction)

    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/scenes/scene%2Fone/player-actions')
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual({
      id: 'action:1',
      canonicalPersonId: 'canonical/player-7',
      type: 'shot',
      startTime: 1.2,
      endTime: 2.1,
      keypoints: [
        { kind: 'wind-up', time: 1.3 },
        { kind: 'contact', time: 1.75 },
        { kind: 'recovery', time: 2 },
      ],
    })
  })

  it('deletes exactly one encoded manual action id', async () => {
    const scene = { id: 'scene/one', payload: { playerActions: [] } }
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => scene })
    vi.stubGlobal('fetch', fetch)

    await api.deletePlayerAction('scene/one', 'action/one')

    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/scenes/scene%2Fone/player-actions/action%2Fone')
    expect(init.method).toBe('DELETE')
  })
})
