import { afterEach, describe, expect, it, vi } from 'vitest'
import type { PlayerAction } from '../types/playerActions'
import { sceneClient } from './api/scenes'
import { reconstructionClient } from './api/reconstruction'
import { identityClient } from './api/identities'
import { matchClient } from './api/matches'
import { playerActionClient } from './api/playerActions'

const PROJECT_ID = 'project/one'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('canonical roster binding API', () => {
  it('enqueues model comparison instead of waiting for an inference report', async () => {
    const queued = {
      runId: 'model-comparison-run',
      sceneId: 'scene/one',
      kind: 'model-comparison',
      status: 'queued',
    }
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => queued })
    vi.stubGlobal('fetch', fetch)

    expect(await reconstructionClient.compareModels(PROJECT_ID, 'scene/one')).toEqual(queued)

    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/projects/project%2Fone/scenes/scene%2Fone/compare-models')
    expect(init.method).toBe('POST')
  })

  it('never treats embedded dense scene arrays as a playback fallback', async () => {
    const scene = {
      id: 'scene-dense',
      title: 'Scene',
      version: 1,
      duration: 2,
      payload: {
        pitch: { length: 105, width: 68 },
        teams: [],
        tracks: [{
          id: 'track-1', label: 'Player', teamId: 'home', color: '#fff', number: 1,
          externalPlayerId: null,
          keyframes: [{ t: 1, x: 2, z: 3, confidence: 1 }],
          observations: [],
        }],
        canonicalPeople: [],
        ball: { keyframes: [{ t: 1, x: 2, z: 3, confidence: 1 }] },
        videoAsset: { reconstruction: { status: 'ready' } },
      },
    }
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => scene })
    vi.stubGlobal('fetch', fetch)

    const loaded = await sceneClient.get(PROJECT_ID, 'scene-dense')

    expect(loaded.payload.tracks[0].keyframes).toEqual([])
    expect(loaded.payload.ball.keyframes).toEqual([])
    expect(fetch).toHaveBeenCalledOnce()
  })

  it('surfaces an artifact window failure instead of silently using scene arrays', async () => {
    const scene = {
      id: 'scene-artifact',
      title: 'Scene',
      version: 1,
      duration: 2,
      payload: {
        pitch: { length: 105, width: 68 },
        teams: [],
        tracks: [],
        canonicalPeople: [],
        ball: { keyframes: [] },
        videoAsset: {
          reconstruction: {
            status: 'ready',
            artifactManifest: {
              schemaVersion: 1,
              artifacts: { identityTimeline: { id: 'sha256:x' } },
            },
          },
        },
      },
    }
    const fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => scene })
      .mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: async () => ({ detail: 'Reconstruction artifact is unavailable or corrupt' }),
      })
    vi.stubGlobal('fetch', fetch)

    await expect(sceneClient.get(PROJECT_ID, 'scene-artifact')).rejects.toThrow(
      'Reconstruction artifact is unavailable or corrupt',
    )
    expect(fetch).toHaveBeenCalledTimes(2)
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

    const result = await identityClient.updateRosterBinding(
      PROJECT_ID,
      'scene/one',
      'canonical/person-2',
      externalPlayerId,
    )

    expect(result).toEqual(scene)
    expect(fetch).toHaveBeenCalledOnce()
    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe(
      '/api/projects/project%2Fone/scenes/scene%2Fone/canonical-people/canonical%2Fperson-2/roster-binding',
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

    const result = await identityClient.clearRosterBinding(
      PROJECT_ID,
      'scene/one',
      'canonical/person-2',
    )

    expect(result).toEqual(scene)
    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe(
      '/api/projects/project%2Fone/scenes/scene%2Fone/canonical-people/canonical%2Fperson-2/roster-binding',
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

    expect(await identityClient.review(PROJECT_ID, 'scene/one')).toEqual(review)
    expect(await identityClient.rejectRosterCandidate(
      PROJECT_ID,
      'scene/one',
      'canonical/person-2',
      'player/10',
    )).toEqual(scene)

    expect(fetch.mock.calls[0][0]).toBe('/api/projects/project%2Fone/scenes/scene%2Fone/identity-review')
    const [path, init] = fetch.mock.calls[1] as [string, RequestInit]
    expect(path).toBe(
      '/api/projects/project%2Fone/scenes/scene%2Fone/canonical-people/canonical%2Fperson-2/roster-rejections',
    )
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual({ external_player_id: 'player/10' })
  })

  it('refreshes a persisted match snapshot without calling the catalog endpoint', async () => {
    const response = { id: 'match-1', sync: { state: 'synced' } }
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => response })
    vi.stubGlobal('fetch', fetch)

    expect(await matchClient.refresh(PROJECT_ID)).toEqual(response)

    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/projects/project%2Fone/match/refresh')
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
    const response = { id: 'match-1', sync: { state: 'manual' } }
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => response })
    vi.stubGlobal('fetch', fetch)

    expect(await matchClient.import(PROJECT_ID, payload)).toEqual(response)

    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/projects/project%2Fone/match/import')
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

    await expect(matchClient.import(PROJECT_ID, {
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
    await playerActionClient.upsert(PROJECT_ID, 'scene/one', fullAction)

    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/projects/project%2Fone/scenes/scene%2Fone/player-actions')
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

    await playerActionClient.remove(PROJECT_ID, 'scene/one', 'action/one')

    const [path, init] = fetch.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/projects/project%2Fone/scenes/scene%2Fone/player-actions/action%2Fone')
    expect(init.method).toBe('DELETE')
  })
})
