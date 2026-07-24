import { afterEach, describe, expect, it, vi } from 'vitest'
import { sceneClient } from './scenes'
import type { SceneDocument } from '../../types/scene'

function scene(revision: number): SceneDocument {
  return {
    id: 'video-1',
    title: 'Timeline',
    version: 1,
    revision,
    duration: 10,
    payload: {
      pitch: { length: 105, width: 68 },
      teams: [],
      tracks: [],
      eventBindings: [],
      ball: { mode: 'automatic', keyframes: [] },
      videoAsset: {
        id: 'asset-1',
        segments: [
          {
            id: 'shot-01',
            layout: { group: 1, variant: 'A', label: '1-A', role: 'original', confidence: 1 },
          },
        ],
        segmentLayout: { status: 'edited', groups: [] },
      },
    },
  } as unknown as SceneDocument
}

function stubFetch() {
  const calls: Array<{ path: string; method: string; body: Record<string, unknown> }> = []
  vi.stubGlobal('fetch', vi.fn(async (path: string, init?: RequestInit) => {
    calls.push({
      path,
      method: init?.method ?? 'GET',
      body: init?.body ? JSON.parse(String(init.body)) : {},
    })
    return { ok: true, status: 200, json: async () => scene(99) } as unknown as Response
  }))
  return calls
}

afterEach(() => { vi.unstubAllGlobals() })

describe('scene editor commands', () => {
  it('carry only their own domain and never a scene document', async () => {
    const calls = stubFetch()

    await sceneClient.saveTitle('project-1', 'video-1', 'New name')
    await sceneClient.saveEventBindings('project-1', 'video-1', [
      { sceneTime: 1.5, externalEventId: 'event-1', label: 'Goal', type: 'goal' },
    ] as SceneDocument['payload']['eventBindings'])
    await sceneClient.saveTrackMetadata('project-1', 'video-1', 'auto-home-02', { number: 9 })
    await sceneClient.saveSegmentLayout('project-1', scene(16))
    await sceneClient.setFrameExcluded('project-1', 'video-1', 318, true)

    expect(calls.map((call) => call.method)).toEqual(['PUT', 'PUT', 'PUT', 'PUT', 'PUT'])
    expect(calls.map((call) => call.path.split('/scenes/video-1')[1])).toEqual([
      '/title',
      '/event-bindings',
      '/tracks/auto-home-02/metadata',
      '/segment-layout',
      '/frame-exclusions/318',
    ])
    // The architectural invariant: no request ever carries a document
    // revision, so the scene revision fence can never refuse an editor edit.
    for (const call of calls) {
      expect(call.body).not.toHaveProperty('revision')
      expect(call.body).not.toHaveProperty('payload')
    }
    expect(calls[0].body).toEqual({ title: 'New name' })
    expect(calls[2].body).toEqual({ number: 9 })
    expect(calls[4].body).toEqual({ excluded: true })
  })

  it('sends the whole layout so the server can normalize event grouping', async () => {
    const calls = stubFetch()

    await sceneClient.saveSegmentLayout('project-1', scene(16))

    expect(calls[0].body).toEqual({
      status: 'edited',
      segments: [
        {
          id: 'shot-01',
          group: 1,
          variant: 'A',
          label: '1-A',
          role: 'original',
          confidence: 1,
        },
      ],
    })
  })

  it('builds a generation-pinned exact analysis JPEG URL', () => {
    expect(sceneClient.exactAnalysisFrameUrl(
      'project-1',
      'video-1',
      'generation/with spaces',
      318,
    )).toBe(
      '/api/projects/project-1/scenes/video-1/analysis-frame-generations/'
      + 'generation%2Fwith%20spaces/frames/318',
    )
  })
})
