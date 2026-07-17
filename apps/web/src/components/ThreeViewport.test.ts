import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import type { PlayerActionPlaybackState } from '../lib/playerActions'
import type { SceneDocument } from '../types'
import ThreeViewport from './ThreeViewport.vue'

const scene = {
  id: 'scene-1',
  title: 'Test scene',
  version: 1,
  duration: 10,
  payload: {
    pitch: { length: 105, width: 68 },
    matchBinding: null,
    teams: [],
    tracks: [],
    ball: { keyframes: [] },
    eventBindings: [],
    cameraCuts: [],
  },
} as SceneDocument

const activeAction: PlayerActionPlaybackState = {
  action: {
    id: 'action-shot-1',
    canonicalPersonId: 'person-1',
    type: 'shot',
    startTime: 1,
    endTime: 2,
    keypoints: [{ kind: 'contact', time: 1.5 }],
    confidence: 0.9,
    status: 'confirmed',
    source: 'manual',
  },
  phase: 0.5,
  durationSeconds: 1,
  elapsedSeconds: 0.5,
  nearestKeypoint: {
    keypoint: { kind: 'contact', time: 1.5 },
    kind: 'contact',
    time: 1.5,
    phase: 0.5,
    offsetSeconds: 0.1,
    distanceSeconds: 0.1,
  },
}

function render(activePlayerAction: PlayerActionPlaybackState | null) {
  return renderToString(createSSRApp(ThreeViewport, {
    scene,
    currentTime: 1.6,
    selectedTrackId: null,
    editMode: false,
    showTrails: true,
    showLabels: true,
    frameAnalysis: null,
    activePlayerAction,
  }))
}

describe('ThreeViewport action preview', () => {
  it('shows renderer-neutral action timing and semantic keypoint state', async () => {
    const html = await render(activeAction)
    expect(html).toContain('action-preview-hud')
    expect(html).toMatch(/<code[^>]*>shot<\/code>/)
    expect(html).toMatch(/<strong[^>]*>Shot<\/strong>/)
    expect(html).toContain('50%')
    expect(html).toContain('Contact')
    expect(html).toContain('0.10s ago')
    expect(html).toContain('role="progressbar"')
  })

  it('does not render an action HUD without an active interval', async () => {
    expect(await render(null)).not.toContain('action-preview-hud')
  })
})
