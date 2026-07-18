import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import EditorTimelineSurface from './EditorTimelineSurface.vue'
import type { VideoSegment } from '../../types/media'
import type { SceneDocument } from '../../types/scene'

const segment: VideoSegment = {
  id: 'shot-1',
  label: 'Shot 1',
  start: 10,
  end: 15,
  duration: 5,
  score: 0.9,
  layout: {
    group: 1,
    variant: 'A',
    label: '1-A',
    role: 'original',
    confidence: 0.9,
  },
}

function document(
  id: string,
  duration: number,
  sourceStart: number,
  segments: VideoSegment[],
): SceneDocument {
  return {
    id,
    title: id,
    version: 1,
    revision: 1,
    duration,
    payload: {
      pitch: { length: 105, width: 68 },
      videoAsset: {
        id: 'asset-1',
        filename: 'match.mp4',
        mediaUrl: '/match.mp4',
        posterUrl: '/poster.jpg',
        fps: 25,
        frameCount: 1500,
        processingState: 'ready',
        sourceStart,
        segments,
        segmentLayout: segments.length ? {
          status: 'confirmed',
          method: 'shot-order-fallback',
          confidence: 0.9,
          scoreChangeTimes: [],
          groups: [{
            id: 'event-1',
            index: 1,
            label: 'Event 1',
            segmentIds: ['shot-1'],
            replayCount: 0,
          }],
          warnings: [],
        } : undefined,
      },
      teams: [],
      tracks: [],
      ball: { keyframes: [] },
      eventBindings: [],
      cameraCuts: [],
    },
  }
}

describe('EditorTimelineSurface', () => {
  it('renders the root master timeline while the active document is a segment', async () => {
    const timelineScene = document('timeline-scene', 60, 0, [segment])
    const activeScene = document('segment-scene', 5, 10, [])
    const html = await renderToString(createSSRApp(EditorTimelineSurface, {
      scene: activeScene,
      timelineScene,
      sceneVideo: activeScene.payload.videoAsset,
      currentTime: 0,
      playbackRate: 1,
      playing: false,
      timeLabel: '00:00.00',
      reconstructionRunning: false,
      reconstructionMutationLocked: false,
      selectedActionActorId: null,
      selectedActionActorLabel: '',
      showPlayerActionTimeline: false,
      segmentLayout: {
        groupEditing: ref(false),
        selection: ref<string[]>([]),
        canSplitSelection: ref(false),
        segmentGroupColor: () => '#55aaff',
        toggleGroupEditing: vi.fn(),
        splitSelection: vi.fn(),
        saveGroupMap: vi.fn(),
      },
      manualBall: { mode: ref('automatic') },
      playerActions: { visible: ref(false) },
    }))

    expect(html).toContain('Full video timeline')
    expect(html).toContain('click any segment to open')
    expect(html).toContain('1-A')
    expect(html).toContain('10.0–15.0s')
    expect(html).toContain('aria-label="1-A, Original, 10.00 to 15.00 seconds"')
    expect(html).toContain('5.00s')
    expect(html).not.toContain('Edit groups')
  })
})
