import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import type { CalibrationFrameEvidence, Keyframe } from '../types'
import { resolvePathProjectionContext } from '../lib/pathProjection'
import VideoPathTrackingOverlay from './VideoPathTrackingOverlay.vue'

const calibration = (status: CalibrationFrameEvidence['status'] = 'accepted'): CalibrationFrameEvidence => ({
  sourceFrameIndex: 0,
  sampleIndex: 0,
  sceneTime: 0,
  sourceTime: 0,
  status,
  source: 'test',
  projectionSource: 'direct',
  frameWidth: 100,
  frameHeight: 60,
  imageToPitch: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
})

const keyframe = (t: number, evidence: 'observed' | 'inferred'): Keyframe => ({
  t,
  x: 10 + t * 10,
  z: 20 + t * 5,
  confidence: evidence === 'observed' ? 0.9 : 0.3,
  observed: evidence === 'observed',
  presenceState: evidence === 'observed' ? 'observed' : 'inferred-gap',
})

describe('VideoPathTrackingOverlay', () => {
  it('renders observed and inferred world paths through the current camera', async () => {
    const html = await renderToString(createSSRApp(VideoPathTrackingOverlay, {
      enabled: true,
      currentTime: 0,
      color: '#71e2aa',
      subjectLabel: 'Away track 02',
      projectionContext: resolvePathProjectionContext([calibration()], 0),
      keyframes: [
        keyframe(0, 'observed'),
        keyframe(1, 'observed'),
        keyframe(2, 'inferred'),
        keyframe(3, 'inferred'),
      ],
    }))

    expect(html).toContain('video-path-tracking-overlay')
    expect(html).toContain('data-projection-mode="exact"')
    expect(html).toMatch(/class="(?:observed path-edge|path-edge observed)"/)
    expect(html).toMatch(/class="(?:inferred path-edge|path-edge inferred)"/)
    expect(html).toContain('Away track 02 path on the current calibrated video frame')
    expect(html).toContain('path-current-marker')
    expect(html).toContain('cx="10"')
    expect(html).toContain('cy="20"')
  })

  it('renders nothing when disabled, uncalibrated, or missing a real path', async () => {
    const render = (props: Record<string, unknown>) => renderToString(createSSRApp(
      VideoPathTrackingOverlay,
      {
        enabled: true,
        currentTime: 0,
        projectionContext: resolvePathProjectionContext([calibration()], 0),
        keyframes: [keyframe(0, 'observed'), keyframe(1, 'observed')],
        ...props,
      },
    ))

    expect(await render({ enabled: false })).not.toContain('<svg')
    expect(await render({ projectionContext: resolvePathProjectionContext([calibration('rejected')], 0) })).not.toContain('<svg')
    expect(await render({ keyframes: [keyframe(0, 'observed')] })).not.toContain('<svg')
  })

  it('does not invent a current marker outside the reconstructed time range', async () => {
    const html = await renderToString(createSSRApp(VideoPathTrackingOverlay, {
      enabled: true,
      currentTime: 3,
      projectionContext: resolvePathProjectionContext([
        { ...calibration(), sceneTime: 3, sourceTime: 3 },
      ], 3),
      keyframes: [keyframe(0, 'observed'), keyframe(1, 'observed')],
    }))

    expect(html).toContain('video-path-tracking-overlay')
    expect(html).not.toContain('path-current-marker')
  })
})
