import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import PathTrackingLegend from './PathTrackingLegend.vue'

describe('PathTrackingLegend', () => {
  it('renders nothing while the layer is off', async () => {
    const html = await renderToString(createSSRApp(PathTrackingLegend, { enabled: false }))
    expect(html).not.toContain('path-tracking-legend')
  })

  it('asks for a tracked object when the enabled layer has no selection', async () => {
    const html = await renderToString(createSSRApp(PathTrackingLegend, {
      enabled: true,
      surfaceLabel: '3D scene',
    }))

    expect(html).toContain('aria-label="Path tracking on 3D scene"')
    expect(html).toContain('Path tracking')
    expect(html).toContain('Select a tracked player or ball to show its path')
    expect(html).toContain('empty')
  })

  it('identifies a full-highlight player path and its observed/inferred styles', async () => {
    const html = await renderToString(createSSRApp(PathTrackingLegend, {
      enabled: true,
      subjectKind: 'player',
      subjectLabel: 'Home track 02',
      subjectColor: '#ff3344',
      sampleCount: 18,
    }))

    expect(html).toContain('Home track 02')
    expect(html).toContain('Player · full highlight · 18 samples')
    expect(html).toContain('Observed')
    expect(html).toContain('Inferred')
    expect(html).toContain('--path-subject-color:#ff3344')
  })

  it('explains that an identity without a render track has no path', async () => {
    const html = await renderToString(createSSRApp(PathTrackingLegend, {
      enabled: true,
      unavailableLabel: 'Player A',
    }))

    expect(html).toContain('Player A has no reconstructed path')
  })

  it('does not promise a drawable path from a single selected sample', async () => {
    const html = await renderToString(createSSRApp(PathTrackingLegend, {
      enabled: true,
      subjectKind: 'ball',
      subjectLabel: 'Match ball',
      sampleCount: 1,
    }))

    expect(html).toContain('Ball · no path · 1 sample')
    expect(html).not.toContain('Path segment legend')
    expect(html).toContain('empty')
  })

  it('explains when this surface cannot project an otherwise valid path', async () => {
    const html = await renderToString(createSSRApp(PathTrackingLegend, {
      enabled: true,
      subjectKind: 'player',
      subjectLabel: 'Away track 02',
      sampleCount: 18,
      surfaceUnavailableReason: 'No trusted calibration for this frame',
    }))

    expect(html).toContain('Away track 02')
    expect(html).toContain('No trusted calibration for this frame')
    expect(html).toContain('unavailable')
    expect(html).not.toContain('Path segment legend')
  })
})
