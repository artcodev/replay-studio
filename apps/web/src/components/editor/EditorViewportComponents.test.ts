import { createSSRApp, h } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import type { FrameAnalysis } from '../../types/analysis'
import type { PitchCalibrationDraft } from '../../types/calibration'
import FrameDetectionOverlay from './FrameDetectionOverlay.vue'
import PitchCalibrationOverlay from './PitchCalibrationOverlay.vue'
import PitchCalibrationPanel from './PitchCalibrationPanel.vue'
import VideoReviewPane from './VideoReviewPane.vue'

const calibrationDraft = {
  sceneId: 'scene-1',
  sceneTime: 1.25,
  frameIndex: 30,
  frameWidth: 1280,
  frameHeight: 720,
  source: 'frame-evidence',
  preset: 'penalty-area-right',
  confidence: 0.88,
  alignmentError: 3.8,
  quality: 'review',
  anchors: [{
    id: 'anchor-1',
    label: 'Penalty corner',
    image: { x: 400, y: 300 },
    pitch: { x: 35, z: 20 },
  }],
  markings: [{
    id: 'marking-1',
    kind: 'line',
    points: [{ x: 100, y: 200 }, { x: 800, y: 210 }],
  }],
  imageToPitch: [],
  warnings: [],
} satisfies PitchCalibrationDraft

const frameAnalysis = {
  sceneId: 'scene-1',
  requestedTime: 1.25,
  sceneTime: 1.25,
  sourceTime: 2.25,
  frameIndex: 30,
  frameWidth: 1280,
  frameHeight: 720,
  model: 'detector',
  projectionMode: 'metric',
  calibrationStatus: 'ready',
  matchedTracks: 1,
  people: [{
    id: 'person-1',
    confidence: 0.91,
    bbox: { x: 100, y: 120, width: 50, height: 140 },
    pitch: { x: 12, z: 4 },
    jerseyColor: '#2f5cff',
    annotationId: null,
    annotationLabel: null,
    kind: 'away-player',
    source: 'automatic',
    matchedTrackId: 'track-1',
    matchedTrackLabel: 'Away 8',
    teamId: 'away',
    matchDistance: 0.1,
    metricStatus: 'accepted',
    metricReason: null,
    positionSource: 'observation',
    correctionAction: null,
    correctionScope: null,
    mergeTargetId: null,
    sourceTrackId: 'track-1',
    previewState: 'uncorrected',
  }],
  annotations: [{
    id: 'annotation-1',
    sceneTime: 1.25,
    sourceTime: 2.25,
    frameIndex: 30,
    bbox: { x: 600, y: 150, width: 40, height: 120 },
    kind: 'ignore',
    label: null,
    externalPlayerId: null,
    action: 'exclude',
    updatedAt: '2026-07-18T00:00:00Z',
  }],
  correctionSummary: { confirmed: 0, excluded: 1, merged: 0 },
  ballCandidates: [{
    id: 'ball-1',
    confidence: 0.84,
    image: { x: 500, y: 400 },
    pitch: { x: 1, z: 2 },
    primary: true,
  }],
  warnings: [],
} as FrameAnalysis

describe('editor viewport presentation boundaries', () => {
  it('renders calibration evidence and editable anchors in its dedicated overlay', async () => {
    const html = await renderToString(createSSRApp(PitchCalibrationOverlay, {
      draft: calibrationDraft,
      diagnostics: {
        evidence: null,
        points: [{
          id: 'kp-1',
          label: 'Corner',
          image: { x: 400, y: 300 },
          projected: { x: 402, y: 301 },
          residualPx: 2.2,
          inlier: true,
        }],
        lines: [],
        status: 'review',
        sourceStatus: null,
        method: 'keypoints',
        keypointCount: 1,
        inlierCount: 1,
        inlierRatio: 1,
        residualP50: 2.2,
        residualP95: 2.2,
        precision: 1,
        recall: 1,
        f1: 1,
        visibleSide: 'right',
        visibleSideTrusted: false,
        rejectionReasons: [],
      },
      qaFrame: null,
      qaFrameSize: { width: 1280, height: 720 },
      qaMarkings: [],
    }))
    expect(html).toContain('pitch-calibration-overlay')
    expect(html).toContain('calibration-marking')
    expect(html).toContain('calibration-detected-keypoint')
    expect(html).toContain('Calibration anchor 1: Penalty corner')
  })

  it('renders frame people, exclusions, and ball candidates in the detection overlay', async () => {
    const html = await renderToString(createSSRApp(FrameDetectionOverlay, {
      analysis: frameAnalysis,
      selectedPersonId: 'person-1',
      labeling: true,
      draft: null,
      canonicalId: () => 'canonical-1',
      personLabel: () => 'Away 8',
      selectionDescription: () => 'linked to 3D',
    }))
    expect(html).toContain('frame-analysis-overlay labeling')
    expect(html).toContain('Away 8')
    expect(html).toContain('EXCLUDED')
    expect(html).toContain('BALL 84%')
    expect(html).toContain('aria-pressed="true"')
  })

  it('keeps calibration commands and diagnostics in the calibration panel', async () => {
    const html = await renderToString(createSSRApp(PitchCalibrationPanel, {
      draft: calibrationDraft,
      activeAtCurrentTime: false,
      diagnostics: null,
      warnings: ['Review the edge'],
      preset: 'penalty-area-right',
      presets: [{ value: 'penalty-area-right', label: 'Right penalty area' }],
      loading: false,
      applying: false,
    }))
    expect(html).toContain('pitch-calibration-panel left')
    expect(html).toContain('Move the playhead back to 1.25s')
    expect(html).toContain('Review the edge')
    expect(html).toContain('Calibrate again')
    expect(html).toContain('Apply &amp; rebuild')
  })

  it('keeps media, transform controls, and overlay slots in the video pane', async () => {
    const app = createSSRApp({
      render: () => h(VideoReviewPane, {
        asset: {
          id: 'video-1',
          filename: 'clip.mp4',
          mediaUrl: '/clip.mp4',
          posterUrl: '/poster.jpg',
          fps: 25,
          frameCount: 100,
          processingState: 'ready',
        },
        transform: { scale: 1, x: 0, y: 0 },
        transformStyle: { transform: 'translate3d(0, 0, 0) scale(1)' },
        zoomPercent: 100,
        panning: false,
        minScale: 1,
        maxScale: 4,
        caption: 'Original clip',
      }, {
        default: () => h('svg', { class: 'test-overlay' }),
        floating: () => h('aside', { class: 'test-floating' }),
      }),
    })
    const html = await renderToString(app)
    expect(html).toContain('video-review-viewport')
    expect(html).toContain('src="/clip.mp4"')
    expect(html).toContain('test-overlay')
    expect(html).toContain('test-floating')
    expect(html).toContain('clip.mp4 · 25.00 FPS')
  })
})
