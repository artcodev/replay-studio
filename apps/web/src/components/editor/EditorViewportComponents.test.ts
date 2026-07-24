import { createSSRApp, h } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import type { FrameAnalysis } from '../../types/analysis'
import type { PitchCalibrationDraft } from '../../types/calibration'
import FrameDetectionOverlay from './FrameDetectionOverlay.vue'
import PitchCalibrationOverlay from './PitchCalibrationOverlay.vue'
import PitchCalibrationPanel from './PitchCalibrationPanel.vue'
import SelectedTrackProjectionOverlay from './SelectedTrackProjectionOverlay.vue'
import TrackProjectionDebugCard from './TrackProjectionDebugCard.vue'
import VideoReviewPane from './VideoReviewPane.vue'
import { DEFAULT_VIDEO_OVERLAY_OPTIONS } from '../../lib/videoOverlayOptions'

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
          id: 40,
          image: { x: 400, y: 300 },
          pitch: { x: 32.5, z: -1.71 },
          projected: { x: 402, y: 301 },
          residualPx: 2.2,
          groundResidualMetres: 1.08,
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
        groundResidualP50: 0.32,
        groundResidualP95: 1.08,
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
    expect(html).toContain('KP 40')
    expect(html).toContain('1.08m ground')
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
      overlayOptions: { ...DEFAULT_VIDEO_OVERLAY_OPTIONS },
    }))
    expect(html).toContain('frame-analysis-overlay labeling')
    expect(html).toContain('Away 8')
    expect(html).toContain('EXCLUDED')
    expect(html).toContain('BALL 84%')
    expect(html).toContain('aria-pressed="true"')
  })

  it('renders the selected stored bbox through the exact frame homography', async () => {
    const html = await renderToString(createSSRApp(SelectedTrackProjectionOverlay, {
      enabled: true,
      label: 'Home track 06',
      observations: [{
        frameIndex: 30,
        sourceFrameIndex: 30,
        sceneTime: 1.25,
        bbox: { x: 100, y: 120, width: 50, height: 140 },
        confidence: 0.91,
        metricStatus: 'rejected',
        metricReason: 'trajectory-fragment-rejected',
      }],
      calibrationFrames: [{
        sourceFrameIndex: 30,
        sampleIndex: 30,
        sceneTime: 1.25,
        sourceTime: 2.25,
        status: 'accepted',
        solutionStatus: 'direct-accepted',
        source: 'direct',
        projectionSource: 'direct',
        imageToPitch: [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]],
      }],
      pitch: { length: 105, width: 68 },
      currentTime: 1.25,
      frameSize: { width: 1280, height: 720 },
      contactPointProfile: 'bbox-bottom',
    }))

    expect(html).toContain('selected-projection-debug-overlay')
    expect(html).toContain('Home track 06 · #30 · rejected')
    expect(html).toContain('x 12.50 · z 26.00')
    expect(html).toContain('class="debug-contact"')
  })

  it('renders recognizable pitch markings and high-contrast position markers in the debugger', async () => {
    const html = await renderToString(createSSRApp(TrackProjectionDebugCard, {
      label: 'Home track 06',
      observations: [
        {
          frameIndex: 29,
          sourceFrameIndex: 29,
          sceneTime: 1.21,
          bbox: { x: 95, y: 120, width: 50, height: 140 },
          confidence: 0.9,
          metricStatus: 'accepted',
        },
        {
          frameIndex: 30,
          sourceFrameIndex: 30,
          sceneTime: 1.25,
          bbox: { x: 100, y: 120, width: 50, height: 140 },
          confidence: 0.91,
          metricStatus: 'accepted',
        },
      ],
      tracks: [],
      calibrationFrames: [29, 30].map((frame) => ({
        sourceFrameIndex: frame,
        sampleIndex: frame,
        sceneTime: frame === 29 ? 1.21 : 1.25,
        sourceTime: frame === 29 ? 2.21 : 2.25,
        status: 'accepted' as const,
        solutionStatus: 'direct-accepted' as const,
        source: 'direct',
        projectionSource: 'direct' as const,
        imageToPitch: [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]],
      })),
      pitch: { length: 105, width: 68 },
      currentTime: 1.25,
      contactPointProfile: 'bbox-bottom',
    }))

    expect(html).toContain('Football pitch markings')
    expect(html).toContain('penalty-area left')
    expect(html).toContain('goal-area right')
    expect(html).toContain('penalty-arc left')
    expect(html).toContain('current-halo')
    expect(html).toContain('previous-point')
    expect(html).toContain('Previous frame')
    expect(html).not.toContain('marker-end')
    expect(html).toContain('Zoom in projection minimap')
    expect(html).toContain('Zoom out projection minimap')
    expect(html).toContain('Reset projection minimap')
    expect(html).toContain('Wheel to zoom · drag to pan')
    expect(html).toContain('Recalibrate exact frame #30')
    expect(html).toContain('Motion-comp. matrix mismatch')
  })

  it('hides only the overlay layers the View menu switched off', async () => {
    const html = await renderToString(createSSRApp(FrameDetectionOverlay, {
      analysis: frameAnalysis,
      selectedPersonId: 'person-1',
      labeling: true,
      draft: null,
      canonicalId: () => 'canonical-1',
      personLabel: () => 'Away 8',
      selectionDescription: () => 'linked to 3D',
      overlayOptions: {
        ...DEFAULT_VIDEO_OVERLAY_OPTIONS,
        ballBoxes: false,
        manualMarks: false,
        identityLabels: false,
      },
    }))
    // Switched-off groups disappear...
    expect(html).not.toContain('BALL 84%')
    expect(html).not.toContain('EXCLUDED')
    expect(html).not.toContain('91%')
    // ...while the layers left enabled still render, and the box keeps its
    // accessible name even when its visible label is hidden.
    expect(html).toContain('frame-person-box')
    expect(html).toContain('jersey-swatch')
    expect(html).toContain('aria-label="Away 8, 91 percent, linked to 3D"')
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
    expect(html).toContain('Save frame correction')
    expect(html).toContain('Saving stages this frame')
    expect(html).toContain('finalization is a separate action')
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
