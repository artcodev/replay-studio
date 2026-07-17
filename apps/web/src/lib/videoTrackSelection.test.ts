import { describe, expect, it } from 'vitest'
import {
  canonicalSelectionAfterFrameAnalysis,
  frameMetricBadge,
  linkedFrameMetricSelectionStatus,
  renderTrackForFramePerson,
  selectedFramePeople,
  selectionAfterFrameAnalysis,
  videoTrackSelectionStatus,
} from './videoTrackSelection'

const analysis = {
  people: [
    { id: 'person-1', matchedTrackId: 'track-wrong', canonicalPersonId: 'canonical-7' },
    { id: 'person-2', matchedTrackId: null },
    { id: 'person-3', matchedTrackId: 'track-7', canonicalPersonId: 'canonical-7' },
  ],
}

describe('video and 3D track selection', () => {
  it('uses only detector-backed frame matches for the video highlight', () => {
    expect(selectedFramePeople(analysis, 'track-7').map((person) => person.id)).toEqual(['person-3'])
    expect(selectedFramePeople(analysis, 'track-7', 'canonical-7').map((person) => person.id)).toEqual([
      'person-1', 'person-3',
    ])
    expect(selectedFramePeople(analysis, 'track-8')).toEqual([])
    expect(selectedFramePeople(null, 'track-7')).toEqual([])
  })

  it('uses canonical identity before a stale render-track match', () => {
    expect(selectedFramePeople(analysis, 'track-wrong', 'canonical-7').map((person) => person.id)).toEqual([
      'person-1', 'person-3',
    ])
  })

  it('maps a video person to the canonical 3D actor before the stale matchedTrackId', () => {
    const tracks = [
      { id: 'track-wrong', canonicalPersonId: 'canonical-other' },
      { id: 'track-right', canonicalPersonId: 'canonical-7' },
    ]
    expect(renderTrackForFramePerson(analysis.people[0], tracks)?.id).toBe('track-right')
  })

  it('returns no render actor for a valid canonical identity that is not projected', () => {
    expect(renderTrackForFramePerson({
      matchedTrackId: null,
      canonicalPersonId: 'canonical-video-only',
    }, [{ id: 'track-7', canonicalPersonId: 'canonical-7' }])).toBeNull()
  })

  it('falls back to matchedTrackId for legacy frame analyses without canonical ids', () => {
    const legacy = { people: [{ id: 'legacy', matchedTrackId: 'track-7' }] }
    expect(selectedFramePeople(legacy, 'track-7', 'canonical-7')).toEqual(legacy.people)
  })

  it('reports a visible selected track and preserves duplicate match evidence', () => {
    expect(videoTrackSelectionStatus(analysis, 'track-7', 'Player 7', {
      analyzing: false,
      observedAtCurrentTime: true,
      canonicalPersonId: 'canonical-7',
    })).toEqual({
      state: 'visible',
      label: 'Visible in video',
      detail: 'Player 7 · 2 detections',
      matchCount: 2,
    })
  })

  it('reports a visible canonical person even when it has no render actor', () => {
    expect(videoTrackSelectionStatus(analysis, null, 'Canonical player 7', {
      analyzing: false,
      observedAtCurrentTime: null,
      canonicalPersonId: 'canonical-7',
    })).toEqual({
      state: 'visible',
      label: 'Visible in video',
      detail: 'Canonical player 7 · 2 detections',
      matchCount: 2,
    })
  })

  it('does not invent a bbox for an interpolated track', () => {
    expect(videoTrackSelectionStatus(analysis, 'track-8', 'Player 8', {
      analyzing: false,
      observedAtCurrentTime: false,
    })).toMatchObject({
      state: 'inferred',
      matchCount: 0,
    })
  })

  it('distinguishes a detector miss from an unchecked or pending frame', () => {
    expect(videoTrackSelectionStatus(analysis, 'track-8', 'Player 8', {
      analyzing: false,
      observedAtCurrentTime: true,
    })?.state).toBe('missing')
    expect(videoTrackSelectionStatus(null, 'track-8', 'Player 8', {
      analyzing: false,
      observedAtCurrentTime: null,
    })?.state).toBe('unchecked')
    expect(videoTrackSelectionStatus(null, 'track-8', 'Player 8', {
      analyzing: true,
      observedAtCurrentTime: null,
    })?.state).toBe('checking')
  })

  it('preserves the track explicitly selected in 3D after frame analysis', () => {
    expect(selectionAfterFrameAnalysis('track-8', 'track-8', 'track-7', 'track-8')).toBe('track-8')
  })

  it('preserves the current track when Analyze Frame finds another person first', () => {
    expect(selectionAfterFrameAnalysis('track-8', 'track-8', 'track-7')).toBe('track-8')
  })

  it('selects the first match only when no track was selected before analysis', () => {
    expect(selectionAfterFrameAnalysis(null, null, 'track-7')).toBe('track-7')
  })

  it('does not attach the first canonical identity to a selected legacy track', () => {
    expect(canonicalSelectionAfterFrameAnalysis(
      null,
      null,
      'legacy-track-8',
      null,
      'canonical-first-result',
    )).toBeNull()
  })

  it('preserves an offscreen canonical identity and a newer canonical click', () => {
    expect(canonicalSelectionAfterFrameAnalysis(
      'canonical-offscreen',
      'canonical-offscreen',
      null,
      null,
      'canonical-first-result',
    )).toBe('canonical-offscreen')
    expect(canonicalSelectionAfterFrameAnalysis(
      null,
      'canonical-newer-click',
      null,
      null,
      'canonical-first-result',
    )).toBe('canonical-newer-click')
  })

  it('lets a newer user click win over a late analysis response', () => {
    expect(selectionAfterFrameAnalysis('track-7', 'track-9', 'track-8', 'track-7')).toBe('track-9')
  })

  it('keeps visible identity evidence distinct from an uncertain 3D position', () => {
    const person = {
      matchedTrackId: 'track-7',
      metricStatus: 'rejected' as const,
      metricReason: 'outside calibrated pitch',
      positionSource: 'track-inferred' as const,
    }

    expect(frameMetricBadge(person)).toBe('UNCERTAIN')
    expect(linkedFrameMetricSelectionStatus(person, 'Player 7')).toEqual({
      state: 'uncertain',
      label: 'Visible · 3D position uncertain',
      detail: 'Player 7 · outside calibrated pitch',
      matchCount: 1,
    })
  })

  it('marks accepted observation projections as metric without overriding visible status', () => {
    const person = {
      matchedTrackId: 'track-7',
      metricStatus: 'accepted' as const,
      metricReason: null,
      positionSource: 'observation' as const,
    }

    expect(frameMetricBadge(person)).toBe('METRIC')
    expect(linkedFrameMetricSelectionStatus(person, 'Player 7')).toBeNull()
  })

  it('treats a track-inferred position as uncertain even for a linked bbox', () => {
    const person = {
      matchedTrackId: 'track-7',
      metricStatus: null,
      metricReason: null,
      positionSource: 'track-inferred' as const,
    }

    expect(frameMetricBadge(person)).toBe('UNCERTAIN')
    expect(linkedFrameMetricSelectionStatus(person, null)?.detail).toBe(
      'track-7 · using a track-inferred 3D position',
    )
  })

  it('does not turn an unmatched video detection into a metric selection state', () => {
    const person = {
      matchedTrackId: null,
      metricStatus: 'unprojected' as const,
      metricReason: 'calibration unavailable',
      positionSource: null,
    }

    expect(frameMetricBadge(person)).toBe('UNCERTAIN')
    expect(linkedFrameMetricSelectionStatus(person, null)).toBeNull()
  })

  it('keeps uncertain metric QA attached to a canonical identity without a render-track id', () => {
    const person = {
      matchedTrackId: null,
      canonicalPersonId: 'canonical-7',
      metricStatus: 'unprojected' as const,
      metricReason: 'calibration unavailable',
      positionSource: 'track-inferred' as const,
    }

    expect(linkedFrameMetricSelectionStatus(person, 'Canonical player 7')).toEqual({
      state: 'uncertain',
      label: 'Visible · 3D position uncertain',
      detail: 'Canonical player 7 · calibration unavailable',
      matchCount: 1,
    })
  })
})
