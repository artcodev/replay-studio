import { describe, expect, it } from 'vitest'
import { trackPresenceAtTime, trackPresenceSummary } from './trackPresence'
import type { Track } from '../types/tracking'

const track: Track = {
  id: 'player-7',
  label: 'Player 7',
  teamId: 'home',
  color: '#ffffff',
  number: 7,
  externalPlayerId: null,
  presence: {
    policy: 'observed-window-with-latent-gaps',
    coverage: 0.25,
    observationCount: 2,
    inferredKeyframeCount: 1,
    observedStart: 1,
    observedEnd: 2,
    observedSpanRatio: 0.25,
    sampleCadenceSeconds: 0.2,
  },
  keyframes: [
    { t: 1, x: 2, z: 2, confidence: 0.9, observed: true, presenceState: 'observed', positionUncertaintyMetres: 1 },
    { t: 1.5, x: 3, z: 3, confidence: 0.18, observed: false, presenceState: 'inferred-gap', positionUncertaintyMetres: 4 },
    { t: 2, x: 4, z: 4, confidence: 0.85, observed: true, presenceState: 'observed', positionUncertaintyMetres: 1.5 },
  ],
}

describe('track presence inspector data', () => {
  it('distinguishes the observed window from internal inference', () => {
    expect(trackPresenceAtTime(track, 0).label).toBe('NOT OBSERVED')
    expect(trackPresenceAtTime(track, 1).label).toBe('OBSERVED')
    expect(trackPresenceAtTime(track, 1.5).label).toBe('INFERRED · GAP')
    expect(trackPresenceAtTime(track, 4).label).toBe('NOT OBSERVED')
    expect(trackPresenceAtTime(track, 4).uncertaintyMetres).toBeNull()
  })

  it('interpolates uncertainty without changing presence semantics', () => {
    const snapshot = trackPresenceAtTime(track, 1.25)
    expect(snapshot.state).toBe('inferred-gap')
    expect(snapshot.uncertaintyMetres).toBe(2.5)
  })

  it('only reports detector-backed evidence at an observed timestamp', () => {
    expect(trackPresenceAtTime(track, 1.0005).state).toBe('observed')
    expect(trackPresenceAtTime(track, 1.002).state).toBe('inferred-gap')
    expect(trackPresenceAtTime(track, 1.75).state).toBe('inferred-gap')
    expect(trackPresenceAtTime(track, 1.9995).state).toBe('observed')
  })

  it('reports the observed and inferred sample mix from track presence metadata', () => {
    const summary = trackPresenceSummary(track)
    expect(summary).toMatchObject({
      timelineCoverage: 0.25,
      observedSpanRatio: 0.25,
      observationCount: 2,
      inferredKeyframeCount: 1,
    })
    expect(summary.observedSampleRatio).toBeCloseTo(2 / 3)
    expect(summary.inferredSampleRatio).toBeCloseTo(1 / 3)
  })
})
