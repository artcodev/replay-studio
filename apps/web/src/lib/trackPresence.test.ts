import { describe, expect, it } from 'vitest'
import { trackPresenceAtTime, trackPresenceSummary } from './trackPresence'
import type { Track } from '../types'

const track: Track = {
  id: 'player-7',
  label: 'Player 7',
  teamId: 'home',
  color: '#ffffff',
  number: 7,
  externalPlayerId: null,
  presence: {
    policy: 'continuous-latent',
    coverage: 1,
    observationCount: 2,
    inferredKeyframeCount: 3,
    observedStart: 1,
    observedEnd: 2,
    observedSpanRatio: 0.25,
    sampleCadenceSeconds: 0.2,
  },
  keyframes: [
    { t: 0, x: 1, z: 1, confidence: 0.18, observed: false, presenceState: 'inferred-before-first', positionUncertaintyMetres: 5 },
    { t: 1, x: 2, z: 2, confidence: 0.9, observed: true, presenceState: 'observed', positionUncertaintyMetres: 1 },
    { t: 1.5, x: 3, z: 3, confidence: 0.18, observed: false, presenceState: 'inferred-gap', positionUncertaintyMetres: 4 },
    { t: 2, x: 4, z: 4, confidence: 0.85, observed: true, presenceState: 'observed', positionUncertaintyMetres: 1.5 },
    { t: 4, x: 5, z: 5, confidence: 0.18, observed: false, presenceState: 'inferred-after-last', positionUncertaintyMetres: 8 },
  ],
}

describe('track presence inspector data', () => {
  it('distinguishes observed and all inferred phases at the current time', () => {
    expect(trackPresenceAtTime(track, 0).label).toBe('INFERRED · BEFORE')
    expect(trackPresenceAtTime(track, 1).label).toBe('OBSERVED')
    expect(trackPresenceAtTime(track, 1.5).label).toBe('INFERRED · GAP')
    expect(trackPresenceAtTime(track, 4).label).toBe('INFERRED · AFTER')
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
    expect(trackPresenceSummary(track)).toEqual({
      timelineCoverage: 1,
      observedSampleRatio: 0.4,
      inferredSampleRatio: 0.6,
      observedSpanRatio: 0.25,
      observationCount: 2,
      inferredKeyframeCount: 3,
    })
  })
})
