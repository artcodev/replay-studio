import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import TrackPresenceCard from './TrackPresenceCard.vue'
import type { Track } from '../types'

const track: Track = {
  id: 'player-10',
  label: 'Player 10',
  teamId: 'away',
  color: '#327bff',
  number: 10,
  externalPlayerId: null,
  presence: {
    policy: 'continuous-latent',
    coverage: 1,
    observationCount: 1,
    inferredKeyframeCount: 3,
    observedStart: 1,
    observedEnd: 1,
    observedSpanRatio: 0,
  },
  keyframes: [
    { t: 0, x: 1, z: 1, confidence: 0.18, observed: false, presenceState: 'inferred-before-first', positionUncertaintyMetres: 4.7 },
    { t: 1, x: 2, z: 2, confidence: 0.9, observed: true, presenceState: 'observed', positionUncertaintyMetres: 0.8 },
    { t: 2, x: 2, z: 2, confidence: 0.18, observed: false, presenceState: 'inferred-after-last', positionUncertaintyMetres: 3.2 },
  ],
}

describe('TrackPresenceCard', () => {
  it('renders current inference, uncertainty, and the observed/inferred sample mix', async () => {
    const html = await renderToString(createSSRApp(TrackPresenceCard, { track, currentTime: 0 }))

    expect(html).toContain('INFERRED · BEFORE')
    expect(html).toContain('± 4.70 m')
    expect(html).toContain('Timeline presence')
    expect(html).toContain('100%')
    expect(html).toContain('Observed 25%')
    expect(html).toContain('Inferred 75%')
    expect(html).toContain('Sample evidence mix')
    expect(html).toContain('1 observed · 3 inferred keyframes')
    expect(html).toContain('aria-label="Sample evidence mix: observed 25%, inferred 75%"')
  })

  it('does not present time between detector samples as observed', async () => {
    const betweenSamplesTrack: Track = {
      ...track,
      presence: {
        ...track.presence!,
        observationCount: 2,
        inferredKeyframeCount: 0,
        observedEnd: 2,
      },
      keyframes: [
        { t: 1, x: 2, z: 2, confidence: 0.9, observed: true, presenceState: 'observed' },
        { t: 2, x: 3, z: 3, confidence: 0.9, observed: true, presenceState: 'observed' },
      ],
    }
    const html = await renderToString(createSSRApp(TrackPresenceCard, {
      track: betweenSamplesTrack,
      currentTime: 1.5,
    }))

    expect(html).toContain('INFERRED · GAP')
    expect(html).not.toContain('>OBSERVED<')
  })
})
