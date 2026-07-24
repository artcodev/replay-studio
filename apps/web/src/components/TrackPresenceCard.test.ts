import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import TrackPresenceCard from './TrackPresenceCard.vue'
import type { Track } from '../types/tracking'

const track: Track = {
  id: 'player-10',
  label: 'Player 10',
  teamId: 'away',
  color: '#327bff',
  number: 10,
  externalPlayerId: null,
  presence: {
    policy: 'observed-window-with-latent-gaps',
    coverage: 0.25,
    observationCount: 1,
    inferredKeyframeCount: 1,
    observedStart: 1,
    observedEnd: 1,
    observedSpanRatio: 0,
  },
  keyframes: [
    { t: 1, x: 2, z: 2, confidence: 0.9, observed: true, presenceState: 'observed', positionUncertaintyMetres: 0.8 },
    { t: 1.5, x: 2, z: 2, confidence: 0.18, observed: false, presenceState: 'inferred-gap', positionUncertaintyMetres: 3.2 },
  ],
}

describe('TrackPresenceCard', () => {
  it('renders current inference, uncertainty, and the observed/inferred sample mix', async () => {
    const html = await renderToString(createSSRApp(TrackPresenceCard, { track, currentTime: 0 }))

    expect(html).toContain('NOT OBSERVED')
    expect(html).toContain('Not reported')
    expect(html).toContain('Timeline presence')
    expect(html).toContain('25%')
    expect(html).toContain('Observed 50%')
    expect(html).toContain('Inferred 50%')
    expect(html).toContain('Sample evidence mix')
    expect(html).toContain('1 observed · 1 inferred keyframes')
    expect(html).toContain('aria-label="Sample evidence mix: observed 50%, inferred 50%"')
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
