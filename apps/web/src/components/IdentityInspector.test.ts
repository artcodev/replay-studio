import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import type { CanonicalPerson } from '../types/identity'
import type { ExternalPlayer } from '../types/match'
import IdentityInspector from './IdentityInspector.vue'

const roster: ExternalPlayer[] = [
  { id: 'player-8', name: 'Confirmed Eight', number: '8', position: 'Midfielder' },
  { id: 'player-10', name: 'Candidate Ten', number: '10', position: 'Forward' },
]

function identity(overrides: Partial<CanonicalPerson> = {}): CanonicalPerson {
  return {
    canonicalPersonId: 'canonical-away-02',
    displayName: 'Away actor 02',
    identityStatus: 'provisional',
    identityConfidence: 0.86,
    identitySource: 'reid+trajectory',
    teamId: 'away',
    role: 'player',
    jerseyNumber: '10',
    externalPlayerId: null,
    memberTrackletIds: ['tracklet-a', 'tracklet-b'],
    observationCount: 41,
    evidence: [{
      id: 'reid-1',
      kind: 'reid',
      label: 'Appearance agreement',
      confidence: 0.91,
      supportCount: 12,
      sampleCount: 14,
      source: 'identity-worker',
      model: 'osnet',
    }],
    rosterCandidates: [{
      externalPlayerId: 'player-10',
      rank: 1,
      score: 0.88,
      reasons: ['jersey number agrees', 'team agrees'],
    }],
    conflicts: [{
      id: 'conflict-1',
      code: 'jersey-read-disagreement',
      message: 'One crop reads number 18.',
      severity: 'review',
    }],
    ...overrides,
  }
}

describe('IdentityInspector', () => {
  it('shows canonical evidence and candidates without presenting a suggestion as confirmed', async () => {
    const html = await renderToString(createSSRApp(IdentityInspector, {
      identity: identity(),
      rosterPlayers: roster,
    }))

    expect(html).toMatch(/<h3[^>]*>Away actor 02<\/h3>/)
    expect(html).toContain('Anonymous identity · no confirmed roster binding')
    expect(html).toContain('Identity confidence')
    expect(html).toContain('86%')
    expect(html).toContain('Source tracklets')
    expect(html).toContain('Appearance agreement')
    expect(html).toContain('identity-worker · osnet')
    expect(html).toContain('jersey read disagreement')
    expect(html).toContain('#10 · Candidate Ten')
    expect(html).toContain('Roster binding')
    expect(html).toContain('Choose any player, including one not suggested by the resolver.')
    expect(html).toContain('aria-label="Full roster player"')
    expect(html).toContain('#8 · Confirmed Eight')
    expect(html).toContain('aria-label="Bind selected roster player"')
    expect(html).toContain('Suggestions stay unconfirmed until you choose one.')
    expect(html).not.toContain('Confirmed roster binding')
  })

  it('uses a roster name in the heading only for an accepted external player id', async () => {
    const html = await renderToString(createSSRApp(IdentityInspector, {
      identity: identity({
        externalPlayerId: 'player-8',
        identityStatus: 'resolved',
        identitySource: 'manual',
      }),
      rosterPlayers: roster,
    }))

    expect(html).toMatch(/<h3[^>]*>Confirmed Eight<\/h3>/)
    expect(html).toContain('Confirmed roster binding · #8 · Confirmed Eight')
    expect(html).toContain('>RESOLVED<')
    expect(html).toContain('aria-label="Unbind roster player from Confirmed Eight"')
    expect(html).toContain('> UNBIND ROSTER PLAYER </button>')
  })

  it('declares the confirmation event and renders an explicit candidate action', async () => {
    const component = IdentityInspector as unknown as {
      emits?: string[] | Record<string, unknown>
    }
    const declaredEvents = Array.isArray(component.emits)
      ? component.emits
      : Object.keys(component.emits ?? {})
    const html = await renderToString(createSSRApp(IdentityInspector, {
      identity: identity(),
      rosterPlayers: roster,
    }))

    expect(declaredEvents).toContain('confirm-roster')
    expect(declaredEvents).toContain('unbind-roster')
    expect(declaredEvents).toContain('clear-roster-binding')
    expect(html).toContain('aria-label="Confirm Candidate Ten for Away actor 02"')
    expect(html).toContain('>CONFIRM</button>')
  })

  it('shows clear only for an explicit dedicated Unbind tombstone', async () => {
    const withoutTombstone = await renderToString(createSSRApp(IdentityInspector, {
      identity: identity({ externalPlayerId: null }),
      rosterPlayers: roster,
    }))
    const withTombstone = await renderToString(createSSRApp(IdentityInspector, {
      identity: identity({ externalPlayerId: null }),
      rosterPlayers: roster,
      dedicatedUnbindActive: true,
    }))

    expect(withoutTombstone).not.toContain('CLEAR MANUAL UNBIND')
    expect(withTombstone).toContain('CLEAR MANUAL UNBIND')
    expect(withTombstone).toContain('aria-label="Clear manual roster Unbind for Away actor 02"')
    expect(withTombstone).toContain('Remove the explicit Unbind tombstone')
  })

  it('keeps the complete roster bind control available without resolver candidates', async () => {
    const html = await renderToString(createSSRApp(IdentityInspector, {
      identity: identity({ rosterCandidates: [] }),
      rosterPlayers: roster,
    }))

    expect(html).toContain('aria-label="Full roster player"')
    expect(html).toContain('value="player-8"')
    expect(html).toContain('value="player-10"')
    expect(html).toContain('> BIND </button>')
    expect(html).toContain('No roster candidates are available.')
  })

  it('renders the real multi-angle identity evidence contract with a stable label', async () => {
    const multiAngleEvidence = {
      id: 'angle-31ab2f:identity-evidence:source-8',
      kind: 'multi-angle-identity' as const,
      label: 'Aligned replay identity evidence',
      sourceSceneId: 'source-angle',
      sourceCanonicalPersonId: 'source-8',
      signals: ['external-player-match', 'reliable-jersey-match'],
      confidence: 0.9,
      alignmentConfidence: 0.82,
      alignmentMethod: 'motion-dtw',
      observationCount: 14,
    }
    const html = await renderToString(createSSRApp(IdentityInspector, {
      identity: identity({
        evidence: [multiAngleEvidence],
        multiAngleEvidence: [{
          ...multiAngleEvidence,
          observations: [{
            observationId: 'angle-31ab2f:observation:foreign-observation',
            frameIndex: 4,
            sceneTime: 1.2,
            bbox: { x: 120, y: 80, width: 30, height: 90 },
            confidence: 0.87,
          }],
          sourceTrackletIds: ['angle-31ab2f:tracklet:source-track-8'],
        }],
        sourcePassIds: ['source-angle'],
      }),
      rosterPlayers: roster,
    }))

    expect(html).toContain('Multi-angle identity · Aligned replay identity evidence')
    expect(html).toContain('90%')
    expect(html).not.toContain('undefined')
  })
})
