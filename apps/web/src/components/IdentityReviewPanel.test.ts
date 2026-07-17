import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import type {
  IdentityReviewLinkCandidate,
  IdentityReviewObservation,
  IdentityReviewWorkerState,
} from '../lib/identityReview'
import type { CanonicalPerson, ExternalPlayer } from '../types'
import IdentityReviewPanel from './IdentityReviewPanel.vue'

const roster: ExternalPlayer[] = [
  { id: 'player-8', name: 'Candidate Eight', number: '8', position: 'Midfielder' },
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
    observations: [],
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
    rosterCandidates: [
      {
        externalPlayerId: 'player-8',
        rank: 1,
        confidence: 0.74,
        reasons: ['team agrees'],
      },
      {
        externalPlayerId: 'player-10',
        rank: 2,
        confidence: 0.91,
        reasons: ['jersey number agrees', 'team agrees'],
      },
    ],
    conflicts: [{
      id: 'conflict-1',
      code: 'jersey-read-disagreement',
      message: 'One independent crop reads number 18.',
      severity: 'review',
    }],
    ...overrides,
  }
}

const observations: IdentityReviewObservation[] = [
  {
    id: 'observation-crop',
    observationId: 'observation-crop',
    frameIndex: 24,
    sceneTime: 0.8,
    bbox: { x: 120, y: 80, width: 30, height: 90 },
    cropUrl: '/api/artifacts/crop-24.jpg',
    quality: 0.94,
    confidence: 0.91,
  },
  {
    id: 'observation-frame',
    observationId: 'observation-frame',
    frameIndex: 12,
    sceneTime: 0.4,
    bbox: { x: 300, y: 90, width: 35, height: 94 },
    frameWidth: 1280,
    frameHeight: 720,
    previewUrl: '/api/artifacts/frame-12.jpg',
    quality: 0.81,
    confidence: 0.88,
    rejectionReasons: ['back visibility low'],
  },
]

const workers: IdentityReviewWorkerState[] = [
  {
    id: 'reid',
    label: 'Player ReID',
    status: 'ready',
    backend: 'prtreid-bpbreid-soccernet',
    modelVersion: 'checkpoint-31',
    requestedCount: 41,
    usableCount: 12,
    rejectedCount: 29,
    rejectionReasons: ['crop too small'],
  },
  {
    id: 'jersey-ocr',
    label: 'Jersey OCR',
    status: 'unavailable',
    detail: 'Connection refused',
  },
]

const links: IdentityReviewLinkCandidate[] = [
  {
    id: 'edge-review',
    targetCanonicalPersonId: 'canonical-away-07',
    targetLabel: 'Away actor 07',
    status: 'review',
    confidence: 0.78,
    source: 'reid',
    reasons: ['appearance close', 'ambiguous successor'],
  },
  {
    id: 'edge-rejected',
    targetCanonicalPersonId: 'canonical-home-04',
    status: 'rejected',
    reasons: ['team conflict'],
  },
]

describe('IdentityReviewPanel', () => {
  it('renders ranked evidence, crops, worker readiness, link rejections, and conflicts honestly', async () => {
    const html = await renderToString(createSSRApp(IdentityReviewPanel, {
      identity: identity(),
      rosterPlayers: roster,
      observations,
      workerStates: workers,
      linkCandidates: links,
    }))

    expect(html).toContain('Selected canonical person evidence summary')
    expect(html).toContain('Away actor 02')
    expect(html).toContain('86%')
    expect(html).toContain('Appearance agreement')
    expect(html).toContain('/api/artifacts/crop-24.jpg')
    expect(html).toContain('aria-label="Inspect frame 24 for Away actor 02"')
    expect(html).toContain('back visibility low')
    expect(html).toContain('Player ReID')
    expect(html).toContain('prtreid-bpbreid-soccernet · checkpoint-31')
    expect(html).toContain('Requested 41 · usable 12 · rejected 29')
    expect(html).toContain('Connection refused')
    expect(html).toContain('Rejected by resolver')
    expect(html).toContain('team conflict')
    expect(html).toContain('jersey read disagreement')
    const rankedCandidateHtml = html.slice(html.indexOf('Roster candidates'))
    expect(rankedCandidateHtml.indexOf('Candidate Ten')).toBeLessThan(rankedCandidateHtml.indexOf('Candidate Eight'))
    expect(html).toContain('Ranked suggestions remain hypotheses until explicitly bound.')
  })

  it('distinguishes unavailable diagnostics from supplied empty results', async () => {
    const unavailable = await renderToString(createSSRApp(IdentityReviewPanel, {
      identity: identity({ evidence: [], rosterCandidates: [], conflicts: [] }),
      rosterPlayers: roster,
    }))
    const empty = await renderToString(createSSRApp(IdentityReviewPanel, {
      identity: identity({ evidence: [], rosterCandidates: [], conflicts: [] }),
      rosterPlayers: roster,
      observations: [],
      workerStates: [],
      linkCandidates: [],
    }))

    expect(unavailable).toContain('No observation previews are available')
    expect(unavailable).toContain('Worker readiness was not supplied')
    expect(unavailable).not.toContain('Identity-link review')
    expect(empty).toContain('No reviewable crop, frame preview, or valid detector box was supplied')
    expect(empty).toContain('No ReID or jersey OCR workers are configured')
    expect(empty).toContain('There are no pending or rejected identity-link hypotheses')
    expect(empty).toContain('The resolver produced no roster candidates')
    expect(empty).toContain('No published identity conflicts')
  })

  it('renders a focused accessible prompt when no canonical person is selected', async () => {
    const html = await renderToString(createSSRApp(IdentityReviewPanel))

    expect(html).toContain('No person selected')
    expect(html).toContain('Select a canonical person in the video or 3D scene')
    expect(html).not.toContain('Best observations')
    expect(html).not.toContain('Roster candidates')
  })

  it('declares explicit review events and renders accessible action controls', async () => {
    const component = IdentityReviewPanel as unknown as {
      emits?: string[] | Record<string, unknown>
    }
    const declaredEvents = Array.isArray(component.emits)
      ? component.emits
      : Object.keys(component.emits ?? {})
    const html = await renderToString(createSSRApp(IdentityReviewPanel, {
      identity: identity(),
      rosterPlayers: roster,
      observations,
      workerStates: workers,
      linkCandidates: links,
    }))

    expect(declaredEvents).toEqual(expect.arrayContaining([
      'bind-candidate',
      'reject-candidate',
      'cannot-link',
      'inspect-frame',
      'unbind-roster',
      'clear-roster-binding',
    ]))
    expect(html).toContain('aria-label="Bind Candidate Ten to Away actor 02"')
    expect(html).toContain('aria-label="Reject Candidate Eight as a candidate for Away actor 02"')
    expect(html).toContain('aria-label="Reject identity link to Away actor 07"')
    expect(html).toContain('aria-label="Mark Away actor 02 and Away actor 07 as cannot-link"')
    expect(html).toContain('aria-label="Inspect frame 24 for Away actor 02"')
  })

  it('keeps confirmed-binding undo controls without the legacy inspector', async () => {
    const confirmed = await renderToString(createSSRApp(IdentityReviewPanel, {
      identity: identity({ externalPlayerId: 'player-10', identityStatus: 'resolved' }),
      rosterPlayers: roster,
    }))
    const unbound = await renderToString(createSSRApp(IdentityReviewPanel, {
      identity: identity({ externalPlayerId: null }),
      rosterPlayers: roster,
      dedicatedUnbindActive: true,
    }))

    expect(confirmed).toContain('aria-label="Unbind roster player from Candidate Ten"')
    expect(unbound).toContain('aria-label="Clear manual roster Unbind for Away actor 02"')
    expect(unbound).toContain('explicit Unbind decision blocks roster proposals')
  })

  it('keeps all 52 saved players manually selectable when the resolver abstains', async () => {
    const fullRoster: ExternalPlayer[] = Array.from({ length: 52 }, (_, index) => ({
      id: `player-${index + 1}`,
      name: `Player ${index + 1}`,
      number: String(index % 26 + 1),
      team_id: index < 26 ? 'spain' : 'belgium',
      team_name: index < 26 ? 'Spain' : 'Belgium',
    }))
    const html = await renderToString(createSSRApp(IdentityReviewPanel, {
      identity: identity({ rosterCandidates: [] }),
      rosterPlayers: fullRoster,
      observations: [],
      workerStates: [],
      linkCandidates: [],
    }))

    expect(html).toContain('Manual roster binding')
    expect(html).toContain('aria-label="Manual roster player"')
    expect(html).toContain('aria-label="Bind selected roster player"')
    expect(html).toContain('BIND SELECTED')
    expect(html).toContain('value="player-52"')
    expect(html).toContain('#26 · Player 52 · Belgium')
    expect((html.match(/<option/g) ?? []).length).toBe(53)
    expect(html).toContain('The resolver produced no roster candidates')
  })

  it('initializes the manual picker from the current binding and disables a no-op bind', async () => {
    const fullRoster: ExternalPlayer[] = Array.from({ length: 52 }, (_, index) => ({
      id: `player-${index + 1}`,
      name: `Player ${index + 1}`,
      number: String(index % 26 + 1),
      team_id: index < 26 ? 'spain' : 'belgium',
      team_name: index < 26 ? 'Spain' : 'Belgium',
    }))
    const html = await renderToString(createSSRApp(IdentityReviewPanel, {
      identity: identity({
        externalPlayerId: 'player-37',
        identityStatus: 'resolved',
        rosterCandidates: [],
      }),
      rosterPlayers: fullRoster,
    }))

    expect(html).toMatch(/<option(?=[^>]*value="player-37")(?=[^>]*selected)[^>]*>/)
    expect(html).toMatch(/<button(?=[^>]*disabled)(?=[^>]*aria-label="Bind selected roster player")[^>]*>/)
    expect(html).toContain('Current binding · #11 · Player 37 · Belgium')
    expect(html).toContain('aria-label="Unbind roster player from Player 37"')
  })

  it('fails closed when imported roster rows reuse one external player ID', async () => {
    const html = await renderToString(createSSRApp(IdentityReviewPanel, {
      identity: identity({
        externalPlayerId: 'duplicate-player',
        rosterCandidates: [],
      }),
      rosterPlayers: [
        { id: 'duplicate-player', name: 'First row', number: '7', team_name: 'Spain' },
        { id: 'duplicate-player', name: 'Second row', number: '17', team_name: 'Spain' },
      ],
    }))

    expect((html.match(/duplicate ID/g) ?? []).length).toBe(2)
    expect(html).toContain('This external player ID occurs 2 times')
    expect(html).toMatch(/<button(?=[^>]*disabled)(?=[^>]*aria-label="Bind selected roster player")[^>]*>/)
  })
})
