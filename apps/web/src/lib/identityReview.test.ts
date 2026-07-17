import { describe, expect, it } from 'vitest'
import {
  cannotLinkReviewDecision,
  canonicalReviewObservations,
  identityReviewWorkerStatusLabel,
  identityReviewItemObservations,
  identityReviewWorkerStates,
  inspectIdentityObservationDecision,
  linkReviewDecision,
  manualRosterBindingDecision,
  observationHasReviewEvidence,
  rosterReviewDecision,
  topIdentityReviewObservations,
  type IdentityReviewObservation,
} from './identityReview'

function observation(
  id: string,
  overrides: Partial<IdentityReviewObservation> = {},
): IdentityReviewObservation {
  return {
    id,
    frameIndex: 1,
    sceneTime: 0.1,
    bbox: { x: 10, y: 12, width: 24, height: 58 },
    confidence: 0.8,
    ...overrides,
  }
}

describe('identity review presentation helpers', () => {
  it('ranks review observations by explicit quality and keeps the result bounded', () => {
    const rows = [
      observation('confidence-only', { confidence: 0.99 }),
      observation('quality-high', { quality: 0.92, confidence: 0.4 }),
      observation('quality-low', { quality: 0.72, cropUrl: '/crop.jpg' }),
    ]

    expect(topIdentityReviewObservations(rows, 2).map((item) => item.id)).toEqual([
      'quality-high',
      'quality-low',
    ])
    expect(topIdentityReviewObservations(rows, 0)).toEqual([])
  })

  it('fails closed when an observation has neither a URL nor a usable box', () => {
    expect(observationHasReviewEvidence(observation('url', {
      bbox: null,
      previewUrl: '/frame.jpg',
    }))).toBe(true)
    expect(observationHasReviewEvidence(observation('box'))).toBe(true)
    expect(observationHasReviewEvidence(observation('invalid', {
      bbox: { x: 1, y: 1, width: 0, height: 20 },
      previewUrl: ' ',
    }))).toBe(false)
  })

  it('maps persisted observations without inventing an image preview', () => {
    const rows = canonicalReviewObservations({
      observations: [{
        observationId: 'observation-8',
        frameIndex: 8,
        sceneTime: 0.8,
        bbox: { x: 20, y: 30, width: 40, height: 90 },
        confidence: 0.91,
        metricStatus: 'rejected',
        metricReason: 'calibration unavailable',
      }],
    })

    expect(rows).toEqual([{
      id: 'observation-8',
      observationId: 'observation-8',
      frameIndex: 8,
      sceneTime: 0.8,
      bbox: { x: 20, y: 30, width: 40, height: 90 },
      confidence: 0.91,
      source: null,
      rejectionReasons: ['calibration unavailable'],
    }])
    expect(rows[0]).not.toHaveProperty('previewUrl')
  })

  it('uses explicit worker-state labels', () => {
    expect(identityReviewWorkerStatusLabel('ready')).toBe('Ready')
    expect(identityReviewWorkerStatusLabel('invalid-response')).toBe('Invalid response')
    expect(identityReviewWorkerStatusLabel('no-observations')).toBe('No observations')
  })

  it('builds explicit, typed commands for every review decision', () => {
    const link = { id: 'edge-7', targetCanonicalPersonId: 'canonical-away-07' }
    const crop = observation('observation-24', {
      observationId: 'immutable-24',
      frameIndex: 24,
      sceneTime: 0.8,
    })

    expect(rosterReviewDecision('canonical-away-02', 'player-10')).toEqual({
      canonicalPersonId: 'canonical-away-02',
      kind: 'roster',
      candidateId: 'player-10',
      externalPlayerId: 'player-10',
    })
    expect(linkReviewDecision('canonical-away-02', link)).toEqual({
      canonicalPersonId: 'canonical-away-02',
      kind: 'identity-link',
      candidateId: 'edge-7',
      targetCanonicalPersonId: 'canonical-away-07',
    })
    expect(cannotLinkReviewDecision('canonical-away-02', link)).toEqual({
      canonicalPersonId: 'canonical-away-02',
      candidateId: 'edge-7',
      targetCanonicalPersonId: 'canonical-away-07',
    })
    expect(inspectIdentityObservationDecision('canonical-away-02', crop)).toEqual({
      canonicalPersonId: 'canonical-away-02',
      observationId: 'immutable-24',
      frameIndex: 24,
      sceneTime: 0.8,
      bbox: { x: 10, y: 12, width: 24, height: 58 },
    })
  })

  it('builds manual roster bindings only for one unique saved player', () => {
    const fullRoster = Array.from({ length: 52 }, (_, index) => ({
      id: `player-${index + 1}`,
    }))

    expect(manualRosterBindingDecision(
      'canonical-away-02',
      null,
      'player-52',
      fullRoster,
    )).toEqual({
      canonicalPersonId: 'canonical-away-02',
      externalPlayerId: 'player-52',
    })
    expect(manualRosterBindingDecision(
      'canonical-away-02',
      'player-52',
      'player-52',
      fullRoster,
    )).toBeNull()
    expect(manualRosterBindingDecision(
      'canonical-away-02',
      null,
      'duplicate-player',
      [...fullRoster, { id: 'duplicate-player' }, { id: 'duplicate-player' }],
    )).toBeNull()
    expect(manualRosterBindingDecision(
      'canonical-away-02',
      null,
      'missing-player',
      fullRoster,
    )).toBeNull()
  })

  it('adapts backend review crops and worker health without inventing evidence', () => {
    expect(identityReviewItemObservations({
      representativeObservations: [{
        observationId: 'obs-8',
        frameIndex: 8,
        sceneTime: 0.32,
        bbox: { x: 12, y: 20, width: 30, height: 70 },
        confidence: 0.87,
        reviewQuality: 1.14,
        cropUrl: '/api/crop/obs-8',
        reid: { status: 'rejected', rejectionReasons: ['too-small'] },
        jerseyOcr: { status: 'ambiguous', rejectionReasons: ['two-digits-disagree'] },
      }],
    })).toEqual([{
      id: 'obs-8',
      observationId: 'obs-8',
      frameIndex: 8,
      sceneTime: 0.32,
      bbox: { x: 12, y: 20, width: 30, height: 70 },
      cropUrl: '/api/crop/obs-8',
      confidence: 0.87,
      quality: 1.14,
      rejectionReasons: ['too-small', 'two-digits-disagree'],
    }])

    expect(identityReviewWorkerStates({
      workers: {
        identity: {
          status: 'ready',
          backend: 'prtreid',
          modelVersion: 'v1',
          requestedObservationCount: 12,
          usableObservationCount: 8,
          rejectedObservationCount: 4,
        },
        jerseyOcr: { status: 'unavailable', detail: 'connection refused' },
      },
    })).toEqual([
      expect.objectContaining({
        id: 'reid',
        status: 'ready',
        requestedCount: 12,
        usableCount: 8,
        rejectedCount: 4,
      }),
      expect.objectContaining({
        id: 'jersey-ocr',
        status: 'unavailable',
        detail: 'connection refused',
      }),
    ])
    expect(identityReviewWorkerStates(null)).toBeNull()
  })
})
