import { describe, expect, it } from 'vitest'
import type { CanonicalPerson, ExternalPlayer } from '../types'
import {
  canonicalPersonDisplayName,
  canonicalPersonSourceTrackletCount,
  identityConfidenceLabel,
  rosterConfirmationPayload,
  topRosterCandidates,
} from './identityResolution'

const roster: ExternalPlayer[] = [
  { id: 'player-8', name: 'Confirmed Eight', number: '8', position: 'Midfielder' },
  { id: 'player-10', name: 'Candidate Ten', number: '10', position: 'Forward' },
]

function canonicalPerson(overrides: Partial<CanonicalPerson> = {}): CanonicalPerson {
  return {
    canonicalPersonId: 'canonical-away-02',
    displayName: 'Away actor 02',
    identityStatus: 'provisional',
    identityConfidence: 0.834,
    identitySource: 'reid+trajectory',
    teamId: 'away',
    role: 'player',
    jerseyNumber: null,
    externalPlayerId: null,
    memberTrackletIds: ['tracklet-1', 'tracklet-1', 'tracklet-8'],
    evidence: [],
    rosterCandidates: [
      { externalPlayerId: 'player-10', confidence: 0.91, reasons: ['jersey number agrees'] },
      { externalPlayerId: 'player-8', confidence: 0.74 },
    ],
    conflicts: [],
    ...overrides,
  }
}

describe('canonical identity presentation', () => {
  it('never promotes the leading roster candidate into the canonical display name', () => {
    const identity = canonicalPerson()

    expect(topRosterCandidates(identity, roster)[0]?.name).toBe('Candidate Ten')
    expect(canonicalPersonDisplayName(identity, roster)).toBe('Away actor 02')
  })

  it('uses a roster name only after its external id is explicitly accepted', () => {
    const identity = canonicalPerson({
      externalPlayerId: 'player-8',
      identityStatus: 'resolved',
      identitySource: 'manual',
    })

    expect(canonicalPersonDisplayName(identity, roster)).toBe('Confirmed Eight')
  })

  it('sorts candidates deterministically, hydrates roster metadata, and limits output', () => {
    const identity = canonicalPerson({
      rosterCandidates: [
        { externalPlayerId: 'unknown-b', name: 'Unknown B', confidence: 0.5, rank: 2 },
        { externalPlayerId: 'player-8', confidence: 0.9, rank: 3 },
        { externalPlayerId: 'player-10', confidence: 0.9, rank: 1 },
        { externalPlayerId: 'unknown-a', name: 'Unknown A', confidence: -2 },
      ],
    })

    expect(topRosterCandidates(identity, roster, 3).map((candidate) => ({
      id: candidate.externalPlayerId,
      name: candidate.name,
      number: candidate.number,
    }))).toEqual([
      { id: 'player-10', name: 'Candidate Ten', number: '10' },
      { id: 'player-8', name: 'Confirmed Eight', number: '8' },
      { id: 'unknown-b', name: 'Unknown B', number: null },
    ])
  })

  it('formats bounded confidence and deduplicates source tracklets', () => {
    const identity = canonicalPerson()

    expect(identityConfidenceLabel(identity.identityConfidence)).toBe('83%')
    expect(identityConfidenceLabel(2)).toBe('100%')
    expect(identityConfidenceLabel(Number.NaN)).toBe('—')
    expect(canonicalPersonSourceTrackletCount(identity)).toBe(2)
  })

  it('builds an explicit roster confirmation command', () => {
    expect(rosterConfirmationPayload('canonical-away-02', 'player-10')).toEqual({
      canonicalPersonId: 'canonical-away-02',
      externalPlayerId: 'player-10',
    })
  })
})
