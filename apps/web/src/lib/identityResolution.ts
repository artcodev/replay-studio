import type {
  CanonicalIdentityEvidenceKind,
  CanonicalPerson,
  CanonicalPersonStatus,
  CanonicalRosterCandidate,
  ExternalPlayer,
} from '../types'

export type ResolvedRosterCandidate = Omit<CanonicalRosterCandidate, 'confidence'> & {
  confidence: number | null
  name: string
  number: string | null
  position: string | null
  teamId: string | null
}

const STATUS_LABELS: Record<CanonicalPersonStatus, string> = {
  resolved: 'RESOLVED',
  provisional: 'PROVISIONAL',
  excluded: 'EXCLUDED',
}

const EVIDENCE_LABELS: Record<CanonicalIdentityEvidenceKind, string> = {
  manual: 'Manual',
  reid: 'ReID',
  'jersey-ocr': 'Jersey OCR',
  team: 'Team',
  role: 'Role',
  trajectory: 'Trajectory',
  'multi-pass': 'Multi-pass',
  'multi-angle-identity': 'Multi-angle identity',
  'roster-prior': 'Roster prior',
}

export function normalizedIdentityConfidence(value: number | null | undefined): number | null {
  if (value === null || value === undefined || !Number.isFinite(value)) return null
  return Math.min(1, Math.max(0, value))
}

export function identityConfidenceLabel(value: number | null | undefined): string {
  const normalized = normalizedIdentityConfidence(value)
  return normalized === null ? '—' : `${Math.round(normalized * 100)}%`
}

export function canonicalPersonStatusLabel(status: CanonicalPersonStatus): string {
  return STATUS_LABELS[status]
}

export function identityEvidenceKindLabel(kind: CanonicalIdentityEvidenceKind): string {
  return EVIDENCE_LABELS[kind]
}

export function confirmedRosterPlayer(
  identity: Pick<CanonicalPerson, 'externalPlayerId'>,
  rosterPlayers: readonly ExternalPlayer[],
): ExternalPlayer | null {
  if (!identity.externalPlayerId) return null
  return rosterPlayers.find((player) => player.id === identity.externalPlayerId) ?? null
}

/**
 * The display name only uses an accepted binding. Suggested roster candidates
 * are deliberately ignored so the UI cannot present a hypothesis as a fact.
 */
export function canonicalPersonDisplayName(
  identity: Pick<CanonicalPerson, 'canonicalPersonId' | 'displayName' | 'externalPlayerId'>,
  rosterPlayers: readonly ExternalPlayer[],
): string {
  const confirmed = confirmedRosterPlayer(identity, rosterPlayers)
  return confirmed?.name.trim() || identity.displayName.trim() || identity.canonicalPersonId
}

export function canonicalPersonSourceTrackletCount(
  identity: Pick<CanonicalPerson, 'memberTrackletIds'>,
): number {
  return new Set(identity.memberTrackletIds.filter(Boolean)).size
}

function candidateRank(candidate: CanonicalRosterCandidate): number {
  return candidate.rank !== null
    && candidate.rank !== undefined
    && Number.isFinite(candidate.rank)
    && candidate.rank >= 0
    ? candidate.rank
    : Number.POSITIVE_INFINITY
}

function hydrateCandidate(
  candidate: CanonicalRosterCandidate,
  rosterPlayers: readonly ExternalPlayer[],
): ResolvedRosterCandidate {
  const rosterPlayer = rosterPlayers.find((player) => player.id === candidate.externalPlayerId)
  return {
    ...candidate,
    confidence: candidate.confidence ?? candidate.score ?? null,
    name: candidate.name?.trim() || rosterPlayer?.name.trim() || candidate.externalPlayerId,
    number: candidate.number ?? rosterPlayer?.number ?? null,
    position: candidate.position ?? rosterPlayer?.position ?? null,
    teamId: candidate.teamId ?? rosterPlayer?.team_id ?? null,
  }
}

export function topRosterCandidates(
  identity: Pick<CanonicalPerson, 'rosterCandidates'>,
  rosterPlayers: readonly ExternalPlayer[] = [],
  limit = 3,
): ResolvedRosterCandidate[] {
  if (!Number.isFinite(limit) || limit <= 0) return []
  return identity.rosterCandidates
    .map((candidate) => hydrateCandidate(candidate, rosterPlayers))
    .sort((left, right) => (
      (normalizedIdentityConfidence(right.confidence) ?? -1)
      - (normalizedIdentityConfidence(left.confidence) ?? -1)
      || candidateRank(left) - candidateRank(right)
      || left.externalPlayerId.localeCompare(right.externalPlayerId)
    ))
    .slice(0, Math.floor(limit))
}

export function rosterConfirmationPayload(
  canonicalPersonId: string,
  externalPlayerId: string,
): { canonicalPersonId: string; externalPlayerId: string } {
  return { canonicalPersonId, externalPlayerId }
}
