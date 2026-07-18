import { identityConfidenceLabel } from '../../lib/identityResolution'
import type { IdentityReviewObservation } from '../../lib/identityReview'
import type { CanonicalIdentityEvidence } from '../../types/identity'
import type { ExternalPlayer } from '../../types/match'

function rosterNumberSortValue(value: string | null | undefined): number {
  if (!value?.trim()) return Number.POSITIVE_INFINITY
  const number = Number(value)
  return Number.isFinite(number) ? number : Number.POSITIVE_INFINITY
}

export function orderedIdentityRosterPlayers(players: readonly ExternalPlayer[]): ExternalPlayer[] {
  return [...players].sort((left, right) => (
    String(left.team_name || left.team_id || '').localeCompare(String(right.team_name || right.team_id || ''))
    || rosterNumberSortValue(left.number) - rosterNumberSortValue(right.number)
    || left.name.localeCompare(right.name)
    || left.id.localeCompare(right.id)
  ))
}

export function identityRosterPlayerLabel(player: ExternalPlayer): string {
  return [
    player.number ? `#${player.number}` : null,
    player.name,
    player.team_name || player.team_id || null,
  ].filter(Boolean).join(' · ')
}

export function formatIdentityReviewTime(seconds: number): string {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
  const minutes = Math.floor(safe / 60)
  const remainder = safe - minutes * 60
  return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(3).padStart(6, '0')}`
}

export function identityEvidenceDetail(evidence: CanonicalIdentityEvidence): string {
  const parts: string[] = []
  if (evidence.value !== null && evidence.value !== undefined && String(evidence.value).trim()) {
    parts.push(String(evidence.value))
  }
  if (evidence.confidence !== null && evidence.confidence !== undefined) {
    parts.push(identityConfidenceLabel(evidence.confidence))
  }
  if (evidence.supportCount !== undefined) {
    parts.push(evidence.sampleCount !== undefined
      ? `${evidence.supportCount}/${evidence.sampleCount} samples`
      : `${evidence.supportCount} samples`)
  }
  return parts.join(' · ') || 'Recorded evidence'
}

export function identityObservationBoxStyle(
  observation: IdentityReviewObservation,
): Record<string, string> | null {
  const box = observation.bbox
  const width = Number(observation.frameWidth)
  const height = Number(observation.frameHeight)
  if (!box || !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null
  }
  const clamp = (value: number) => Math.min(100, Math.max(0, value))
  return {
    left: `${clamp(box.x / width * 100)}%`,
    top: `${clamp(box.y / height * 100)}%`,
    width: `${clamp(box.width / width * 100)}%`,
    height: `${clamp(box.height / height * 100)}%`,
  }
}
