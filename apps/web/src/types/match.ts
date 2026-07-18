export type ExternalLineupEntry = {
  id: string
  player_id: string
  player_name: string
  team_id?: string | null
  team_name?: string | null
  side: 'home' | 'away' | 'unknown'
  position?: string | null
  number?: string | null
  role: 'starter' | 'substitute' | 'unknown'
  order: number
  /** Provider formation for this team, for example `4-3-3`. */
  formation?: string | null
  /** Provider-relative formation cell, for example `2:4`. */
  grid?: string | null
}

export type ExternalSubstitution = {
  id: string
  minute?: number | null
  team_id?: string | null
  team_name?: string | null
  player_out_id?: string | null
  player_out_name?: string | null
  player_in_id?: string | null
  player_in_name?: string | null
  label: string
}

export type ExternalRosterQuality = {
  status: 'automatic-ready' | 'partial' | 'unavailable'
  playerCount: number
  homePlayerCount: number
  awayPlayerCount: number
  automaticIdentityEligible: boolean
  manualIdentityEligible: boolean
  reasons: string[]
}

/** Stable server-side match-data adapter identifier. Keys never reach the web app. */
export type MatchDataProviderId = 'api-football' | 'thesportsdb' | (string & {})

export type MatchDataProvider = {
  id: MatchDataProviderId
  name: string
  configured: boolean
  available: boolean
  reason?: string | null
  capabilities?: string[]
}

export type MatchDataProviderCatalog = {
  providers: MatchDataProvider[]
  defaultProvider: MatchDataProviderId
  /** Configured product preference before availability fallback is applied. */
  preferredProvider?: MatchDataProviderId
}

export type ManualMatchImportRequest = {
  /** ManualMatchEvent intentionally has no provider-owned team/provider fields. */
  event: Omit<ExternalEvent, 'home' | 'away' | 'provider'>
  teams: { home: ExternalTeam; away: ExternalTeam }
  players: ExternalPlayer[]
  lineup?: ExternalLineupEntry[]
  timeline?: TimelineEvent[]
  substitutions?: ExternalSubstitution[]
  provenance?: {
    label?: string | null
    reference?: string | null
    capturedAt?: string | null
    notes?: string | null
  } | null
}

export type CanonicalMatchSyncState =
  | 'not-configured'
  | 'manual'
  | 'syncing'
  | 'synced'
  | 'ready'
  | 'partial'
  | 'unavailable'
  | 'failed'

export type CanonicalMatchTeam = {
  id: string
  name: string
  shortName?: string | null
  badgeUrl?: string | null
}

export type CanonicalRosterRole = 'starter' | 'substitute' | 'squad' | 'unknown'

export type CanonicalRosterPlayer = {
  id: string
  teamId: string
  name: string
  number?: string | null
  position?: string | null
  role: CanonicalRosterRole
  goalkeeper?: boolean
}

export type CanonicalMatchEventKind =
  | 'goal'
  | 'own-goal'
  | 'penalty-goal'
  | 'yellow-card'
  | 'red-card'
  | 'substitution'
  | 'var'
  | 'period-start'
  | 'period-end'
  | 'other'

export type CanonicalMatchEvent = {
  id: string
  kind: CanonicalMatchEventKind
  minute?: number | null
  addedTime?: number | null
  teamId?: string | null
  playerId?: string | null
  secondaryPlayerId?: string | null
  label: string
  detail?: string | null
}

export type CanonicalSubstitution = {
  id: string
  teamId?: string | null
  minute?: number | null
  addedTime?: number | null
  playerOutId?: string | null
  playerInId?: string | null
  label?: string | null
}

export type CanonicalMatch = {
  id: string
  revision: number
  snapshotId: string
  snapshotHash: string
  name?: string | null
  competition?: string | null
  season?: string | null
  kickoffAt?: string | null
  status?: string | null
  score: {
    home: number | null
    away: number | null
  }
  homeTeam: CanonicalMatchTeam
  awayTeam: CanonicalMatchTeam
  roster: CanonicalRosterPlayer[]
  events: CanonicalMatchEvent[]
  substitutions: CanonicalSubstitution[]
  sync: {
    state: CanonicalMatchSyncState
    syncedAt: string | null
    stale: boolean
    warnings: string[]
  }
}

/** Provider-neutral search result. Upstream identifiers remain server-side. */
export type MatchCandidate = {
  id: string
  name: string
  date?: string | null
  time?: string | null
  status?: string | null
  competition?: string | null
  season?: string | null
  homeTeam: { name: string; badge?: string | null }
  awayTeam: { name: string; badge?: string | null }
  score: { home: number | null; away: number | null }
  thumbnail?: string | null
}


export type ExternalTeam = {
  id: string
  name: string
  badge?: string | null
}

export type ExternalEvent = {
  id: string
  provider?: MatchDataProviderId | null
  name: string
  date?: string | null
  time?: string | null
  status?: string | null
  league?: string | null
  season?: string | null
  home: ExternalTeam
  away: ExternalTeam
  home_score?: number | null
  away_score?: number | null
  thumbnail?: string | null
}

export type ExternalPlayer = {
  id: string
  name: string
  team_id?: string | null
  team_name?: string | null
  position?: string | null
  number?: string | null
  thumbnail?: string | null
  lineup_role?: 'starter' | 'substitute' | 'unknown'
  lineup_order?: number | null
}

export type TimelineEvent = {
  id: string
  minute?: number | null
  type: string
  label: string
  player_id?: string | null
  player_name?: string | null
  team_id?: string | null
  team_name?: string | null
  secondary_player_id?: string | null
  secondary_player_name?: string | null
  detail?: string | null
}
