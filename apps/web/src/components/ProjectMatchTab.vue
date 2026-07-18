<script setup lang="ts">
import { computed } from 'vue'
import type {
  CanonicalMatch,
  CanonicalMatchEvent,
  CanonicalRosterPlayer,
  CanonicalRosterRole,
} from '../types/match'

const props = withDefaults(defineProps<{
  match?: CanonicalMatch | null
  loading?: boolean
  busy?: boolean
  disabled?: boolean
  allowImport?: boolean
}>(), {
  match: null,
  loading: false,
  busy: false,
  disabled: false,
  allowImport: true,
})

const emit = defineEmits<{
  refresh: []
  import: []
}>()

const roleOrder: Record<CanonicalRosterRole, number> = {
  starter: 0,
  substitute: 1,
  squad: 2,
  unknown: 3,
}

const eventKindLabels: Record<CanonicalMatchEvent['kind'], string> = {
  goal: 'Goal',
  'own-goal': 'Own goal',
  'penalty-goal': 'Penalty goal',
  'yellow-card': 'Yellow card',
  'red-card': 'Red card',
  substitution: 'Substitution',
  var: 'VAR',
  'period-start': 'Period start',
  'period-end': 'Period end',
  other: 'Event',
}

const syncLabels: Record<CanonicalMatch['sync']['state'], string> = {
  'not-configured': 'Not connected',
  manual: 'Manual data',
  syncing: 'Syncing',
  synced: 'Synced',
  ready: 'Synced',
  partial: 'Partial data',
  unavailable: 'Unavailable',
  failed: 'Sync failed',
}

const isManual = computed(() => props.match?.sync.state === 'manual')
const canRefresh = computed(() => Boolean(
  props.match
  && !props.disabled
  && !props.busy
  && props.match.sync.state !== 'manual'
  && props.match.sync.state !== 'not-configured'
  && props.match.sync.state !== 'syncing',
))

const homeRoster = computed(() => rosterForTeam(props.match?.homeTeam.id))
const awayRoster = computed(() => rosterForTeam(props.match?.awayTeam.id))
const orderedEvents = computed(() => [...(props.match?.events ?? [])].sort((left, right) => (
  (left.minute ?? Number.MAX_SAFE_INTEGER) - (right.minute ?? Number.MAX_SAFE_INTEGER)
  || (left.addedTime ?? 0) - (right.addedTime ?? 0)
  || left.id.localeCompare(right.id)
)))

function rosterForTeam(teamId?: string) {
  if (!teamId) return []
  return (props.match?.roster ?? [])
    .filter((player) => player.teamId === teamId)
    .sort((left, right) => (
      roleOrder[left.role] - roleOrder[right.role]
      || Number(left.number ?? Number.MAX_SAFE_INTEGER) - Number(right.number ?? Number.MAX_SAFE_INTEGER)
      || left.name.localeCompare(right.name)
    ))
}

function playerLabel(player: CanonicalRosterPlayer) {
  return `${player.number ? `#${player.number} · ` : ''}${player.name}`
}

function playerName(playerId?: string | null) {
  return props.match?.roster.find((player) => player.id === playerId)?.name ?? null
}

function matchTime(minute?: number | null, addedTime?: number | null) {
  if (minute === null || minute === undefined) return '—'
  return `${minute}${addedTime ? `+${addedTime}` : ''}′`
}

function scoreValue(value: number | null) {
  return value === null ? '–' : String(value)
}

function kickoffLabel(value?: string | null) {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString([], {
    dateStyle: 'medium',
    timeStyle: 'short',
  })
}

function syncedLabel(value: string | null) {
  if (!value) return 'Sync time unavailable'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return `Updated ${date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}`
}
</script>

<template>
  <section class="project-match-tab" aria-labelledby="project-match-title">
    <header>
      <div>
        <p class="eyebrow">Canonical project data</p>
        <h2 id="project-match-title">Match</h2>
      </div>
      <div v-if="match" class="sync-state" :class="`sync-${match.sync.state}`">
        <strong>{{ syncLabels[match.sync.state] }}</strong>
        <span v-if="match.sync.stale">Stale</span>
      </div>
    </header>

    <p v-if="loading && !match" class="match-state" role="status">Loading match data…</p>
    <div v-else-if="!match" class="match-empty">
      <strong>No match assigned</strong>
      <p>{{ allowImport ? 'Import a roster and event timeline now. Automatic synchronization can be configured later, or choose a catalog match.' : 'Use the match catalog above to select a fixture and load its roster and events.' }}</p>
      <button v-if="allowImport" type="button" :disabled="disabled || busy" @click="emit('import')">Import match JSON</button>
    </div>

    <template v-else>
      <div class="match-identity">
        <div class="match-meta">
          <strong>{{ match.name || `${match.homeTeam.name} vs ${match.awayTeam.name}` }}</strong>
          <span v-if="match.competition">{{ match.competition }}<template v-if="match.season"> · {{ match.season }}</template></span>
          <span v-if="kickoffLabel(match.kickoffAt)">{{ kickoffLabel(match.kickoffAt) }}</span>
          <span v-if="match.status">Status · {{ match.status }}</span>
        </div>
        <div class="scoreboard" :aria-label="`${match.homeTeam.name} ${scoreValue(match.score.home)}, ${match.awayTeam.name} ${scoreValue(match.score.away)}`">
          <div>
            <img v-if="match.homeTeam.badgeUrl" :src="match.homeTeam.badgeUrl" alt="" />
            <span>{{ match.homeTeam.shortName || match.homeTeam.name }}</span>
          </div>
          <strong>{{ scoreValue(match.score.home) }}<i>:</i>{{ scoreValue(match.score.away) }}</strong>
          <div>
            <img v-if="match.awayTeam.badgeUrl" :src="match.awayTeam.badgeUrl" alt="" />
            <span>{{ match.awayTeam.shortName || match.awayTeam.name }}</span>
          </div>
        </div>
      </div>

      <div v-if="isManual" class="manual-state" role="status">
        <div>
          <strong>Manual match data</strong>
          <span>This snapshot changes only when you import a replacement file.</span>
        </div>
        <button v-if="allowImport" type="button" :disabled="disabled || busy" @click="emit('import')">Replace JSON</button>
      </div>
      <div v-else class="automatic-sync-state">
        <div>
          <strong>{{ syncedLabel(match.sync.syncedAt) }}</strong>
          <span v-if="match.sync.stale">The saved snapshot may be outdated. Refresh before identity review.</span>
          <span v-else>Roster and events are saved as a project snapshot.</span>
        </div>
        <button type="button" :disabled="!canRefresh" @click="emit('refresh')">
          {{ busy || match.sync.state === 'syncing' ? 'Refreshing…' : 'Refresh match' }}
        </button>
        <button v-if="allowImport" type="button" :disabled="disabled || busy" @click="emit('import')">Import JSON</button>
      </div>

      <div v-if="match.sync.warnings.length" class="sync-warnings" role="alert">
        <strong>Match data needs review</strong>
        <span v-for="warning in match.sync.warnings" :key="warning">{{ warning }}</span>
      </div>

      <section class="roster-section" aria-labelledby="project-roster-title">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Identity source</p>
            <h3 id="project-roster-title">Roster</h3>
          </div>
          <span>{{ match.roster.length }} players</span>
        </div>
        <div class="team-rosters">
          <article v-for="team in [match.homeTeam, match.awayTeam]" :key="team.id">
            <h4>{{ team.name }}</h4>
            <ul v-if="(team.id === match.homeTeam.id ? homeRoster : awayRoster).length">
              <li
                v-for="player in team.id === match.homeTeam.id ? homeRoster : awayRoster"
                :key="player.id"
              >
                <span>{{ playerLabel(player) }}</span>
                <small>{{ player.goalkeeper ? 'Goalkeeper' : player.position || 'Position unknown' }} · {{ player.role }}</small>
              </li>
            </ul>
            <p v-else>No players saved for this team.</p>
          </article>
        </div>
      </section>

      <section class="events-section" aria-labelledby="project-events-title">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Match clock</p>
            <h3 id="project-events-title">Events</h3>
          </div>
          <span>{{ match.events.length }} events</span>
        </div>
        <ol v-if="orderedEvents.length" class="event-list">
          <li v-for="event in orderedEvents" :key="event.id">
            <time>{{ matchTime(event.minute, event.addedTime) }}</time>
            <div>
              <strong>{{ eventKindLabels[event.kind] }} · {{ event.label }}</strong>
              <span v-if="playerName(event.playerId)">{{ playerName(event.playerId) }}</span>
              <small v-if="event.detail">{{ event.detail }}</small>
            </div>
          </li>
        </ol>
        <p v-else class="section-empty">No match events are saved.</p>
      </section>

      <section v-if="match.substitutions.length" class="substitution-section" aria-labelledby="project-substitutions-title">
        <div class="section-heading">
          <h3 id="project-substitutions-title">Substitutions</h3>
          <span>{{ match.substitutions.length }}</span>
        </div>
        <ul>
          <li v-for="substitution in match.substitutions" :key="substitution.id">
            <time>{{ matchTime(substitution.minute, substitution.addedTime) }}</time>
            <span>{{ substitution.label || `${playerName(substitution.playerOutId) || 'Player out'} → ${playerName(substitution.playerInId) || 'Player in'}` }}</span>
          </li>
        </ul>
      </section>
    </template>
  </section>
</template>

<style scoped>
.project-match-tab {
  display: grid;
  gap: 18px;
  min-width: 0;
}

header,
.section-heading,
.manual-state,
.automatic-sync-state,
.match-identity {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

h2,
h3,
h4,
p {
  margin: 0;
}

h2 {
  font-size: 20px;
}

h3 {
  font-size: 16px;
}

.eyebrow,
.match-meta span,
.section-heading > span,
small,
.manual-state span,
.automatic-sync-state span {
  color: #8fa2b8;
  font-size: 12px;
}

.eyebrow {
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: .12em;
}

.sync-state {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 5px 9px;
  border-radius: 999px;
  background: #173c30;
  color: #79e2aa;
  font-size: 11px;
}

.sync-state span {
  padding-left: 7px;
  border-left: 1px solid currentColor;
}

.sync-manual {
  background: #26364c;
  color: #c6d7eb;
}

.sync-failed,
.sync-state:has(span) {
  background: #492d2a;
  color: #ffb19f;
}

.match-empty,
.manual-state,
.automatic-sync-state,
.sync-warnings,
.roster-section,
.events-section,
.substitution-section {
  padding: 16px;
  border: 1px solid #293b50;
  border-radius: 12px;
  background: #101a26;
}

.match-empty,
.sync-warnings {
  display: grid;
  gap: 8px;
}

.match-empty p,
.section-empty {
  color: #9cacc0;
  font-size: 13px;
}

button {
  min-height: 36px;
  padding: 0 13px;
  border: 1px solid #40566d;
  border-radius: 8px;
  background: #17283a;
  color: #e7f2ff;
  cursor: pointer;
}

button:disabled {
  cursor: not-allowed;
  opacity: .48;
}

.match-meta {
  display: grid;
  gap: 5px;
}

.scoreboard {
  display: grid;
  grid-template-columns: minmax(72px, 1fr) auto minmax(72px, 1fr);
  align-items: center;
  gap: 14px;
  min-width: min(430px, 55vw);
}

.scoreboard > div {
  display: grid;
  justify-items: center;
  gap: 5px;
  text-align: center;
  font-size: 12px;
}

.scoreboard img {
  width: 32px;
  height: 32px;
  object-fit: contain;
}

.scoreboard > strong {
  font-size: 25px;
}

.scoreboard i {
  margin: 0 7px;
  color: #6f8399;
  font-style: normal;
}

.manual-state > div,
.automatic-sync-state > div {
  display: grid;
  flex: 1;
  gap: 4px;
}

.automatic-sync-state {
  justify-content: flex-start;
}

.sync-warnings {
  border-color: #6b4e2d;
  color: #f3c37c;
}

.team-rosters {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 13px;
}

.team-rosters article {
  padding: 12px;
  border-radius: 9px;
  background: #152232;
}

ul,
ol {
  margin: 10px 0 0;
  padding: 0;
  list-style: none;
}

.team-rosters li {
  display: grid;
  gap: 2px;
  padding: 7px 0;
  border-top: 1px solid #26384c;
  font-size: 13px;
}

.event-list li,
.substitution-section li {
  display: grid;
  grid-template-columns: 50px 1fr;
  gap: 10px;
  padding: 9px 0;
  border-top: 1px solid #26384c;
}

time {
  color: #77e6ff;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
}

.event-list li div {
  display: grid;
  gap: 3px;
}

@media (max-width: 760px) {
  .match-identity,
  .manual-state,
  .automatic-sync-state {
    align-items: stretch;
    flex-direction: column;
  }

  .scoreboard {
    min-width: 0;
  }

  .team-rosters {
    grid-template-columns: 1fr;
  }
}
</style>
