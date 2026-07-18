<script setup lang="ts">
import { ref } from 'vue'
import type { CanonicalPerson } from '../../types/identity'
import type { SceneSummary } from '../../types/scene'
import type { Track } from '../../types/tracking'
import type { CanonicalMatch } from '../../types/match'

type MatchTeam = { name: string; color: string }

defineProps<{
  projects: SceneSummary[]
  activeProjectId: string | null
  internalSceneLabel: string | null
  sceneTitle: string
  homeTeam: MatchTeam
  awayTeam: MatchTeam
  matchContextLabel: string
  match: CanonicalMatch | null
  matchRefreshAvailable: boolean
  matchRefreshing: boolean
  rosterImporting: boolean
  mutationLocked: boolean
  rosterImportError: string | null
  trackedObjectCount: number
  trackQuery: string
  tracks: Track[]
  identities: CanonicalPerson[]
  selectedTrackId: string | null
  selectedCanonicalPersonId: string | null
  ballMatchesQuery: boolean
  ballSelected: boolean
  ballTrajectoryMode: 'automatic' | 'manual'
  ballKeyframeCount: number
  trackQuality: (track: Track) => number
}>()

const emit = defineEmits<{
  'navigate-scene': [sceneId: string]
  'refresh-roster': []
  'import-roster': [event: Event]
  'update:trackQuery': [value: string]
  'select-track': [trackId: string]
  'select-identity': [canonicalPersonId: string]
  'select-ball': []
}>()

const rosterFileInput = ref<HTMLInputElement | null>(null)

function chooseRosterFile() {
  if (rosterFileInput.value) rosterFileInput.value.value = ''
  rosterFileInput.value?.click()
}
</script>

<template>
  <aside class="panel left-panel">
    <div class="panel-section scene-switcher">
      <div class="section-heading"><span>Project timeline</span></div>
      <select :value="activeProjectId ?? ''" aria-label="Video project" @change="emit('navigate-scene', ($event.target as HTMLSelectElement).value)">
        <option v-for="item in projects" :key="item.id" :value="item.id">{{ item.title }}</option>
      </select>
      <div v-if="internalSceneLabel && activeProjectId" class="project-context">
        <span>{{ internalSceneLabel }}</span>
        <strong>{{ sceneTitle }}</strong>
        <button @click="emit('navigate-scene', activeProjectId)">← Back to full timeline</button>
      </div>
    </div>

    <div class="panel-section teams-card">
      <p class="section-label">Project match roster</p>
      <div class="score-row">
        <div><i :style="{ background: homeTeam.color }" /><span>{{ homeTeam.name }}</span></div>
        <strong>VS</strong>
        <div><span>{{ awayTeam.name }}</span><i :style="{ background: awayTeam.color }" /></div>
      </div>
      <small>{{ matchContextLabel }}</small>
      <small v-if="match">Canonical project roster available</small>
      <div class="match-snapshot-refresh" role="group" aria-label="Project match roster tools">
        <small>Applies to the full timeline and every shot in this video project.</small>
        <small v-if="match?.sync.stale">Saved project match data is stale.</small>
        <div class="match-snapshot-buttons">
          <button
            v-if="matchRefreshAvailable"
            type="button"
            :disabled="matchRefreshing || rosterImporting || mutationLocked"
            @click="emit('refresh-roster')"
          >{{ matchRefreshing ? 'Refreshing…' : 'Refresh project roster' }}</button>
          <button
            type="button"
            :disabled="rosterImporting || matchRefreshing || mutationLocked"
            @click="chooseRosterFile"
          >{{ rosterImporting ? 'Importing…' : 'Import project roster JSON' }}</button>
        </div>
        <input
          ref="rosterFileInput"
          hidden
          type="file"
          accept=".json,application/json"
          aria-label="Choose roster JSON file"
          @change="emit('import-roster', $event)"
        />
        <small>Example format: <code>data/matches/spain-belgium-2026-qf.json</code>. Choose the file explicitly; it updates the whole video project.</small>
        <small v-if="rosterImportError" class="match-import-error" role="alert">{{ rosterImportError }}</small>
      </div>
    </div>

    <div class="tracks-header">
      <span>Tracked objects</span>
      <span>{{ trackedObjectCount }}</span>
    </div>
    <div v-if="tracks.length || identities.length" class="track-search" role="search" aria-label="Tracked objects">
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" /><path d="m15.5 15.5 5 5" /></svg>
      <input :value="trackQuery" type="search" aria-label="Search tracked objects" placeholder="Find player, number or track…" @input="emit('update:trackQuery', ($event.target as HTMLInputElement).value)" />
      <button v-if="trackQuery" type="button" aria-label="Clear tracked object search" @click="emit('update:trackQuery', '')">×</button>
    </div>
    <div class="track-list">
      <button
        v-for="track in tracks"
        :key="track.id"
        class="track-row"
        :class="{ selected: selectedTrackId === track.id }"
        @click="emit('select-track', track.id)"
      >
        <span class="jersey-dot" :style="{ background: track.color }">{{ track.number }}</span>
        <span class="track-copy">
          <strong>{{ track.label }}</strong>
          <small>{{ track.externalPlayerId ? 'Linked player' : 'Unbound track' }}</small>
        </span>
        <span class="confidence" :class="{ weak: trackQuality(track) < 85 }" title="Average confidence across observed frames">{{ trackQuality(track) }}</span>
      </button>
      <button
        v-for="identity in identities"
        :key="identity.canonicalPersonId"
        class="track-row identity-only-row"
        :class="{ selected: selectedCanonicalPersonId === identity.canonicalPersonId && !selectedTrackId }"
        @click="emit('select-identity', identity.canonicalPersonId)"
      >
        <span class="jersey-dot">{{ identity.jerseyNumber || '?' }}</span>
        <span class="track-copy">
          <strong>{{ identity.displayName }}</strong>
          <small>Canonical identity · not projected in 3D</small>
        </span>
        <span class="confidence" :class="{ weak: (identity.identityConfidence ?? 0) < .85 }">{{ identity.identityConfidence == null ? '—' : Math.round(identity.identityConfidence * 100) }}</span>
      </button>
      <p v-if="trackQuery && !tracks.length && !identities.length && !ballMatchesQuery" class="track-search-empty">No tracked objects match “{{ trackQuery }}”.</p>
      <button v-if="ballMatchesQuery" class="track-row ball-row" :class="{ selected: ballSelected }" type="button" @click="emit('select-ball')">
        <span class="ball-icon">●</span>
        <span class="track-copy">
          <strong>Match ball</strong>
          <small>{{ ballTrajectoryMode === 'manual' ? 'Manual' : 'Automatic' }} · {{ ballKeyframeCount }} keypoints</small>
        </span>
      </button>
    </div>
  </aside>
</template>
