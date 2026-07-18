<script setup lang="ts">
import type { MatchCandidate } from '../types/match'

withDefaults(defineProps<{
  query?: string
  date?: string
  candidates?: MatchCandidate[]
  loading?: boolean
  selectingId?: string | null
  error?: string | null
  disabled?: boolean
}>(), {
  query: '',
  date: '',
  candidates: () => [],
  loading: false,
  selectingId: null,
  error: null,
  disabled: false,
})

const emit = defineEmits<{
  'update:query': [value: string]
  'update:date': [value: string]
  searchQuery: []
  searchDate: []
  select: [candidate: MatchCandidate]
}>()

function score(value: number | null) {
  return value === null ? '–' : String(value)
}
</script>

<template>
  <section class="match-search" aria-labelledby="match-search-title">
    <header>
      <div>
        <p>Match catalog</p>
        <h2 id="match-search-title">Find or change match</h2>
      </div>
      <span>Normalized by the server</span>
    </header>

    <div class="search-controls">
      <form @submit.prevent="emit('searchQuery')">
        <input
          type="search"
          :value="query"
          :disabled="disabled || loading"
          aria-label="Search match by teams"
          placeholder="Spain vs Belgium"
          @input="emit('update:query', ($event.target as HTMLInputElement).value)"
        />
        <button type="submit" :disabled="disabled || loading || query.trim().length < 3">Search</button>
      </form>
      <form @submit.prevent="emit('searchDate')">
        <input
          type="date"
          :value="date"
          :disabled="disabled || loading"
          aria-label="Match date"
          @input="emit('update:date', ($event.target as HTMLInputElement).value)"
        />
        <button type="submit" :disabled="disabled || loading || !date">Fixtures</button>
      </form>
    </div>

    <p v-if="loading" class="state" role="status">Loading matches…</p>
    <p v-else-if="error" class="error" role="alert">{{ error }}</p>
    <div v-else-if="candidates.length" class="candidate-list">
      <button
        v-for="candidate in candidates"
        :key="candidate.id"
        type="button"
        :disabled="disabled || Boolean(selectingId)"
        @click="emit('select', candidate)"
      >
        <span class="candidate-time">{{ candidate.date || 'Date pending' }}<small>{{ candidate.time?.slice(0, 5) || candidate.status || '' }}</small></span>
        <span class="candidate-teams"><strong>{{ candidate.homeTeam.name }}</strong><strong>{{ candidate.awayTeam.name }}</strong><small>{{ candidate.competition || candidate.name }}</small></span>
        <span class="candidate-score"><strong>{{ score(candidate.score.home) }}</strong><strong>{{ score(candidate.score.away) }}</strong></span>
        <i>{{ selectingId === candidate.id ? 'Selecting…' : 'Select' }}</i>
      </button>
    </div>
    <p v-else class="hint">Search by teams or load fixtures for a date. The browser receives one stable match shape regardless of the configured source.</p>
  </section>
</template>

<style scoped>
.match-search {
  display: grid;
  gap: 14px;
  padding: 18px;
  border: 1px solid #32465d;
  border-radius: 12px;
  background: #101c2b;
}

header,
.search-controls,
.candidate-list button {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

h2,
p {
  margin: 0;
}

header p,
header > span,
.hint,
.candidate-time,
.candidate-teams small {
  color: #8fa2b8;
  font-size: 12px;
}

header p {
  text-transform: uppercase;
  letter-spacing: .12em;
}

h2 {
  margin-top: 3px;
  font-size: 18px;
}

.search-controls form {
  display: flex;
  flex: 1;
  gap: 8px;
}

input,
button {
  min-height: 40px;
  border: 1px solid #3b5068;
  border-radius: 8px;
  color: #e9f2fb;
  background: #152437;
}

input {
  min-width: 0;
  flex: 1;
  padding: 0 12px;
}

button {
  padding: 0 13px;
  cursor: pointer;
}

button:disabled {
  cursor: not-allowed;
  opacity: .55;
}

.candidate-list {
  display: grid;
  gap: 8px;
}

.candidate-list button {
  width: 100%;
  padding: 10px 12px;
  text-align: left;
}

.candidate-list button:hover:not(:disabled) {
  border-color: #5d8fbd;
  background: #1b3047;
}

.candidate-time,
.candidate-teams,
.candidate-score {
  display: grid;
  gap: 3px;
}

.candidate-time {
  min-width: 100px;
}

.candidate-time small {
  color: #cad7e4;
}

.candidate-teams {
  min-width: 0;
  flex: 1;
}

.candidate-score strong {
  font-size: 17px;
}

.candidate-list i {
  color: #81b9ec;
  font-size: 12px;
  font-style: normal;
}

.state,
.hint,
.error {
  padding: 12px;
  border-radius: 8px;
  background: #152437;
}

.error {
  color: #ffc2b8;
  background: #3b2024;
}

@media (max-width: 760px) {
  header,
  .search-controls {
    align-items: stretch;
    flex-direction: column;
  }

  .candidate-list button {
    align-items: start;
    flex-wrap: wrap;
  }
}
</style>
