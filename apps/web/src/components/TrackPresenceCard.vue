<script setup lang="ts">
import { computed } from 'vue'
import { trackPresenceAtTime, trackPresenceSummary } from '../lib/trackPresence'
import type { Track } from '../types'

const props = defineProps<{
  track: Track
  currentTime: number
}>()

const snapshot = computed(() => trackPresenceAtTime(props.track, props.currentTime))
const summary = computed(() => trackPresenceSummary(props.track))

function percent(value: number | null) {
  return value === null ? '—' : `${Math.round(value * 100)}%`
}

const uncertainty = computed(() => (
  snapshot.value.uncertaintyMetres === null
    ? 'Not reported'
    : `± ${snapshot.value.uncertaintyMetres.toFixed(2)} m`
))

const sampleMixLabel = computed(() => (
  `Sample evidence mix: observed ${percent(summary.value.observedSampleRatio)}, inferred ${percent(summary.value.inferredSampleRatio)}`
))
</script>

<template>
  <section class="track-presence-card" aria-label="Selected player presence">
    <header>
      <span>Current presence</span>
      <strong :class="snapshot.observed ? 'observed' : 'inferred'">{{ snapshot.label }}</strong>
    </header>
    <p>{{ snapshot.detail }}</p>

    <dl>
      <div>
        <dt>Position uncertainty</dt>
        <dd>{{ uncertainty }}</dd>
      </div>
      <div>
        <dt>Timeline presence</dt>
        <dd>{{ percent(summary.timelineCoverage) }}</dd>
      </div>
      <div v-if="summary.observedSpanRatio !== null">
        <dt>Observed span</dt>
        <dd>{{ percent(summary.observedSpanRatio) }}</dd>
      </div>
    </dl>

    <span class="sample-mix-title">Sample evidence mix</span>
    <div
      v-if="summary.observedSampleRatio !== null && summary.inferredSampleRatio !== null"
      class="presence-coverage"
      role="img"
      :aria-label="sampleMixLabel"
      :title="sampleMixLabel"
    >
      <i class="observed" :style="{ width: percent(summary.observedSampleRatio) }" />
      <i class="inferred" :style="{ width: percent(summary.inferredSampleRatio) }" />
    </div>
    <div class="coverage-legend">
      <span><i class="observed" />Observed {{ percent(summary.observedSampleRatio) }}</span>
      <span><i class="inferred" />Inferred {{ percent(summary.inferredSampleRatio) }}</span>
    </div>
    <small>{{ summary.observationCount }} observed · {{ summary.inferredKeyframeCount }} inferred keyframes</small>
  </section>
</template>

<style scoped>
.track-presence-card {
  margin-top: 14px;
  padding: 13px;
  border: 1px solid rgba(118, 169, 255, .24);
  border-radius: 3px;
  background: rgba(118, 169, 255, .035);
}

header,
dl > div,
.coverage-legend {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.sample-mix-title {
  display: block;
  margin-top: 12px;
  color: #89928d;
  font-size: 10px;
}

header > span,
dt {
  color: #89928d;
  font-size: 12px;
}

header strong,
dd,
.coverage-legend,
small {
  font-family: 'DM Mono', monospace;
}

header strong {
  font-size: 11px;
  letter-spacing: .04em;
}

header strong.observed { color: #71e2aa; }
header strong.inferred { color: #76a9ff; }

p {
  margin: 7px 0 11px;
  color: #77817c;
  font-size: 10px;
  line-height: 1.45;
}

dl {
  display: grid;
  gap: 7px;
  margin: 0;
}

dt { font-size: 10px; }
dd {
  margin: 0;
  color: #cbd5cf;
  font-size: 10px;
}

.presence-coverage {
  display: flex;
  height: 5px;
  margin: 5px 0 8px;
  overflow: hidden;
  border-radius: 4px;
  background: #252b28;
}

.presence-coverage i { display: block; height: 100%; }
i.observed { background: #71e2aa; }
i.inferred { background: #76a9ff; }

.coverage-legend {
  justify-content: flex-start;
  gap: 14px;
  color: #89928d;
  font-size: 9px;
}

.coverage-legend span {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}

.coverage-legend i {
  width: 6px;
  height: 6px;
  border-radius: 50%;
}

small {
  display: block;
  margin-top: 7px;
  color: #59615e;
  font-size: 9px;
}
</style>
