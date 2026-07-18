<script setup lang="ts">
import { computed } from 'vue'
import type { AnalysisJob } from '../types/project'
import {
  isAnalysisJobActive,
  isAnalysisJobCancelable,
  orderAnalysisJobs,
} from '../lib/analysisJobs'

const props = withDefaults(defineProps<{
  jobs?: AnalysisJob[]
  loading?: boolean
  error?: string | null
  cancelingJobIds?: string[]
  lastUpdatedAt?: string | null
}>(), {
  jobs: () => [],
  loading: false,
  error: null,
  cancelingJobIds: () => [],
  lastUpdatedAt: null,
})

const emit = defineEmits<{
  cancel: [runId: string]
  retry: []
}>()

const orderedJobs = computed(() => orderAnalysisJobs(props.jobs))
const activeCount = computed(() => props.jobs.filter(isAnalysisJobActive).length)
const terminalCount = computed(() => props.jobs.length - activeCount.value)

const kindLabels: Record<string, string> = {
  'video-processing': 'Prepare video',
  reconstruction: 'Reconstruct moment',
  'multi-pass': 'Multi-angle analysis',
  'model-comparison': 'Compare detection models',
  'match-sync': 'Sync match data',
}

const statusLabels: Record<AnalysisJob['status'], string> = {
  queued: 'Queued',
  running: 'Running',
  cancelling: 'Cancelling',
  cancelled: 'Cancelled',
  succeeded: 'Completed',
  failed: 'Failed',
}

function kindLabel(kind: string) {
  return kindLabels[kind] ?? kind.replace(/[-_]+/g, ' ')
}

function boundedPercent(job: AnalysisJob) {
  return Math.max(0, Math.min(100, Math.round(job.progress.percent)))
}

function progressCount(job: AnalysisJob) {
  if (job.progress.total <= 0) return null
  return `${Math.max(0, job.progress.completed)} / ${job.progress.total}`
}

function etaLabel(seconds: number | null) {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return null
  if (seconds < 60) return `${Math.ceil(seconds)}s left`
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.ceil(seconds % 60)
  return `${minutes}m ${remainder}s left`
}

function updatedLabel(value: string | null) {
  if (!value) return null
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? null : date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}
</script>

<template>
  <section class="analysis-jobs-panel" aria-labelledby="analysis-jobs-title">
    <header>
      <div>
        <p class="eyebrow">Project compute</p>
        <h2 id="analysis-jobs-title">Analysis jobs</h2>
      </div>
      <div class="job-summary" aria-label="Analysis job summary">
        <strong>{{ activeCount }}</strong><span>active</span>
        <strong>{{ terminalCount }}</strong><span>finished</span>
      </div>
    </header>

    <div v-if="error" class="jobs-error" role="alert">
      <span>{{ error }}</span>
      <button type="button" @click="emit('retry')">Retry</button>
    </div>
    <p v-else-if="loading && !orderedJobs.length" class="jobs-state" role="status">
      Loading analysis jobs…
    </p>
    <div v-else-if="!orderedJobs.length" class="jobs-empty">
      <strong>No analysis jobs yet</strong>
      <span>Start analysis from a moment. Its progress will appear here without downloading the full scene.</span>
    </div>

    <div v-else class="job-list">
      <article
        v-for="job in orderedJobs"
        :key="job.id"
        class="analysis-job"
        :class="`status-${job.status}`"
        :data-status="job.status"
      >
        <div class="job-heading">
          <div>
            <span class="status-pill">{{ statusLabels[job.status] }}</span>
            <h3>{{ kindLabel(job.kind) }}</h3>
            <small v-if="job.segmentId">Moment {{ job.segmentId }}</small>
          </div>
          <button
            v-if="isAnalysisJobCancelable(job)"
            type="button"
            class="cancel-job"
            :disabled="cancelingJobIds.includes(job.id)"
            :aria-label="`Cancel ${kindLabel(job.kind)}`"
            @click="emit('cancel', job.id)"
          >
            {{ cancelingJobIds.includes(job.id) ? 'Cancelling…' : 'Cancel' }}
          </button>
        </div>

        <div class="job-progress-copy">
          <strong>{{ job.progress.label || job.phase || statusLabels[job.status] }}</strong>
          <span>{{ boundedPercent(job) }}%</span>
        </div>
        <div
          class="job-progress"
          role="progressbar"
          :aria-label="`${kindLabel(job.kind)} progress`"
          aria-valuemin="0"
          aria-valuemax="100"
          :aria-valuenow="boundedPercent(job)"
        >
          <i :style="{ width: `${boundedPercent(job)}%` }" />
        </div>
        <p v-if="job.progress.detail">{{ job.progress.detail }}</p>
        <div class="job-meta">
          <span v-if="job.phase">Phase · {{ job.phase }}</span>
          <span v-if="progressCount(job)">{{ progressCount(job) }}</span>
          <span v-if="etaLabel(job.progress.etaSeconds)">{{ etaLabel(job.progress.etaSeconds) }}</span>
        </div>
        <p v-if="job.error" class="job-error" role="alert">{{ job.error }}</p>
      </article>
    </div>

    <footer v-if="updatedLabel(lastUpdatedAt)">
      Compact progress updated {{ updatedLabel(lastUpdatedAt) }}
      <span v-if="loading" role="status"> · refreshing…</span>
    </footer>
  </section>
</template>

<style scoped>
.analysis-jobs-panel {
  display: grid;
  gap: 16px;
  min-width: 0;
}

header,
.job-heading,
.job-progress-copy,
.job-meta,
.jobs-error {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

h2,
h3,
p {
  margin: 0;
}

h2 {
  font-size: 20px;
}

h3 {
  margin-top: 7px;
  font-size: 15px;
}

.eyebrow,
small,
footer,
.job-meta,
.jobs-empty span {
  color: #8fa2b8;
  font-size: 12px;
}

.eyebrow {
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: .12em;
}

.job-summary {
  display: grid;
  grid-template-columns: auto auto;
  gap: 2px 7px;
  color: #8fa2b8;
  font-size: 11px;
}

.job-summary strong {
  color: #eef6ff;
  text-align: right;
}

.job-list {
  display: grid;
  gap: 10px;
}

.analysis-job,
.jobs-empty,
.jobs-error {
  padding: 14px;
  border: 1px solid #26384c;
  border-radius: 12px;
  background: #101a26;
}

.jobs-empty {
  display: grid;
  gap: 6px;
  text-align: center;
}

.jobs-error,
.job-error {
  color: #ffb4a8;
}

.status-pill {
  display: inline-flex;
  padding: 3px 7px;
  border-radius: 999px;
  background: #203246;
  color: #b9c9db;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.status-running .status-pill,
.status-queued .status-pill {
  background: #123d4a;
  color: #77e6ff;
}

.status-succeeded .status-pill {
  background: #173c30;
  color: #79e2aa;
}

.status-failed .status-pill {
  background: #49272a;
  color: #ff9e93;
}

.cancel-job,
.jobs-error button {
  min-height: 34px;
  padding: 0 12px;
  border: 1px solid #40566d;
  border-radius: 8px;
  background: transparent;
  color: #d7e5f4;
  cursor: pointer;
}

.cancel-job:disabled {
  cursor: wait;
  opacity: .55;
}

.job-progress-copy {
  margin-top: 13px;
  font-size: 12px;
}

.job-progress {
  height: 7px;
  margin-top: 7px;
  overflow: hidden;
  border-radius: 999px;
  background: #243244;
}

.job-progress i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #47c7ea, #72e4aa);
}

.analysis-job > p {
  margin-top: 9px;
  color: #aebed0;
  font-size: 12px;
}

.job-meta {
  justify-content: flex-start;
  flex-wrap: wrap;
  margin-top: 9px;
}

.job-meta span + span::before {
  content: '·';
  margin-right: 12px;
}

footer {
  text-align: right;
}
</style>
