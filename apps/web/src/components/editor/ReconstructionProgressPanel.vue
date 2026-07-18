<script setup lang="ts">
import type { ReconstructionPhase, ReconstructionProgress } from '../../types/reconstruction'

defineProps<{
  progress: ReconstructionProgress | null
  phases: ReconstructionPhase[]
  frameCount: number
  activeRunId: string | null
  canCancel: boolean
  cancelling: boolean
}>()

const emit = defineEmits<{ cancel: [] }>()

function durationLabel(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return 'estimating…'
  const rounded = Math.max(0, Math.ceil(seconds))
  if (rounded < 60) return `${rounded}s`
  const minutes = Math.floor(rounded / 60)
  const remainder = rounded % 60
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`
}

function phaseStatusLabel(status: ReconstructionPhase['status']) {
  if (status === 'completed') return 'COMPLETED'
  if (status === 'current') return 'CURRENT'
  return 'PENDING'
}
</script>

<template>
  <section class="analysis-progress-panel" aria-live="polite" aria-label="Video analysis progress">
    <div class="analysis-progress-heading">
      <div>
        <span>ANALYSIS PIPELINE · PHASE {{ progress?.phaseIndex ?? 1 }} OF {{ progress?.phaseCount ?? phases.length }}</span>
        <strong>{{ progress?.label ?? 'Waiting to start' }}</strong>
        <small>{{ progress?.detail ?? `Queued ${frameCount} sampled frames for analysis.` }}</small>
      </div>
      <div class="analysis-progress-actions">
        <b>{{ progress?.overallPercent ?? 0 }}%</b>
        <button v-if="activeRunId" type="button" :disabled="!canCancel || cancelling" @click="emit('cancel')">
          {{ cancelling ? 'Cancelling…' : 'Cancel' }}
        </button>
      </div>
    </div>
    <div class="analysis-overall-track" aria-hidden="true"><i :style="{ width: `${progress?.overallPercent ?? 0}%` }" /></div>
    <div class="analysis-progress-meta">
      <span v-if="progress?.total">{{ progress.completed }} / {{ progress.total }} in current phase</span>
      <span>{{ durationLabel(progress?.elapsedSeconds ?? 0) }} elapsed</span>
      <span>{{ progress?.etaSeconds === null || progress?.etaSeconds === undefined ? 'Estimating remaining time…' : `≈ ${durationLabel(progress.etaSeconds)} remaining` }}</span>
    </div>
    <ol class="analysis-phase-list">
      <li v-for="(phase, index) in phases" :key="phase.id" :class="phase.status">
        <i aria-hidden="true">{{ phase.status === 'completed' ? '✓' : index + 1 }}</i>
        <span><strong>{{ phase.label }}</strong><small>{{ phaseStatusLabel(phase.status) }}</small></span>
      </li>
    </ol>
  </section>
</template>
