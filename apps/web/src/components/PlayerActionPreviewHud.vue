<script setup lang="ts">
import { computed } from 'vue'
import {
  playerActionColor,
  playerActionLabel,
  type PlayerActionPlaybackState,
} from '../lib/playerActions'

const props = defineProps<{
  action: PlayerActionPlaybackState
}>()

const keypointLabels = {
  'wind-up': 'Wind-up',
  contact: 'Contact',
  release: 'Release',
  apex: 'Apex',
  impact: 'Impact',
  recovery: 'Recovery',
} as const

const preview = computed(() => {
  const state = props.action
  const phase = Math.max(0, Math.min(1, state.phase))
  const keypoint = state.nearestKeypoint
  let keypointTiming: string | null = null
  if (keypoint) {
    const distance = Math.abs(keypoint.offsetSeconds)
    keypointTiming = distance < 0.005
      ? 'now'
      : keypoint.offsetSeconds > 0
        ? `${distance.toFixed(2)}s ago`
        : `in ${distance.toFixed(2)}s`
  }
  return {
    label: playerActionLabel(state.action.type),
    type: state.action.type,
    color: playerActionColor(state.action.type),
    phasePercent: Math.round(phase * 100),
    keypointLabel: keypoint ? keypointLabels[keypoint.kind] : null,
    keypointTiming,
  }
})
</script>

<template>
  <aside
    class="action-preview-hud"
    :style="{ '--action-color': preview.color }"
    role="status"
    aria-live="polite"
    aria-label="Active player action preview"
  >
    <header>
      <span>Active action</span>
      <code>{{ preview.type }}</code>
    </header>
    <strong>{{ preview.label }}</strong>
    <div class="action-phase-heading">
      <span>Normalized phase</span>
      <b>{{ preview.phasePercent }}%</b>
    </div>
    <div
      class="action-phase-track"
      role="progressbar"
      aria-label="Action phase"
      aria-valuemin="0"
      aria-valuemax="100"
      :aria-valuenow="preview.phasePercent"
    >
      <i :style="{ width: `${preview.phasePercent}%` }" />
    </div>
    <div v-if="preview.keypointLabel" class="action-keypoint">
      <span>Nearest keypoint</span>
      <b>{{ preview.keypointLabel }}<small v-if="preview.keypointTiming"> · {{ preview.keypointTiming }}</small></b>
    </div>
  </aside>
</template>

<style scoped>
.action-preview-hud {
  --action-color: #ffd36a;
  position: absolute;
  z-index: 6;
  right: 12px;
  bottom: 12px;
  width: min(228px, calc(100% - 24px));
  padding: 10px;
  border: 1px solid color-mix(in srgb, var(--action-color) 48%, transparent);
  border-radius: 3px;
  background: rgba(7, 10, 10, .88);
  box-shadow: 0 10px 28px rgba(0, 0, 0, .34);
  backdrop-filter: blur(11px);
  pointer-events: none;
}

.action-preview-hud header,
.action-phase-heading,
.action-keypoint {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.action-preview-hud header > span,
.action-phase-heading > span,
.action-keypoint > span {
  color: #78837d;
  font: 500 8px/1.2 'DM Mono', monospace;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.action-preview-hud header code {
  color: color-mix(in srgb, var(--action-color) 84%, white);
  font: 600 8px/1.2 'DM Mono', monospace;
}

.action-preview-hud > strong {
  display: block;
  margin-top: 4px;
  overflow: hidden;
  color: #f0f4f0;
  font: 650 13px/1.3 Inter, sans-serif;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.action-phase-heading { margin-top: 9px; }
.action-phase-heading b { color: var(--action-color); font: 650 9px/1.2 'DM Mono', monospace; }
.action-phase-track { height: 3px; margin-top: 5px; overflow: hidden; background: rgba(255, 255, 255, .1); }
.action-phase-track i { display: block; height: 100%; background: var(--action-color); box-shadow: 0 0 9px color-mix(in srgb, var(--action-color) 65%, transparent); }
.action-keypoint { margin-top: 9px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, .08); }
.action-keypoint b { color: #e3e9e4; font: 600 8px/1.2 'DM Mono', monospace; white-space: nowrap; }
.action-keypoint small { color: #8c9791; font: inherit; font-weight: 500; }

@media (max-width: 760px) {
  .action-preview-hud { right: 8px; bottom: 8px; padding: 8px; }
}
</style>
