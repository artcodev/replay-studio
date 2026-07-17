<script setup lang="ts">
import { computed } from 'vue'

export type PathTrackingSubjectKind = 'player' | 'ball'

const props = withDefaults(defineProps<{
  enabled: boolean
  subjectKind?: PathTrackingSubjectKind | null
  subjectLabel?: string | null
  subjectColor?: string | null
  sampleCount?: number | null
  hasDrawablePath?: boolean | null
  unavailableLabel?: string | null
  surfaceUnavailableReason?: string | null
  surfaceNote?: string | null
  align?: 'left' | 'right'
  topOffset?: 'default' | 'stacked'
  surfaceLabel?: string
}>(), {
  subjectKind: null,
  subjectLabel: null,
  subjectColor: null,
  sampleCount: null,
  hasDrawablePath: null,
  unavailableLabel: null,
  surfaceUnavailableReason: null,
  surfaceNote: null,
  align: 'right',
  topOffset: 'default',
  surfaceLabel: 'scene',
})

const hasSubject = computed(() => Boolean(props.subjectKind && props.subjectLabel))
const hasPath = computed(() => (
  hasSubject.value
  && !props.surfaceUnavailableReason
  && props.hasDrawablePath !== false
  && (props.sampleCount === null || props.sampleCount >= 2)
))
const subjectSummary = computed(() => {
  if (!hasSubject.value) return null
  if (props.surfaceUnavailableReason) return props.surfaceUnavailableReason
  const kind = props.subjectKind === 'ball' ? 'Ball' : 'Player'
  if (!hasPath.value) {
    const samples = props.sampleCount ?? 0
    return `${kind} · no path · ${samples} ${samples === 1 ? 'sample' : 'samples'}`
  }
  const samples = props.sampleCount === null
    ? ''
    : ` · ${props.sampleCount} ${props.sampleCount === 1 ? 'sample' : 'samples'}`
  return `${kind} · full highlight${samples}`
})

const emptyMessage = computed(() => (
  props.unavailableLabel
    ? `${props.unavailableLabel} has no reconstructed path`
    : 'Select a tracked player or ball to show its path'
))

const colorStyle = computed(() => (
  props.subjectColor ? { '--path-subject-color': props.subjectColor } : undefined
))
</script>

<template>
  <aside
    v-if="enabled"
    class="path-tracking-legend"
    :class="[`align-${align}`, `offset-${topOffset}`, { empty: !hasPath, unavailable: Boolean(surfaceUnavailableReason) }]"
    :style="colorStyle"
    role="status"
    aria-live="polite"
    :aria-label="`Path tracking on ${surfaceLabel}`"
  >
    <template v-if="hasSubject">
      <div class="path-legend-heading">
        <i aria-hidden="true" />
        <span>
          <strong>{{ subjectLabel }}</strong>
          <small>{{ subjectSummary }}</small>
          <small v-if="hasPath && surfaceNote" class="surface-note">{{ surfaceNote }}</small>
        </span>
      </div>
      <div v-if="hasPath" class="path-legend-lines" aria-label="Path segment legend">
        <span><i class="observed" aria-hidden="true" />Observed</span>
        <span><i class="inferred" aria-hidden="true" />Inferred</span>
      </div>
    </template>
    <template v-else>
      <div class="path-legend-heading">
        <i aria-hidden="true" />
        <span>
          <strong>Path tracking</strong>
          <small>{{ emptyMessage }}</small>
        </span>
      </div>
    </template>
  </aside>
</template>

<style scoped>
.path-tracking-legend {
  --path-subject-color: #ffd36a;
  position: absolute;
  z-index: 7;
  top: 44px;
  width: min(320px, calc(100% - 24px));
  padding: 10px 11px;
  border: 1px solid color-mix(in srgb, var(--path-subject-color) 46%, transparent);
  border-radius: 3px;
  background: rgba(7, 10, 10, .88);
  box-shadow: 0 8px 24px rgba(0, 0, 0, .3);
  backdrop-filter: blur(10px);
  pointer-events: none;
}

.align-left { left: 12px; }
.align-right { right: 12px; }
.offset-stacked { top: 102px; }

.path-legend-heading {
  display: flex;
  align-items: center;
  gap: 8px;
}

.path-legend-heading > i {
  flex: 0 0 8px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--path-subject-color);
  box-shadow: 0 0 10px color-mix(in srgb, var(--path-subject-color) 72%, transparent);
}

.path-legend-heading > span {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.path-legend-heading strong {
  overflow: hidden;
  color: #edf2ed;
  font: 600 11px/1.25 'DM Mono', monospace;
  letter-spacing: .06em;
  text-overflow: ellipsis;
  text-transform: uppercase;
  white-space: nowrap;
}

.path-legend-heading small {
  overflow: hidden;
  color: #8c9791;
  font: 500 10px/1.4 'DM Mono', monospace;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.path-tracking-legend.unavailable .path-legend-heading small {
  white-space: normal;
}

.path-legend-heading .surface-note {
  color: #b9c3bd;
  white-space: normal;
}

.path-legend-lines {
  display: flex;
  align-items: center;
  gap: 13px;
  margin-top: 8px;
  padding-top: 7px;
  border-top: 1px solid rgba(255, 255, 255, .08);
}

.path-legend-lines span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #8b958f;
  font: 500 9px/1 'DM Mono', monospace;
  letter-spacing: .04em;
  text-transform: uppercase;
}

.path-legend-lines i {
  display: block;
  width: 24px;
  height: 0;
  border-top: 2px solid var(--path-subject-color);
}

.path-legend-lines i.inferred { border-top-style: dashed; opacity: .68; }

.path-tracking-legend.empty {
  border-color: rgba(255, 211, 106, .28);
  background: rgba(20, 18, 10, .9);
}

.path-tracking-legend.empty .path-legend-heading > i {
  background: transparent;
  border: 1px dashed #ffd36a;
  box-shadow: none;
}

.path-tracking-legend.empty .path-legend-heading strong { color: #ffe19a; }
.path-tracking-legend.empty .path-legend-heading small { color: #b5a579; }

@media (prefers-reduced-motion: reduce) {
  .path-tracking-legend { backdrop-filter: none; }
}
</style>
