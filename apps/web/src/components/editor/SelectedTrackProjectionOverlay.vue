<script setup lang="ts">
import { computed } from 'vue'
import {
  buildTrackProjectionDebugSamples,
  nearestProjectionDebugSample,
} from '../../lib/trackProjectionDebug'
import type { CalibrationFrameEvidence } from '../../types/calibration'
import type { ContactPointProfile } from '../../types/reconstruction'
import type { TrackObservation } from '../../types/tracking'

const props = defineProps<{
  enabled: boolean
  label: string | null
  observations: TrackObservation[] | null | undefined
  calibrationFrames: CalibrationFrameEvidence[]
  pitch: { length: number; width: number }
  currentTime: number
  frameSize: { width: number; height: number }
  contactPointProfile: ContactPointProfile
}>()

const samples = computed(() => buildTrackProjectionDebugSamples(
  props.observations,
  props.calibrationFrames,
  props.pitch,
  props.contactPointProfile,
))
const selection = computed(() => nearestProjectionDebugSample(samples.value, props.currentTime))
const sample = computed(() => (
  props.enabled && selection.value?.active ? selection.value.sample : null
))
const metricLabel = computed(() => {
  const value = sample.value
  if (!value) return ''
  const pitch = value.effectivePitch
    ? `x ${value.effectivePitch.x.toFixed(2)} · z ${value.effectivePitch.z.toFixed(2)}`
    : 'metric unavailable'
  const speed = value.speedMetresPerSecond == null
    ? ''
    : ` · ${value.speedMetresPerSecond.toFixed(1)} m/s`
  return `${pitch}${speed}`
})
</script>

<template>
  <svg
    v-if="sample"
    class="selected-projection-debug-overlay"
    :class="sample.severity"
    :viewBox="`0 0 ${frameSize.width} ${frameSize.height}`"
    preserveAspectRatio="xMidYMid meet"
    aria-label="Selected object projection debug overlay"
  >
    <rect
      class="debug-bbox"
      :x="sample.observation.bbox.x"
      :y="sample.observation.bbox.y"
      :width="sample.observation.bbox.width"
      :height="sample.observation.bbox.height"
    />
    <g class="debug-contact">
      <circle :cx="sample.contactPoint.x" :cy="sample.contactPoint.y" r="9" />
      <line :x1="sample.contactPoint.x - 15" :y1="sample.contactPoint.y" :x2="sample.contactPoint.x + 15" :y2="sample.contactPoint.y" />
      <line :x1="sample.contactPoint.x" :y1="sample.contactPoint.y - 15" :x2="sample.contactPoint.x" :y2="sample.contactPoint.y + 15" />
    </g>
    <g class="debug-label" :transform="`translate(${Math.max(8, Math.min(frameSize.width - 330, sample.observation.bbox.x))} ${Math.max(24, sample.observation.bbox.y - 42)})`">
      <rect x="0" y="0" width="322" height="36" rx="3" />
      <text x="8" y="14">{{ label || 'Selected object' }} · #{{ sample.observation.frameIndex }} · {{ sample.observation.metricStatus || 'unknown' }}</text>
      <text x="8" y="29">{{ metricLabel }} · {{ sample.calibration?.projectionSource || 'no calibration' }}</text>
    </g>
  </svg>
</template>

<style scoped>
.selected-projection-debug-overlay { position: absolute; inset: 0; width: 100%; height: 100%; z-index: 9; pointer-events: none; color: #71e2aa; }
.selected-projection-debug-overlay.speed-violation { color: #ffd36a; }
.selected-projection-debug-overlay.identity-split, .selected-projection-debug-overlay.unprojected { color: #ff7867; }
.debug-bbox { fill: none; stroke: currentColor; stroke-width: 3; stroke-dasharray: 10 5; vector-effect: non-scaling-stroke; }
.debug-contact circle { fill: rgba(8, 12, 11, .75); stroke: currentColor; stroke-width: 3; vector-effect: non-scaling-stroke; }
.debug-contact line { stroke: currentColor; stroke-width: 2; vector-effect: non-scaling-stroke; }
.debug-label rect { fill: rgba(8, 12, 11, .92); stroke: currentColor; stroke-width: 1; vector-effect: non-scaling-stroke; }
.debug-label text { fill: #f4f7f3; font: 600 11px 'DM Mono', monospace; }
.debug-label text + text { fill: currentColor; font-size: 10px; }
</style>
