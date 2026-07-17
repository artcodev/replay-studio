<script setup lang="ts">
import { computed } from 'vue'
import type { Keyframe } from '../types'
import {
  projectPitchEdgeInContext,
  projectPitchPointInContext,
  type PathProjectionContext,
} from '../lib/pathProjection'
import {
  buildPathTrackingSegments,
  interpolatePathTrackingSegments,
  pathTrackingPoints,
  pathTrackingOptionsForSubject,
  type PathEvidence,
  type PathTrackingSubjectKind,
} from '../lib/pathTracking'

const props = withDefaults(defineProps<{
  enabled: boolean
  keyframes: Keyframe[]
  projectionContext: PathProjectionContext | null
  currentTime: number
  subjectKind?: PathTrackingSubjectKind
  color?: string
  subjectLabel?: string | null
}>(), {
  color: '#ffd36a',
  subjectLabel: null,
  subjectKind: 'player',
})

type RenderedEdge = {
  id: string
  evidence: PathEvidence
  x1: number
  y1: number
  x2: number
  y2: number
}

const projection = computed(() => (
  props.enabled ? props.projectionContext : null
))
const normalizedKeyframes = computed(() => (
  pathTrackingPoints(props.keyframes).map((point) => point.keyframe)
))
const pathSegments = computed(() => buildPathTrackingSegments(
  normalizedKeyframes.value,
  pathTrackingOptionsForSubject(props.subjectKind),
))

const renderedEdges = computed<RenderedEdge[]>(() => {
  const context = projection.value
  if (!context) return []
  return pathSegments.value.flatMap((segment, segmentIndex) => (
    segment.points.slice(1).flatMap((point, pointIndex): RenderedEdge[] => {
      const previous = segment.points[pointIndex]
      const edge = projectPitchEdgeInContext(context, previous, point)
      if (!edge) return []
      return [{
        id: `${segmentIndex}-${pointIndex}-${previous.t}-${point.t}`,
        evidence: segment.evidence,
        x1: edge.start.x,
        y1: edge.start.y,
        x2: edge.end.x,
        y2: edge.end.y,
      }]
    })
  ))
})

const currentMarker = computed(() => {
  const context = projection.value
  if (!context) return null
  const current = interpolatePathTrackingSegments(pathSegments.value, props.currentTime)
  if (!current) return null
  const point = projectPitchPointInContext(context, current)
  if (
    !point
    || point.x < 0
    || point.x > context.width
    || point.y < 0
    || point.y > context.height
  ) return null
  return point
})

const overlayStyle = computed(() => ({ '--path-color': props.color }))
const overlayLabel = computed(() => (
  `${props.subjectLabel || 'Selected object'} path on the current calibrated video frame`
))
</script>

<template>
  <svg
    v-if="projection && renderedEdges.length"
    class="video-path-tracking-overlay"
    :style="overlayStyle"
    :viewBox="`0 0 ${projection.width} ${projection.height}`"
    preserveAspectRatio="xMidYMid meet"
    role="img"
    :aria-label="overlayLabel"
    :data-projection-mode="projection.mode"
  >
    <title>{{ overlayLabel }}</title>
    <g class="path-underlay" aria-hidden="true">
      <line
        v-for="edge in renderedEdges"
        :key="`underlay-${edge.id}`"
        :class="edge.evidence"
        :x1="edge.x1"
        :y1="edge.y1"
        :x2="edge.x2"
        :y2="edge.y2"
      />
    </g>
    <g aria-hidden="true">
      <line
        v-for="edge in renderedEdges"
        :key="edge.id"
        class="path-edge"
        :class="edge.evidence"
        :x1="edge.x1"
        :y1="edge.y1"
        :x2="edge.x2"
        :y2="edge.y2"
      />
    </g>
    <g v-if="currentMarker" class="path-current-marker" aria-hidden="true">
      <circle class="underlay" :cx="currentMarker.x" :cy="currentMarker.y" r="8" />
      <circle class="marker" :cx="currentMarker.x" :cy="currentMarker.y" r="5" />
    </g>
  </svg>
</template>

<style scoped>
.video-path-tracking-overlay {
  --path-color: #ffd36a;
  position: absolute;
  z-index: 2;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  pointer-events: none;
}

.video-path-tracking-overlay line {
  fill: none;
  stroke-linecap: round;
  stroke-linejoin: round;
  vector-effect: non-scaling-stroke;
}

.path-underlay line {
  stroke: rgba(3, 6, 5, .86);
  stroke-width: 7;
}

.path-underlay line.inferred {
  stroke-dasharray: 7 6;
}

.path-edge {
  stroke: var(--path-color);
  stroke-width: 3;
  filter: drop-shadow(0 0 3px color-mix(in srgb, var(--path-color) 72%, transparent));
}

.path-edge.inferred {
  opacity: .58;
  stroke-dasharray: 7 6;
  filter: none;
}

.path-current-marker .underlay {
  fill: rgba(3, 6, 5, .88);
  stroke: rgba(3, 6, 5, .88);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}

.path-current-marker .marker {
  fill: color-mix(in srgb, var(--path-color) 26%, transparent);
  stroke: color-mix(in srgb, var(--path-color) 88%, white);
  stroke-width: 3;
  vector-effect: non-scaling-stroke;
  filter: drop-shadow(0 0 4px var(--path-color));
}
</style>
