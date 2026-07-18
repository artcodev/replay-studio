<script setup lang="ts">
import type { ComponentPublicInstance } from 'vue'
import { annotationIdentityAction } from '../../lib/identityCorrections'
import { frameMetricBadge } from '../../lib/videoTrackSelection'
import type { FrameAnalysis, FrameAnnotation } from '../../types/analysis'

type FramePerson = FrameAnalysis['people'][number]

defineProps<{
  analysis: FrameAnalysis
  selectedPersonId: string | null
  labeling: boolean
  draft: { bbox: { x: number; y: number; width: number; height: number } } | null
  canonicalId: (person: FramePerson) => string | null
  personLabel: (person: FramePerson) => string
  selectionDescription: (person: FramePerson) => string
}>()

const emit = defineEmits<{
  overlayElement: [element: SVGSVGElement | null]
  startDrag: [event: PointerEvent]
  updateDrag: [event: PointerEvent]
  finishDrag: [event: PointerEvent]
  selectAtPoint: [event: MouseEvent]
  selectPerson: [person: FramePerson]
  selectAnnotation: [annotation: FrameAnnotation]
}>()

function bindOverlay(element: Element | ComponentPublicInstance | null) {
  emit('overlayElement', element instanceof SVGSVGElement ? element : null)
}
</script>

<template>
  <svg
    :ref="bindOverlay"
    class="frame-analysis-overlay"
    :class="{ labeling }"
    :viewBox="`0 0 ${analysis.frameWidth} ${analysis.frameHeight}`"
    preserveAspectRatio="xMidYMid meet"
    aria-label="Current frame detections"
    @pointerdown="emit('startDrag', $event)"
    @pointermove="emit('updateDrag', $event)"
    @pointerup="emit('finishDrag', $event)"
    @pointercancel="emit('finishDrag', $event)"
    @click="emit('selectAtPoint', $event)"
  >
    <g
      v-for="person in analysis.people"
      :key="person.id"
      class="frame-person-box"
      :class="{ matched: canonicalId(person) || person.matchedTrackId, selected: person.id === selectedPersonId, manual: person.annotationId, confirmed: person.previewState === 'confirmed', merged: person.previewState === 'merged', split: person.previewState === 'split' }"
      @pointerdown.stop
    >
      <rect
        :x="person.bbox.x"
        :y="person.bbox.y"
        :width="person.bbox.width"
        :height="person.bbox.height"
        :stroke="canonicalId(person) || person.matchedTrackId ? '#71e2aa' : '#ff8f63'"
        role="button"
        tabindex="0"
        :aria-pressed="person.id === selectedPersonId"
        :aria-label="`${personLabel(person)}, ${Math.round(person.confidence * 100)} percent, ${selectionDescription(person)}`"
        @pointerdown.stop
        @keydown.enter.prevent.stop="emit('selectPerson', person)"
        @keydown.space.prevent.stop="emit('selectPerson', person)"
      />
      <rect
        class="jersey-swatch"
        :x="person.bbox.x"
        :y="Math.max(1, person.bbox.y - 16)"
        width="12"
        height="12"
        :fill="person.jerseyColor"
      />
      <text :x="person.bbox.x + 16" :y="Math.max(11, person.bbox.y - 6)">
        {{ personLabel(person) }} · {{ person.previewState === 'merged' ? `MERGED → ${person.mergeTargetId}` : person.previewState === 'split' ? `SPLIT [${person.rangeStart?.toFixed(2)}, ${person.rangeEnd?.toFixed(2)})` : person.previewState === 'confirmed' ? 'CONFIRMED' : `${Math.round(person.confidence * 100)}%` }}
      </text>
      <text
        v-if="person.id === selectedPersonId"
        class="pitch-position"
        :class="{ uncertain: frameMetricBadge(person) === 'UNCERTAIN' }"
        :x="person.bbox.x"
        :y="person.bbox.y + person.bbox.height + 12"
      >
        {{ frameMetricBadge(person) === 'UNCERTAIN' ? '3D position uncertain' : `x ${person.pitch.x.toFixed(1)} · z ${person.pitch.z.toFixed(1)}` }}
      </text>
    </g>
    <g
      v-for="annotation in analysis.annotations.filter((item) => annotationIdentityAction(item) === 'exclude')"
      :key="annotation.id"
      class="frame-ignore-box"
      @pointerdown.stop
      @click.stop="emit('selectAnnotation', annotation)"
    >
      <rect :x="annotation.bbox.x" :y="annotation.bbox.y" :width="annotation.bbox.width" :height="annotation.bbox.height" />
      <path :d="`M ${annotation.bbox.x} ${annotation.bbox.y} L ${annotation.bbox.x + annotation.bbox.width} ${annotation.bbox.y + annotation.bbox.height} M ${annotation.bbox.x + annotation.bbox.width} ${annotation.bbox.y} L ${annotation.bbox.x} ${annotation.bbox.y + annotation.bbox.height}`" />
      <text :x="annotation.bbox.x" :y="Math.max(14, annotation.bbox.y - 7)">EXCLUDED</text>
    </g>
    <g
      v-for="annotation in analysis.annotations.filter((item) => annotationIdentityAction(item) === 'split')"
      :key="annotation.id"
      class="frame-split-box"
      @pointerdown.stop
      @click.stop="emit('selectAnnotation', annotation)"
    >
      <rect :x="annotation.bbox.x" :y="annotation.bbox.y" :width="annotation.bbox.width" :height="annotation.bbox.height" />
      <line
        :x1="annotation.bbox.x + annotation.bbox.width / 2"
        :y1="annotation.bbox.y - 5"
        :x2="annotation.bbox.x + annotation.bbox.width / 2"
        :y2="annotation.bbox.y + annotation.bbox.height + 5"
      />
      <text :x="annotation.bbox.x" :y="Math.max(14, annotation.bbox.y - 7)">
        SPLIT · {{ annotation.rangeStart?.toFixed(2) }}–{{ annotation.rangeEnd?.toFixed(2) }}s
      </text>
    </g>
    <rect
      v-if="labeling && draft"
      class="frame-annotation-draft"
      :x="draft.bbox.x"
      :y="draft.bbox.y"
      :width="draft.bbox.width"
      :height="draft.bbox.height"
    />
    <g
      v-for="ball in analysis.ballCandidates"
      :key="ball.id"
      class="frame-ball-candidate"
      :class="{ primary: ball.primary }"
      @click.stop
    >
      <circle :cx="ball.image.x" :cy="ball.image.y" :r="ball.primary ? 10 : 7" />
      <text v-if="ball.primary" :x="ball.image.x + 12" :y="ball.image.y - 8">BALL {{ Math.round(ball.confidence * 100) }}%</text>
    </g>
  </svg>
</template>
