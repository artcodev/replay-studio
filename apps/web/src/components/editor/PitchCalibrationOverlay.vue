<script setup lang="ts">
import type { ComponentPublicInstance } from 'vue'
import type { CalibrationFrameDiagnostics } from '../../lib/calibrationDiagnostics'
import type {
  CalibrationEvidenceMarking,
  CalibrationEvidencePoint,
  CalibrationFrameEvidence,
  PitchCalibrationDraft,
} from '../../types/calibration'

defineProps<{
  draft: PitchCalibrationDraft | null
  diagnostics: CalibrationFrameDiagnostics | null
  qaFrame: CalibrationFrameEvidence | null
  qaFrameSize: { width: number; height: number }
  qaMarkings: CalibrationEvidenceMarking[]
}>()

const emit = defineEmits<{
  overlayElement: [element: SVGSVGElement | null]
  updateDrag: [event: PointerEvent]
  finishDrag: [event: PointerEvent]
  startAnchorDrag: [event: PointerEvent, anchorId: string]
  nudgeAnchor: [event: KeyboardEvent, anchorId: string]
}>()

function bindOverlay(element: Element | ComponentPublicInstance | null) {
  emit('overlayElement', element instanceof SVGSVGElement ? element : null)
}

function projectedEvidencePoint(point: CalibrationEvidencePoint) {
  return point.projected ?? point.projectedImage ?? null
}

function calibrationPointResidual(point: CalibrationEvidencePoint) {
  return point.residualVector?.magnitude ?? point.residualPx ?? null
}

function rawLinePoints(line: CalibrationFrameDiagnostics['lines'][number]) {
  if (line.points && line.points.length >= 2) return line.points
  return line.start && line.end ? [line.start, line.end] : []
}
</script>

<template>
  <svg
    v-if="draft"
    :ref="bindOverlay"
    class="pitch-calibration-overlay"
    :viewBox="`0 0 ${draft.frameWidth} ${draft.frameHeight}`"
    preserveAspectRatio="xMidYMid meet"
    aria-label="Pitch calibration overlay"
    @click.stop
    @pointermove.stop.prevent="emit('updateDrag', $event)"
    @pointerup.stop.prevent="emit('finishDrag', $event)"
    @pointercancel.stop.prevent="emit('finishDrag', $event)"
  >
    <polyline
      v-for="marking in draft.markings"
      :key="marking.id"
      class="calibration-marking"
      :class="marking.kind"
      :points="marking.points.map((point) => `${point.x},${point.y}`).join(' ')"
    />
    <polyline
      v-for="(line, index) in diagnostics?.lines ?? []"
      :key="`source-line-${line.id ?? index}`"
      class="calibration-source-line"
      :class="{ accepted: line.accepted ?? line.inlier, rejected: line.accepted === false || line.inlier === false }"
      :points="rawLinePoints(line).map((point) => `${point.x},${point.y}`).join(' ')"
    />
    <line
      v-if="draft.horizon"
      class="calibration-horizon"
      :x1="draft.horizon.start.x"
      :y1="draft.horizon.start.y"
      :x2="draft.horizon.end.x"
      :y2="draft.horizon.end.y"
    />
    <g
      v-for="(point, index) in diagnostics?.points ?? []"
      :key="`detected-${point.id ?? index}`"
      class="calibration-detected-keypoint"
      :class="{ inlier: point.inlier, outlier: point.inlier === false }"
    >
      <line
        v-if="projectedEvidencePoint(point)"
        :x1="point.image.x"
        :y1="point.image.y"
        :x2="projectedEvidencePoint(point)!.x"
        :y2="projectedEvidencePoint(point)!.y"
      />
      <circle class="source" :cx="point.image.x" :cy="point.image.y" r="6" />
      <circle
        v-if="projectedEvidencePoint(point)"
        class="projected"
        :cx="projectedEvidencePoint(point)!.x"
        :cy="projectedEvidencePoint(point)!.y"
        r="4"
      />
      <text :x="point.image.x + 9" :y="point.image.y - 8">
        {{ point.label || `KP ${index + 1}` }}<template v-if="calibrationPointResidual(point) !== null"> · {{ calibrationPointResidual(point)!.toFixed(1) }}px</template>
      </text>
    </g>
    <g
      v-for="(anchor, index) in draft.anchors"
      :key="anchor.id"
      class="calibration-anchor"
      @pointerdown.stop.prevent="emit('startAnchorDrag', $event, anchor.id)"
    >
      <circle
        :cx="anchor.image.x"
        :cy="anchor.image.y"
        r="13"
        role="button"
        tabindex="0"
        :aria-label="`Calibration anchor ${index + 1}: ${anchor.label}`"
        @keydown.prevent="emit('nudgeAnchor', $event, anchor.id)"
      />
      <text :x="anchor.image.x" :y="anchor.image.y + 4">{{ index + 1 }}</text>
      <text class="anchor-label" :x="anchor.image.x + 18" :y="anchor.image.y - 15">{{ anchor.label }}</text>
    </g>
  </svg>
  <svg
    v-else-if="qaFrame && qaMarkings.length"
    class="calibration-qa-overlay"
    :class="qaFrame.status"
    :viewBox="`0 0 ${qaFrameSize.width} ${qaFrameSize.height}`"
    preserveAspectRatio="xMidYMid meet"
    aria-label="Exact stored calibration evidence overlay"
    @click.stop
  >
    <polyline
      v-for="marking in qaMarkings"
      :key="marking.id"
      class="calibration-qa-marking"
      :class="marking.kind"
      :points="marking.points.map((point) => `${point.x},${point.y}`).join(' ')"
    />
    <g
      v-for="(point, index) in qaFrame.keypoints ?? []"
      :key="point.id ?? index"
      class="calibration-qa-keypoint"
      :class="{ inlier: point.inlier, outlier: point.inlier === false }"
    >
      <circle :cx="point.image.x" :cy="point.image.y" r="6" />
      <line v-if="projectedEvidencePoint(point)" :x1="point.image.x" :y1="point.image.y" :x2="projectedEvidencePoint(point)!.x" :y2="projectedEvidencePoint(point)!.y" />
      <circle v-if="projectedEvidencePoint(point)" class="projected" :cx="projectedEvidencePoint(point)!.x" :cy="projectedEvidencePoint(point)!.y" r="4" />
    </g>
    <g class="calibration-qa-axis">
      <line x1="24" :y1="qaFrameSize.height - 28" x2="96" :y2="qaFrameSize.height - 28" />
      <path :d="`M 96 ${qaFrameSize.height - 28} l -12 -7 l 0 14 z`" />
      <text x="24" :y="qaFrameSize.height - 39">PITCH X · LEFT → RIGHT</text>
    </g>
  </svg>
</template>
