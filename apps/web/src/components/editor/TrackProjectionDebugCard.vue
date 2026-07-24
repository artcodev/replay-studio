<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  buildTrackProjectionDebugSamples,
  IDENTITY_SPLIT_METRES_PER_SECOND,
  nearestProjectionDebugSample,
  PLAYER_SPEED_QA_METRES_PER_SECOND,
  projectionPopulationEdge,
} from '../../lib/trackProjectionDebug'
import type { CalibrationFrameEvidence } from '../../types/calibration'
import type { ContactPointProfile } from '../../types/reconstruction'
import type { Track, TrackObservation } from '../../types/tracking'

const props = defineProps<{
  label: string
  observations: TrackObservation[] | null | undefined
  tracks: Track[]
  calibrationFrames: CalibrationFrameEvidence[]
  pitch: { length: number; width: number }
  currentTime: number
  contactPointProfile: ContactPointProfile
  calibrationBusy?: boolean
}>()

const emit = defineEmits<{
  seek: [time: number]
  recalibrate: [time: number]
}>()
const samples = computed(() => buildTrackProjectionDebugSamples(
  props.observations,
  props.calibrationFrames,
  props.pitch,
  props.contactPointProfile,
))
const selection = computed(() => nearestProjectionDebugSample(samples.value, props.currentTime))
const sample = computed(() => selection.value?.sample ?? null)
const previousSample = computed(() => {
  const index = selection.value?.index
  return index != null && index > 0 ? samples.value[index - 1] : null
})
const nextSample = computed(() => {
  const index = selection.value?.index
  return index != null && index < samples.value.length - 1 ? samples.value[index + 1] : null
})
const population = computed(() => projectionPopulationEdge(
  props.tracks,
  props.calibrationFrames,
  props.pitch,
  props.contactPointProfile,
  sample.value?.previous?.observation.frameIndex,
  sample.value?.observation.frameIndex ?? -1,
))
const synchronizedJump = computed(() => (
  population.value.comparedTrackCount >= 4
  && population.value.speedViolationCount / population.value.comparedTrackCount >= 0.25
))
const calibrationDiscontinuity = computed(() => (
  (sample.value?.calibrationContinuityDeltaMetres ?? 0) > 0.5
))

const pitchWidth = 304
const pitchHeight = 196
const pitchPadding = 10
const pitchStripeCount = 8
const minimapZoom = ref(1)
const minimapCenter = ref({ x: pitchWidth / 2, y: pitchHeight / 2 })
const minimapPanning = ref(false)
let minimapPointerId: number | null = null
let minimapPointerPosition: { x: number; y: number } | null = null

const minimapViewport = computed(() => {
  const width = pitchWidth / minimapZoom.value
  const height = pitchHeight / minimapZoom.value
  const x = Math.max(0, Math.min(pitchWidth - width, minimapCenter.value.x - width / 2))
  const y = Math.max(0, Math.min(pitchHeight - height, minimapCenter.value.y - height / 2))
  return { x, y, width, height }
})
const minimapViewBox = computed(() => {
  const viewport = minimapViewport.value
  return `${viewport.x} ${viewport.y} ${viewport.width} ${viewport.height}`
})

function clampMinimapCenter(value: { x: number; y: number }, zoom = minimapZoom.value) {
  const width = pitchWidth / zoom
  const height = pitchHeight / zoom
  return {
    x: Math.max(width / 2, Math.min(pitchWidth - width / 2, value.x)),
    y: Math.max(height / 2, Math.min(pitchHeight - height / 2, value.y)),
  }
}

function setMinimapZoom(
  nextZoom: number,
  anchor?: { worldX: number; worldY: number; normalizedX: number; normalizedY: number },
) {
  const zoom = Math.max(1, Math.min(6, nextZoom))
  if (Math.abs(zoom - minimapZoom.value) < 1e-6) return
  const width = pitchWidth / zoom
  const height = pitchHeight / zoom
  const nextCenter = anchor
    ? {
        x: anchor.worldX - (anchor.normalizedX - 0.5) * width,
        y: anchor.worldY - (anchor.normalizedY - 0.5) * height,
      }
    : minimapCenter.value
  minimapZoom.value = zoom
  minimapCenter.value = clampMinimapCenter(nextCenter, zoom)
}

function resetMinimapViewport() {
  minimapZoom.value = 1
  minimapCenter.value = { x: pitchWidth / 2, y: pitchHeight / 2 }
}

function zoomMinimap(direction: 1 | -1) {
  if (direction > 0 && minimapZoom.value === 1 && sample.value?.effectivePitch) {
    minimapCenter.value = {
      x: pitchX(sample.value.effectivePitch.x),
      y: pitchY(sample.value.effectivePitch.z),
    }
  }
  setMinimapZoom(minimapZoom.value * (direction > 0 ? 1.4 : 1 / 1.4))
}

function onMinimapWheel(event: WheelEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const bounds = svg.getBoundingClientRect()
  if (!bounds.width || !bounds.height) return
  const normalizedX = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width))
  const normalizedY = Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height))
  const viewport = minimapViewport.value
  setMinimapZoom(minimapZoom.value * (event.deltaY < 0 ? 1.25 : 0.8), {
    worldX: viewport.x + normalizedX * viewport.width,
    worldY: viewport.y + normalizedY * viewport.height,
    normalizedX,
    normalizedY,
  })
}

function startMinimapPan(event: PointerEvent) {
  if (minimapZoom.value <= 1) return
  minimapPointerId = event.pointerId
  minimapPointerPosition = { x: event.clientX, y: event.clientY }
  minimapPanning.value = true
  ;(event.currentTarget as SVGSVGElement).setPointerCapture(event.pointerId)
}

function updateMinimapPan(event: PointerEvent) {
  if (minimapPointerId !== event.pointerId || !minimapPointerPosition) return
  const svg = event.currentTarget as SVGSVGElement
  const bounds = svg.getBoundingClientRect()
  if (!bounds.width || !bounds.height) return
  const viewport = minimapViewport.value
  minimapCenter.value = clampMinimapCenter({
    x: minimapCenter.value.x - (event.clientX - minimapPointerPosition.x) * viewport.width / bounds.width,
    y: minimapCenter.value.y - (event.clientY - minimapPointerPosition.y) * viewport.height / bounds.height,
  })
  minimapPointerPosition = { x: event.clientX, y: event.clientY }
}

function finishMinimapPan(event: PointerEvent) {
  if (minimapPointerId !== event.pointerId) return
  const svg = event.currentTarget as SVGSVGElement
  if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId)
  minimapPointerId = null
  minimapPointerPosition = null
  minimapPanning.value = false
}

function panMinimap(horizontal: number, vertical: number) {
  const viewport = minimapViewport.value
  minimapCenter.value = clampMinimapCenter({
    x: minimapCenter.value.x + horizontal * viewport.width * 0.12,
    y: minimapCenter.value.y + vertical * viewport.height * 0.12,
  })
}

function onMinimapKeydown(event: KeyboardEvent) {
  if (event.key === '+' || event.key === '=') zoomMinimap(1)
  else if (event.key === '-' || event.key === '_') zoomMinimap(-1)
  else if (event.key === '0') resetMinimapViewport()
  else if (event.key === 'ArrowLeft') panMinimap(-1, 0)
  else if (event.key === 'ArrowRight') panMinimap(1, 0)
  else if (event.key === 'ArrowUp') panMinimap(0, -1)
  else if (event.key === 'ArrowDown') panMinimap(0, 1)
  else return
  event.preventDefault()
}

watch(() => props.label, resetMinimapViewport)

function pitchX(value: number) {
  return pitchPadding + (value + props.pitch.length / 2) / props.pitch.length * (pitchWidth - pitchPadding * 2)
}
function pitchY(value: number) {
  return pitchPadding + (value + props.pitch.width / 2) / props.pitch.width * (pitchHeight - pitchPadding * 2)
}
function number(value: number | null | undefined, digits = 2) {
  return value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits)
}
function seek(sampleTime: number | null | undefined) {
  if (sampleTime != null) emit('seek', sampleTime)
}
const verdict = computed(() => {
  const value = sample.value
  if (!value) return 'NO OBSERVATIONS'
  if (value.severity === 'identity-split') return `IDENTITY SPLIT · >${IDENTITY_SPLIT_METRES_PER_SECOND} M/S`
  if (value.severity === 'speed-violation') return `SPEED QA · >${PLAYER_SPEED_QA_METRES_PER_SECOND} M/S`
  if (value.severity === 'unprojected') return 'METRIC UNAVAILABLE'
  return 'PROJECTION CONTINUOUS'
})
</script>

<template>
  <section class="projection-debug-card" aria-label="Selected object projection debugger">
    <header>
      <div><span>PROJECTION DEBUGGER</span><strong>{{ label }}</strong></div>
      <i v-if="sample" :class="sample.severity">{{ verdict }}</i>
    </header>

    <div v-if="sample" class="projection-frame-controls">
      <button :disabled="!previousSample" @click="seek(previousSample?.observation.sceneTime)">← #{{ previousSample?.observation.frameIndex ?? '—' }}</button>
      <button class="current" :class="{ distant: !selection?.active }" @click="seek(sample.observation.sceneTime)">
        #{{ sample.observation.frameIndex }} · {{ sample.observation.sceneTime.toFixed(3) }}s
        <small v-if="!selection?.active">nearest · click to jump</small>
      </button>
      <button :disabled="!nextSample" @click="seek(nextSample?.observation.sceneTime)">#{{ nextSample?.observation.frameIndex ?? '—' }} →</button>
    </div>
    <button
      v-if="sample"
      type="button"
      class="projection-recalibrate-frame"
      :disabled="calibrationBusy"
      @click="emit('recalibrate', sample.observation.sceneTime)"
    >{{ calibrationBusy ? 'Calibration busy…' : `Recalibrate exact frame #${sample.observation.frameIndex}` }}</button>

    <template v-if="sample">
      <div class="projection-minimap-shell" :class="{ panning: minimapPanning, zoomed: minimapZoom > 1 }">
        <svg
          class="projection-mini-pitch"
          :viewBox="minimapViewBox"
          role="region"
          tabindex="0"
          aria-label="Projection minimap. Use the mouse wheel to zoom, drag to pan, arrow keys to move, and zero to reset."
          @wheel.prevent="onMinimapWheel"
          @pointerdown.prevent="startMinimapPan"
          @pointermove.prevent="updateMinimapPan"
          @pointerup.prevent="finishMinimapPan"
          @pointercancel.prevent="finishMinimapPan"
          @keydown="onMinimapKeydown"
        >
        <g class="pitch-stripes" aria-hidden="true">
          <rect
            v-for="stripe in pitchStripeCount"
            :key="stripe"
            :x="pitchPadding + (stripe - 1) * (pitchWidth - pitchPadding * 2) / pitchStripeCount"
            :y="pitchPadding"
            :width="(pitchWidth - pitchPadding * 2) / pitchStripeCount"
            :height="pitchHeight - pitchPadding * 2"
            :class="{ alternate: stripe % 2 === 0 }"
          />
        </g>
        <g class="pitch-markings" aria-label="Football pitch markings">
          <rect class="pitch-boundary" :x="pitchPadding" :y="pitchPadding" :width="pitchWidth - pitchPadding * 2" :height="pitchHeight - pitchPadding * 2" />
          <line class="halfway-line" :x1="pitchWidth / 2" :y1="pitchPadding" :x2="pitchWidth / 2" :y2="pitchHeight - pitchPadding" />
          <ellipse
            class="centre-circle"
            :cx="pitchX(0)"
            :cy="pitchY(0)"
            :rx="pitchX(9.15) - pitchX(0)"
            :ry="pitchY(9.15) - pitchY(0)"
          />
          <circle class="pitch-spot centre-spot" :cx="pitchX(0)" :cy="pitchY(0)" r="2" />

          <rect class="penalty-area left" :x="pitchX(-pitch.length / 2)" :y="pitchY(-20.16)" :width="pitchX(-pitch.length / 2 + 16.5) - pitchX(-pitch.length / 2)" :height="pitchY(20.16) - pitchY(-20.16)" />
          <rect class="penalty-area right" :x="pitchX(pitch.length / 2 - 16.5)" :y="pitchY(-20.16)" :width="pitchX(pitch.length / 2) - pitchX(pitch.length / 2 - 16.5)" :height="pitchY(20.16) - pitchY(-20.16)" />
          <rect class="goal-area left" :x="pitchX(-pitch.length / 2)" :y="pitchY(-9.16)" :width="pitchX(-pitch.length / 2 + 5.5) - pitchX(-pitch.length / 2)" :height="pitchY(9.16) - pitchY(-9.16)" />
          <rect class="goal-area right" :x="pitchX(pitch.length / 2 - 5.5)" :y="pitchY(-9.16)" :width="pitchX(pitch.length / 2) - pitchX(pitch.length / 2 - 5.5)" :height="pitchY(9.16) - pitchY(-9.16)" />
          <rect class="goal left" :x="pitchPadding - 6" :y="pitchY(-3.66)" width="6" :height="pitchY(3.66) - pitchY(-3.66)" />
          <rect class="goal right" :x="pitchWidth - pitchPadding" :y="pitchY(-3.66)" width="6" :height="pitchY(3.66) - pitchY(-3.66)" />
          <circle class="pitch-spot penalty-spot left" :cx="pitchX(-pitch.length / 2 + 11)" :cy="pitchY(0)" r="1.7" />
          <circle class="pitch-spot penalty-spot right" :cx="pitchX(pitch.length / 2 - 11)" :cy="pitchY(0)" r="1.7" />
          <path class="penalty-arc left" :d="`M ${pitchX(-pitch.length / 2 + 16.5)} ${pitchY(-7.31)} A ${pitchX(9.15) - pitchX(0)} ${pitchY(9.15) - pitchY(0)} 0 0 1 ${pitchX(-pitch.length / 2 + 16.5)} ${pitchY(7.31)}`" />
          <path class="penalty-arc right" :d="`M ${pitchX(pitch.length / 2 - 16.5)} ${pitchY(-7.31)} A ${pitchX(9.15) - pitchX(0)} ${pitchY(9.15) - pitchY(0)} 0 0 0 ${pitchX(pitch.length / 2 - 16.5)} ${pitchY(7.31)}`" />
          <path class="corner-arc" :d="`M ${pitchPadding} ${pitchPadding + 5} A 5 5 0 0 0 ${pitchPadding + 5} ${pitchPadding}`" />
          <path class="corner-arc" :d="`M ${pitchWidth - pitchPadding - 5} ${pitchPadding} A 5 5 0 0 0 ${pitchWidth - pitchPadding} ${pitchPadding + 5}`" />
          <path class="corner-arc" :d="`M ${pitchPadding} ${pitchHeight - pitchPadding - 5} A 5 5 0 0 1 ${pitchPadding + 5} ${pitchHeight - pitchPadding}`" />
          <path class="corner-arc" :d="`M ${pitchWidth - pitchPadding - 5} ${pitchHeight - pitchPadding} A 5 5 0 0 1 ${pitchWidth - pitchPadding} ${pitchHeight - pitchPadding - 5}`" />
        </g>
        <g class="pitch-axis-labels" aria-hidden="true">
          <text :x="pitchPadding" :y="pitchHeight - 1">−X</text>
          <text :x="pitchWidth / 2" :y="pitchHeight - 1" text-anchor="middle">0</text>
          <text :x="pitchWidth - pitchPadding" :y="pitchHeight - 1" text-anchor="end">+X →</text>
        </g>
        <g v-if="sample.effectivePitch" class="current-marker" :class="sample.severity">
          <circle class="current-halo" :cx="pitchX(sample.effectivePitch.x)" :cy="pitchY(sample.effectivePitch.z)" r="11" />
          <circle class="current-point" :cx="pitchX(sample.effectivePitch.x)" :cy="pitchY(sample.effectivePitch.z)" r="6" />
          <line :x1="pitchX(sample.effectivePitch.x) - 14" :y1="pitchY(sample.effectivePitch.z)" :x2="pitchX(sample.effectivePitch.x) + 14" :y2="pitchY(sample.effectivePitch.z)" />
          <line :x1="pitchX(sample.effectivePitch.x)" :y1="pitchY(sample.effectivePitch.z) - 14" :x2="pitchX(sample.effectivePitch.x)" :y2="pitchY(sample.effectivePitch.z) + 14" />
        </g>
        <circle
          v-if="sample.previous?.effectivePitch"
          class="previous-point"
          :cx="pitchX(sample.previous.effectivePitch.x)"
          :cy="pitchY(sample.previous.effectivePitch.z)"
          r="8"
        />
        </svg>
        <div class="projection-minimap-controls" role="group" aria-label="Projection minimap controls">
          <button type="button" aria-label="Zoom out projection minimap" :disabled="minimapZoom <= 1" @click="zoomMinimap(-1)">−</button>
          <output :aria-label="`Projection minimap zoom ${minimapZoom.toFixed(1)} times`">{{ minimapZoom.toFixed(1) }}×</output>
          <button type="button" aria-label="Zoom in projection minimap" :disabled="minimapZoom >= 6" @click="zoomMinimap(1)">+</button>
          <button type="button" class="reset" aria-label="Reset projection minimap" :disabled="minimapZoom <= 1" @click="resetMinimapViewport">Reset</button>
        </div>
        <small class="projection-minimap-hint">Wheel to zoom · drag to pan</small>
      </div>
      <div class="projection-pitch-legend" aria-label="Position marker legend">
        <span class="previous">Previous frame</span><span class="current">Current frame</span>
      </div>

      <dl class="projection-debug-metrics">
        <div><dt>Contact pixel</dt><dd>{{ number(sample.contactPoint.x, 1) }}, {{ number(sample.contactPoint.y, 1) }}</dd></div>
        <div><dt>Frame homography</dt><dd>{{ sample.matrixPitch ? `${number(sample.matrixPitch.x)}, ${number(sample.matrixPitch.z)}` : 'unavailable' }}</dd></div>
        <div><dt>Stored metric</dt><dd>{{ sample.storedPitch ? `${number(sample.storedPitch.x)}, ${number(sample.storedPitch.z)}` : 'unavailable' }}</dd></div>
        <div><dt>Stored ↔ matrix</dt><dd>{{ number(sample.storedMatrixDeltaMetres, 3) }} m</dd></div>
        <div><dt>Image movement</dt><dd>{{ number(sample.imageDeltaPixels, 1) }} px</dd></div>
        <div><dt>Camera-comp. image movement</dt><dd>{{ number(sample.cameraCompensatedImageDeltaPixels, 1) }} px</dd></div>
        <div><dt>Pitch movement</dt><dd>{{ number(sample.pitchDeltaMetres, 3) }} m</dd></div>
        <div><dt>Bbox-only effect</dt><dd>{{ number(sample.imageMotionPitchDeltaMetres, 3) }} m</dd></div>
        <div><dt>Raw matrix update</dt><dd>{{ number(sample.calibrationMotionPitchDeltaMetres, 3) }} m</dd></div>
        <div><dt>Motion-comp. matrix mismatch</dt><dd :class="{ 'identity-split': (sample.calibrationContinuityDeltaMetres ?? 0) > 0.5 }">{{ number(sample.calibrationContinuityDeltaMetres, 3) }} m</dd></div>
        <div><dt>Implied speed</dt><dd :class="sample.severity">{{ number(sample.speedMetresPerSecond, 1) }} m/s</dd></div>
        <div><dt>Metric verdict</dt><dd>{{ sample.observation.metricStatus || 'unknown' }}<template v-if="sample.observation.metricReason"> · {{ sample.observation.metricReason }}</template></dd></div>
        <div><dt>Calibration</dt><dd>#{{ sample.calibration?.sourceFrameIndex ?? '—' }} · {{ sample.calibration?.projectionSource || 'none' }} · {{ sample.calibration?.solutionStatus || sample.calibration?.status || 'missing' }}</dd></div>
        <div><dt>Calibration anchor</dt><dd>#{{ sample.observation.calibrationFrameIndex ?? '—' }}</dd></div>
        <div><dt>Uncertainty</dt><dd>{{ number(sample.calibration?.positionUncertaintyMetres ?? sample.observation.positionUncertaintyMetres, 3) }} m</dd></div>
        <div><dt>Frame line fit p50 / p95</dt><dd>{{ number(sample.calibration?.reprojectionError, 3) }} / {{ number(sample.calibration?.reprojectionP95, 3) }} px</dd></div>
        <div><dt>Keypoint ground p50 / p95</dt><dd>{{ number(sample.calibration?.groundErrorP50Metres, 3) }} / {{ number(sample.calibration?.groundErrorP95Metres, 3) }} m</dd></div>
        <div><dt>Contact policy</dt><dd>{{ contactPointProfile }}</dd></div>
        <div><dt>Same camera edge</dt><dd>{{ population.speedViolationCount }}/{{ population.comparedTrackCount }} tracks &gt;14 m/s</dd></div>
      </dl>

      <p v-if="synchronizedJump && calibrationDiscontinuity" class="projection-debug-warning">
        Synchronized jump and a motion-compensated matrix mismatch: inspect or recalibrate this frame. The frame-local pixel residual alone cannot detect this temporal discontinuity.
      </p>
      <p v-else-if="synchronizedJump" class="projection-debug-warning">
        Several players jump, but the current homography follows the measured camera motion. Inspect bbox contact points and the camera-compensated image movement before changing calibration.
      </p>
      <p v-else-if="sample.severity !== 'ok'" class="projection-debug-warning">
        The trajectory filter sees an impossible metric step. A small motion-compensated matrix mismatch points to the bbox/contact point; a large one points to calibration continuity.
      </p>

      <div class="projection-debug-timeline" aria-label="Object metric QA timeline">
        <button
          v-for="item in samples"
          :key="`${item.observation.frameIndex}-${item.observation.sceneTime}`"
          :class="[item.severity, { current: item.observation.frameIndex === sample.observation.frameIndex, rejected: item.observation.metricStatus === 'rejected' }]"
          :title="`#${item.observation.frameIndex} · ${item.observation.sceneTime.toFixed(3)}s · ${number(item.speedMetresPerSecond, 1)} m/s · ${item.observation.metricStatus || 'unknown'}`"
          @click="seek(item.observation.sceneTime)"
        />
      </div>
      <small class="projection-debug-help">Click timeline samples or use adjacent-frame buttons. Enable “Selected object debug” in Video View to see the stored bbox contact point over the exact camera overlay.</small>
    </template>
    <p v-else class="projection-debug-empty">This identity has no stored video observations.</p>
  </section>
</template>

<style scoped>
.projection-debug-card { display: grid; gap: 12px; padding: 13px; border: 1px solid rgba(113,226,170,.22); border-radius: 5px; background: rgba(113,226,170,.035); }
.projection-debug-card header { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; }
.projection-debug-card header div { display: grid; gap: 3px; }
.projection-debug-card header span { color: #71e2aa; font: 600 9px 'DM Mono', monospace; letter-spacing: .09em; }
.projection-debug-card header strong { color: #eff4f0; font-size: 14px; }
.projection-debug-card header i { padding: 4px 6px; border: 1px solid currentColor; border-radius: 3px; color: #71e2aa; font: 600 8px 'DM Mono', monospace; font-style: normal; text-align: right; }
.projection-debug-card header i.speed-violation { color: #ffd36a; }
.projection-debug-card header i.identity-split, .projection-debug-card header i.unprojected { color: #ff7867; }
.projection-frame-controls { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 5px; }
.projection-frame-controls button { min-height: 30px; border: 1px solid rgba(255,255,255,.1); border-radius: 3px; background: rgba(255,255,255,.035); color: #aeb8b2; font: 600 9px 'DM Mono', monospace; cursor: pointer; }
.projection-frame-controls button:disabled { opacity: .3; cursor: default; }
.projection-frame-controls .current { display: grid; place-items: center; color: #f1f4f2; }
.projection-frame-controls .current.distant { border-color: rgba(255,211,106,.45); color: #ffd36a; }
.projection-frame-controls small { font-size: 7px; }
.projection-recalibrate-frame { min-height: 32px; border: 1px solid rgba(255,211,106,.38); border-radius: 3px; background: rgba(255,211,106,.065); color: #f0d98c; font: 600 9px 'DM Mono', monospace; cursor: pointer; }
.projection-recalibrate-frame:hover:not(:disabled) { border-color: #ffd36a; color: #fff0ba; }
.projection-recalibrate-frame:disabled { opacity: .45; cursor: wait; }
.projection-minimap-shell { position: relative; overflow: hidden; border-radius: 4px; }
.projection-mini-pitch { display: block; width: 100%; aspect-ratio: 304 / 196; border: 1px solid rgba(152,224,185,.24); border-radius: 4px; background: #14271d; touch-action: none; user-select: none; cursor: zoom-in; }
.projection-mini-pitch:focus-visible { outline: 2px solid rgba(255,211,106,.9); outline-offset: -2px; }
.projection-minimap-shell.zoomed .projection-mini-pitch { cursor: grab; }
.projection-minimap-shell.panning .projection-mini-pitch { cursor: grabbing; }
.projection-minimap-controls { position: absolute; z-index: 2; top: 7px; right: 7px; display: grid; grid-template-columns: 24px 34px 24px 39px; gap: 3px; align-items: center; padding: 3px; border: 1px solid rgba(255,255,255,.14); border-radius: 3px; background: rgba(7,12,9,.86); box-shadow: 0 4px 14px rgba(0,0,0,.3); backdrop-filter: blur(6px); }
.projection-minimap-controls button { height: 23px; padding: 0; border: 1px solid rgba(255,255,255,.16); border-radius: 2px; background: #111a15; color: #e8f0eb; font: 700 12px 'DM Mono', monospace; cursor: pointer; }
.projection-minimap-controls button.reset { font-size: 7px; text-transform: uppercase; }
.projection-minimap-controls button:disabled { opacity: .3; cursor: default; }
.projection-minimap-controls button:focus-visible { border-color: #ffd36a; outline: 1px solid #ffd36a; }
.projection-minimap-controls output { color: #d8e4dc; font: 600 8px 'DM Mono', monospace; text-align: center; }
.projection-minimap-hint { position: absolute; z-index: 1; left: 8px; bottom: 7px; padding: 3px 5px; border-radius: 2px; background: rgba(6,10,8,.68); color: rgba(223,235,227,.68); font: 500 7px 'DM Mono', monospace; pointer-events: none; }
.pitch-stripes rect { fill: rgba(255,255,255,.018); stroke: none; }
.pitch-stripes rect.alternate { fill: rgba(110,210,155,.055); }
.pitch-markings > * { fill: none; stroke: rgba(235,246,239,.72); stroke-width: 1.25; vector-effect: non-scaling-stroke; }
.pitch-markings .pitch-boundary, .pitch-markings .halfway-line { stroke: rgba(247,252,249,.9); stroke-width: 1.5; }
.pitch-markings .goal { stroke: rgba(255,211,106,.85); stroke-width: 1.5; }
.pitch-markings .pitch-spot { fill: rgba(247,252,249,.95); stroke: #14271d; stroke-width: .8; }
.pitch-axis-labels text { fill: rgba(213,229,219,.68); font: 600 7px 'DM Mono', monospace; letter-spacing: .04em; }
.projection-mini-pitch .previous-point { fill: rgba(239,245,241,.12); stroke: rgba(239,245,241,.82); stroke-width: 1.5; stroke-dasharray: 3 2; vector-effect: non-scaling-stroke; }
.projection-mini-pitch .current-marker { color: #71e2aa; }
.projection-mini-pitch .current-marker.speed-violation { color: #ffd36a; }
.projection-mini-pitch .current-marker.identity-split, .projection-mini-pitch .current-marker.unprojected { color: #ff7867; }
.projection-mini-pitch .current-halo { fill: currentColor; opacity: .22; stroke: currentColor; stroke-width: 1; }
.projection-mini-pitch .current-point { fill: currentColor; stroke: #08100c; stroke-width: 2.5; vector-effect: non-scaling-stroke; }
.projection-mini-pitch .current-marker line { stroke: currentColor; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
.projection-pitch-legend { display: flex; flex-wrap: wrap; gap: 5px 12px; color: #7f8b85; font: 500 8px 'DM Mono', monospace; }
.projection-pitch-legend span::before { content: ''; display: inline-block; width: 7px; height: 7px; margin-right: 5px; border-radius: 50%; background: #71e2aa; vertical-align: -1px; }
.projection-pitch-legend span.previous::before { background: rgba(239,245,241,.18); border: 1px solid rgba(239,245,241,.62); box-sizing: border-box; }
.projection-debug-metrics { margin: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 0 12px; }
.projection-debug-metrics > div { min-width: 0; padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,.06); }
.projection-debug-metrics dt { color: #77837d; font: 500 8px 'DM Mono', monospace; text-transform: uppercase; }
.projection-debug-metrics dd { margin: 3px 0 0; color: #dce3df; font: 500 10px/1.35 'DM Mono', monospace; overflow-wrap: anywhere; }
.projection-debug-metrics dd.speed-violation { color: #ffd36a; }
.projection-debug-metrics dd.identity-split { color: #ff7867; }
.projection-debug-warning { margin: 0; padding: 9px; border-left: 2px solid #ffd36a; background: rgba(255,211,106,.07); color: #e7d59e; font: 500 10px/1.5 'DM Mono', monospace; }
.projection-debug-timeline { height: 28px; display: flex; align-items: stretch; gap: 1px; }
.projection-debug-timeline button { min-width: 1px; flex: 1 1 0; padding: 0; border: 0; border-radius: 1px; background: rgba(113,226,170,.42); cursor: pointer; }
.projection-debug-timeline button.speed-violation { background: #ffd36a; }
.projection-debug-timeline button.identity-split, .projection-debug-timeline button.unprojected { background: #ff7867; }
.projection-debug-timeline button.rejected { box-shadow: inset 0 -8px rgba(255,120,103,.55); }
.projection-debug-timeline button.current { outline: 2px solid #fff; outline-offset: 1px; z-index: 1; }
.projection-debug-help, .projection-debug-empty { margin: 0; color: #7f8b85; font: 500 9px/1.5 'DM Mono', monospace; }
</style>
