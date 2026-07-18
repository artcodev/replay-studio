<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  clampManualBallTime,
  MANUAL_BALL_TIME_STEP,
  manualBallTimelineEvents,
  type ManualBallTimelineEvent,
  type ManualBallTimelineState,
} from '../features/manual-ball/manualBallTimelineDomain'
import type { Keyframe } from '../types/tracking'

const props = withDefaults(defineProps<{
  duration: number
  currentTime: number
  keyframes: Keyframe[]
  selectedTime: number | null
  saving?: boolean
  disabled?: boolean
}>(), {
  saving: false,
  disabled: false,
})

const emit = defineEmits<{
  seek: [time: number]
  add: [time: number]
  select: [time: number]
  remove: [time: number]
  updateTime: [value: { from: number; to: number }]
}>()

const timeline = ref<HTMLElement | null>(null)

function finite(value: number, fallback = 0) {
  return Number.isFinite(value) ? value : fallback
}

const safeDuration = computed(() => Math.max(0, finite(props.duration)))

function clampTime(value: number) {
  return clampManualBallTime(value, safeDuration.value)
}

function sameTime(first: number, second: number) {
  return Math.abs(first - second) < MANUAL_BALL_TIME_STEP / 2
}

const markers = computed(() => {
  const unique = new Map<number, { keyframe: Keyframe; time: number }>()
  for (const keyframe of props.keyframes) {
    const time = clampTime(keyframe.t)
    if (!unique.has(time)) unique.set(time, { keyframe, time })
  }
  return [...unique.values()].sort((first, second) => first.time - second.time)
})

const selectedMarker = computed(() => {
  if (props.selectedTime === null || !Number.isFinite(props.selectedTime)) return null
  const selected = clampTime(props.selectedTime)
  return markers.value.find((marker) => sameTime(marker.time, selected)) ?? null
})

const selectedInput = ref('')

watch(
  () => selectedMarker.value?.time ?? null,
  (time) => {
    selectedInput.value = time === null ? '' : time.toFixed(3)
  },
  { immediate: true },
)

const playheadPercent = computed(() => percentAt(clampTime(props.currentTime)))

const interpolationSegments = computed(() => markers.value.slice(0, -1).map((marker, index) => {
  const next = markers.value[index + 1]
  const left = percentAt(marker.time)
  const right = percentAt(next.time)
  return {
    from: marker.time,
    to: next.time,
    left,
    width: Math.max(0, right - left),
  }
}))

const interactionDisabled = computed(() => props.disabled || props.saving)

function percentAt(time: number) {
  return safeDuration.value > 0 ? (time / safeDuration.value) * 100 : 0
}

function formatTime(time: number) {
  const clamped = clampTime(time)
  const minutes = Math.floor(clamped / 60)
  const seconds = clamped - minutes * 60
  return `${String(minutes).padStart(2, '0')}:${seconds.toFixed(3).padStart(6, '0')}`
}

function actionState(): ManualBallTimelineState {
  return {
    duration: safeDuration.value,
    currentTime: props.currentTime,
    keyframeTimes: markers.value.map((marker) => marker.time),
    selectedTime: selectedMarker.value?.time ?? null,
  }
}

function dispatchEvents(events: ManualBallTimelineEvent[]) {
  for (const event of events) {
    if (event.type === 'updateTime') emit('updateTime', event.value)
    else if (event.type === 'seek') emit('seek', event.time)
    else if (event.type === 'add') emit('add', event.time)
    else if (event.type === 'select') emit('select', event.time)
    else emit('remove', event.time)
  }
}

function selectMarker(time: number) {
  if (interactionDisabled.value) return
  dispatchEvents(manualBallTimelineEvents(actionState(), { type: 'select', time }))
}

function addKeypoint() {
  if (interactionDisabled.value) return
  dispatchEvents(manualBallTimelineEvents(actionState(), { type: 'add' }))
}

function removeSelected() {
  if (interactionDisabled.value || !selectedMarker.value) return
  dispatchEvents(manualBallTimelineEvents(actionState(), { type: 'remove' }))
}

function editSelectedTime(event: Event) {
  selectedInput.value = (event.target as HTMLInputElement).value
}

function updateSelectedTime() {
  const selected = selectedMarker.value
  if (interactionDisabled.value || !selected) return

  const requested = Number(selectedInput.value)
  if (!Number.isFinite(requested)) {
    selectedInput.value = selected.time.toFixed(3)
    return
  }

  const events = manualBallTimelineEvents(actionState(), { type: 'update-time', requestedTime: requested })
  const update = events.find((event) => event.type === 'updateTime')
  selectedInput.value = update?.type === 'updateTime' ? update.value.to.toFixed(3) : selected.time.toFixed(3)
  dispatchEvents(events)
}

function seekFromTimeline(event: MouseEvent) {
  if (interactionDisabled.value || !timeline.value || safeDuration.value <= 0) return
  if ((event.target as HTMLElement).closest('.ball-keypoint-marker')) return
  const bounds = timeline.value.getBoundingClientRect()
  if (bounds.width <= 0) return
  const ratio = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width))
  emit('seek', clampTime(ratio * safeDuration.value))
}
</script>

<template>
  <section class="manual-ball-timeline" aria-labelledby="ball-keypoints-title">
    <header class="timeline-header">
      <div>
        <h3 id="ball-keypoints-title">Ball keypoints</h3>
        <p>Manual positions are interpolated between neighboring keypoints.</p>
      </div>
      <span class="keypoint-count">{{ markers.length }} {{ markers.length === 1 ? 'keypoint' : 'keypoints' }}</span>
    </header>

    <div
      ref="timeline"
      class="timeline-track"
      :class="{ disabled: interactionDisabled }"
      role="group"
      aria-label="Manual ball keypoint timeline"
      data-testid="ball-timeline"
      @click="seekFromTimeline"
    >
      <div class="timeline-rail" aria-hidden="true" />
      <div
        v-for="segment in interpolationSegments"
        :key="`${segment.from}-${segment.to}`"
        class="interpolation-segment"
        :style="{ left: `${segment.left}%`, width: `${segment.width}%` }"
        :title="`Interpolated ${formatTime(segment.from)}–${formatTime(segment.to)}`"
        aria-hidden="true"
      />
      <button
        v-for="(marker, index) in markers"
        :key="marker.time"
        type="button"
        class="ball-keypoint-marker"
        :class="{ selected: selectedMarker?.time === marker.time }"
        :style="{ left: `${percentAt(marker.time)}%` }"
        :disabled="interactionDisabled"
        :aria-label="`Ball keypoint ${index + 1} at ${formatTime(marker.time)}`"
        :aria-pressed="selectedMarker?.time === marker.time"
        :title="`Ball keypoint · ${formatTime(marker.time)}`"
        :data-time="marker.time"
        @click.stop="selectMarker(marker.time)"
      >
        <span aria-hidden="true" />
      </button>
      <div
        class="timeline-playhead"
        :style="{ left: `${playheadPercent}%` }"
        :title="`Playhead · ${formatTime(currentTime)}`"
        aria-hidden="true"
      />
      <p v-if="markers.length === 0" class="empty-hint">Add the first keypoint at the current frame.</p>
    </div>

    <div class="timeline-scale" aria-hidden="true">
      <span>{{ formatTime(0) }}</span>
      <strong>{{ formatTime(currentTime) }}</strong>
      <span>{{ formatTime(safeDuration) }}</span>
    </div>

    <div class="timeline-controls">
      <button
        type="button"
        class="primary-action"
        :disabled="interactionDisabled"
        @click="addKeypoint"
      >
        <span aria-hidden="true">＋</span>
        Add keypoint
      </button>

      <label class="time-field">
        <span>Selected time</span>
        <input
          :value="selectedInput"
          data-testid="selected-keypoint-time"
          type="number"
          min="0"
          :max="safeDuration"
          step="0.001"
          inputmode="decimal"
          :disabled="interactionDisabled || !selectedMarker"
          aria-label="Selected ball keypoint time in seconds"
          placeholder="—"
          @input="editSelectedTime"
          @change="updateSelectedTime"
          @keydown.enter.prevent="updateSelectedTime"
        />
        <small>s</small>
      </label>

      <button
        type="button"
        class="delete-action"
        :disabled="interactionDisabled || !selectedMarker"
        @click="removeSelected"
      >
        Delete selected
      </button>

      <span v-if="saving" class="saving-state" role="status">Saving…</span>
    </div>
  </section>
</template>

<style scoped>
.manual-ball-timeline {
  --accent: #ffd36a;
  --accent-strong: #ffbd38;
  container-type: inline-size;
  display: grid;
  grid-template-columns: 120px minmax(160px, 1fr) minmax(235px, auto);
  grid-template-rows: 52px 12px;
  column-gap: 14px;
  padding: 10px 12px;
  border: 1px solid rgba(255, 211, 106, .18);
  border-radius: 4px;
  background: linear-gradient(180deg, rgba(255, 211, 106, .035), rgba(8, 12, 10, .28));
  color: #e7ece7;
}

.timeline-header,
.timeline-controls,
.timeline-scale {
  display: flex;
  align-items: center;
}

.timeline-header {
  grid-column: 1;
  grid-row: 1 / 3;
  align-items: flex-start;
  justify-content: center;
  flex-direction: column;
  gap: 7px;
  min-width: 0;
}

h3,
p {
  margin: 0;
}

h3 {
  color: #f1f4ef;
  font-size: 13px;
  font-weight: 650;
  letter-spacing: .01em;
}

.timeline-header p {
  margin-top: 3px;
  color: #77817c;
  font-size: 10px;
  line-height: 1.35;
}

.keypoint-count,
.timeline-scale,
.time-field input,
.saving-state {
  font-family: 'DM Mono', monospace;
}

.keypoint-count {
  flex: 0 0 auto;
  padding: 4px 7px;
  border-radius: 10px;
  background: rgba(255, 211, 106, .08);
  color: #bca86e;
  font-size: 9px;
}

.timeline-track {
  grid-column: 2;
  grid-row: 1;
  position: relative;
  height: 52px;
  margin: 0 8px;
  cursor: crosshair;
}

.timeline-track.disabled {
  cursor: wait;
  opacity: .62;
}

.timeline-rail,
.interpolation-segment {
  position: absolute;
  top: 27px;
  height: 2px;
  transform: translateY(-50%);
  border-radius: 2px;
}

.timeline-rail {
  right: 0;
  left: 0;
  background: #303733;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .025);
}

.interpolation-segment {
  z-index: 1;
  background: linear-gradient(90deg, var(--accent-strong), #ffe4a0);
  box-shadow: 0 0 8px rgba(255, 201, 73, .22);
}

.ball-keypoint-marker {
  position: absolute;
  z-index: 3;
  top: 27px;
  width: 24px;
  height: 36px;
  padding: 0;
  transform: translate(-50%, -50%);
  border: 0;
  background: transparent;
  cursor: pointer;
}

.ball-keypoint-marker > span {
  display: block;
  width: 10px;
  height: 10px;
  margin: auto;
  transform: rotate(45deg);
  border: 2px solid #151916;
  border-radius: 2px;
  background: var(--accent);
  box-shadow: 0 0 0 1px rgba(255, 211, 106, .52), 0 2px 8px rgba(0, 0, 0, .55);
  transition: transform .14s ease, box-shadow .14s ease;
}

.ball-keypoint-marker:hover > span,
.ball-keypoint-marker.selected > span {
  transform: rotate(45deg) scale(1.28);
  box-shadow: 0 0 0 2px rgba(255, 211, 106, .35), 0 0 12px rgba(255, 211, 106, .5);
}

.ball-keypoint-marker:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 1px;
  border-radius: 3px;
}

.ball-keypoint-marker:disabled {
  cursor: wait;
}

.timeline-playhead {
  position: absolute;
  z-index: 2;
  top: 5px;
  bottom: 4px;
  width: 1px;
  transform: translateX(-50%);
  background: #f3f7f3;
  box-shadow: 0 0 5px rgba(242, 248, 242, .45);
  pointer-events: none;
}

.timeline-playhead::before {
  content: '';
  position: absolute;
  top: -1px;
  left: 50%;
  width: 7px;
  height: 7px;
  transform: translate(-50%, -50%) rotate(45deg);
  background: #f3f7f3;
  border-radius: 1px;
}

.empty-hint {
  position: absolute;
  z-index: 1;
  top: 34px;
  left: 50%;
  transform: translateX(-50%);
  white-space: nowrap;
  color: #636c67;
  font-size: 9px;
  pointer-events: none;
}

.timeline-scale {
  grid-column: 2;
  grid-row: 2;
  justify-content: space-between;
  margin: -2px 2px 0;
  color: #626b66;
  font-size: 8px;
}

.timeline-scale strong {
  color: #aab3ae;
  font-weight: 500;
}

.timeline-controls {
  grid-column: 3;
  grid-row: 1 / 3;
  flex-wrap: wrap;
  align-content: center;
  gap: 6px;
  margin: 0;
  padding: 0;
  border-top: 0;
}

.timeline-controls button {
  min-height: 32px;
  padding: 0 10px;
  border-radius: 3px;
  font-size: 10px;
  cursor: pointer;
}

.primary-action {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid rgba(255, 211, 106, .42);
  background: rgba(255, 211, 106, .1);
  color: #ffe4a0;
}

.delete-action {
  border: 1px solid rgba(255, 116, 116, .23);
  background: rgba(255, 116, 116, .04);
  color: #c99393;
}

.timeline-controls button:hover:not(:disabled) {
  filter: brightness(1.17);
}

.timeline-controls button:focus-visible,
.time-field input:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

.timeline-controls button:disabled {
  opacity: .38;
  cursor: not-allowed;
}

.time-field {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-left: 0;
  color: #808a84;
  font-size: 9px;
}

.time-field input {
  width: 82px;
  height: 30px;
  box-sizing: border-box;
  padding: 0 7px;
  border: 1px solid rgba(238, 243, 235, .12);
  border-radius: 3px;
  background: #111613;
  color: #e2e8e3;
  font-size: 10px;
}

.time-field input:disabled {
  color: #5f6762;
}

.time-field small {
  color: #626b66;
  font-size: 9px;
}

.saving-state {
  color: #bca86e;
  font-size: 9px;
}

@container (max-width: 650px) {
  .manual-ball-timeline {
    grid-template-columns: 112px minmax(0, 1fr);
    grid-template-rows: 52px 12px auto;
  }

  .timeline-controls {
    grid-column: 1 / -1;
    grid-row: 3;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(238, 243, 235, .07);
  }
}

@container (max-width: 430px) {
  .timeline-header {
    align-items: flex-start;
  }

  .time-field {
    order: 3;
    width: 100%;
    margin-left: 0;
  }
}
</style>
