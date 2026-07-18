<script setup lang="ts">
import { computed, ref } from 'vue'
import { PLAYER_ACTION_TYPES, playerActionColor, playerActionLabel } from '../lib/playerActions'
import type { PlayerAction, PlayerActionKeypointKind, PlayerActionType } from '../types/playerActions'
import {
  PLAYER_ACTION_KEYPOINT_KINDS,
  PLAYER_ACTION_TIME_STEP,
  clampPlayerActionTime,
  layoutPlayerActions,
  normalizePlayerAction,
  reducePlayerAction,
  type PlayerActionEdit,
} from '../features/player-actions/playerActionTimelineDomain'

const props = withDefaults(defineProps<{
  canonicalPersonId: string
  personLabel?: string
  duration: number
  currentTime: number
  actions: PlayerAction[]
  selectedActionId: string | null
  saving?: boolean
  disabled?: boolean
}>(), {
  personLabel: '',
  saving: false,
  disabled: false,
})

const emit = defineEmits<{
  seek: [time: number]
  add: [time: number]
  select: [actionId: string]
  update: [action: PlayerAction]
  remove: [actionId: string]
}>()

const timeline = ref<HTMLElement | null>(null)
const safeDuration = computed(() => Math.max(0, Number.isFinite(props.duration) ? props.duration : 0))
const interactionDisabled = computed(() => props.disabled || props.saving)
const actorActions = computed(() => props.actions.filter((action) => action.canonicalPersonId === props.canonicalPersonId))
const layout = computed(() => layoutPlayerActions(actorActions.value, safeDuration.value))
const laneCount = computed(() => Math.max(1, ...layout.value.map((item) => item.lane + 1)))
const timelineHeight = computed(() => Math.max(84, laneCount.value * 36 + 30))
const selectedAction = computed(() => {
  const found = actorActions.value.find((action) => action.id === props.selectedActionId)
  return found ? normalizePlayerAction(found, safeDuration.value) : null
})
const selectedActionEditable = computed(() => selectedAction.value?.source === 'manual')
const clampedCurrentTime = computed(() => clampPlayerActionTime(props.currentTime, safeDuration.value))

function percentAt(time: number) {
  return safeDuration.value > 0 ? (clampPlayerActionTime(time, safeDuration.value) / safeDuration.value) * 100 : 0
}

function formatTime(time: number) {
  const clamped = clampPlayerActionTime(time, safeDuration.value)
  const minutes = Math.floor(clamped / 60)
  const seconds = clamped - minutes * 60
  return `${String(minutes).padStart(2, '0')}:${seconds.toFixed(3).padStart(6, '0')}`
}

function keypointLabel(kind: PlayerActionKeypointKind) {
  return kind.split('-').map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(' ')
}

function selectAction(action: PlayerAction) {
  if (interactionDisabled.value) return
  emit('select', action.id)
  emit('seek', action.startTime)
}

function selectKeypoint(action: PlayerAction, time: number) {
  if (interactionDisabled.value) return
  emit('select', action.id)
  emit('seek', time)
}

function addAction() {
  if (interactionDisabled.value || !props.canonicalPersonId) return
  emit('add', clampedCurrentTime.value)
}

function updateSelected(edit: PlayerActionEdit) {
  if (interactionDisabled.value || !selectedAction.value || !selectedActionEditable.value) return
  emit('update', reducePlayerAction(selectedAction.value, edit, safeDuration.value))
}

function updateActionType(event: Event) {
  updateSelected({ type: 'set-type', value: (event.target as HTMLSelectElement).value as PlayerActionType })
}

function updateBoundary(boundary: 'start' | 'end', event: Event) {
  const input = event.target as HTMLInputElement
  const requested = Number(input.value)
  if (!Number.isFinite(requested) || !selectedAction.value) {
    input.value = selectedAction.value ? selectedAction.value[boundary === 'start' ? 'startTime' : 'endTime'].toFixed(3) : ''
    return
  }
  updateSelected({ type: boundary === 'start' ? 'set-start' : 'set-end', time: requested })
}

function addKeypoint() {
  if (!selectedAction.value) return
  updateSelected({
    type: 'add-keypoint',
    kind: 'contact',
    time: Math.min(selectedAction.value.endTime, Math.max(selectedAction.value.startTime, clampedCurrentTime.value)),
  })
}

function updateKeypointKind(index: number, event: Event) {
  updateSelected({
    type: 'update-keypoint',
    index,
    kind: (event.target as HTMLSelectElement).value as PlayerActionKeypointKind,
  })
}

function updateKeypointTime(index: number, event: Event) {
  const input = event.target as HTMLInputElement
  const requested = Number(input.value)
  const keypoint = selectedAction.value?.keypoints[index]
  if (!Number.isFinite(requested) || !keypoint) {
    input.value = keypoint?.time.toFixed(3) ?? ''
    return
  }
  updateSelected({ type: 'update-keypoint', index, time: requested })
}

function removeKeypoint(index: number) {
  updateSelected({ type: 'remove-keypoint', index })
}

function removeSelected() {
  if (interactionDisabled.value || !selectedAction.value || !selectedActionEditable.value) return
  emit('remove', selectedAction.value.id)
}

function seekFromTimeline(event: MouseEvent) {
  if (interactionDisabled.value || !timeline.value || safeDuration.value <= 0) return
  if ((event.target as HTMLElement).closest('.action-interval, .action-keypoint-marker')) return
  const bounds = timeline.value.getBoundingClientRect()
  if (bounds.width <= 0) return
  const ratio = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width))
  emit('seek', clampPlayerActionTime(ratio * safeDuration.value, safeDuration.value))
}
</script>

<template>
  <section class="player-action-timeline" aria-label="Player action timeline">
    <header class="action-header">
      <div>
        <p class="eyebrow">Player actions</p>
        <h3>{{ personLabel || canonicalPersonId || 'Select a player' }}</h3>
        <p>Intervals prepare reviewed movement and contact phases for future 3D animation.</p>
      </div>
      <div class="header-actions">
        <span class="action-count">{{ layout.length }} {{ layout.length === 1 ? 'action' : 'actions' }}</span>
        <button
          type="button"
          class="add-action"
          :disabled="interactionDisabled || !canonicalPersonId"
          @click="addAction"
        >
          <span aria-hidden="true">＋</span>
          Add action at {{ formatTime(clampedCurrentTime) }}
        </button>
      </div>
    </header>

    <div
      ref="timeline"
      class="action-track"
      :class="{ disabled: interactionDisabled }"
      :style="{ height: `${timelineHeight}px` }"
      role="group"
      :aria-label="`Actions for ${personLabel || canonicalPersonId || 'selected player'}`"
      data-testid="player-action-timeline"
      @click="seekFromTimeline"
    >
      <div class="timeline-rail" aria-hidden="true" />

      <template v-for="item in layout" :key="item.action.id">
        <button
          type="button"
          class="action-interval"
          :class="{
            selected: item.action.id === selectedActionId,
            suggested: item.action.status === 'suggested',
            rejected: item.action.status === 'rejected',
          }"
          :style="{
            left: `${percentAt(item.action.startTime)}%`,
            width: `${Math.max(0, percentAt(item.action.endTime) - percentAt(item.action.startTime))}%`,
            top: `${10 + item.lane * 36}px`,
            '--action-color': playerActionColor(item.action.type),
          }"
          :disabled="interactionDisabled"
          :aria-pressed="item.action.id === selectedActionId"
          :aria-label="`${playerActionLabel(item.action.type)} from ${formatTime(item.action.startTime)} to ${formatTime(item.action.endTime)}`"
          :title="`${playerActionLabel(item.action.type)} · ${formatTime(item.action.startTime)}–${formatTime(item.action.endTime)}`"
          @click.stop="selectAction(item.action)"
        >
          <span class="boundary start" aria-hidden="true" />
          <strong>{{ playerActionLabel(item.action.type) }}</strong>
          <span class="boundary end" aria-hidden="true" />
        </button>

        <button
          v-for="(keypoint, keypointIndex) in item.action.keypoints"
          :key="`${item.action.id}-${keypointIndex}-${keypoint.kind}-${keypoint.time}`"
          type="button"
          class="action-keypoint-marker"
          :class="{ selected: item.action.id === selectedActionId }"
          :style="{
            left: `${percentAt(keypoint.time)}%`,
            top: `${8 + item.lane * 36}px`,
            '--action-color': playerActionColor(item.action.type),
          }"
          :disabled="interactionDisabled"
          :aria-label="`${keypointLabel(keypoint.kind)} at ${formatTime(keypoint.time)} for ${playerActionLabel(item.action.type)}`"
          :title="`${keypointLabel(keypoint.kind)} · ${formatTime(keypoint.time)}`"
          @click.stop="selectKeypoint(item.action, keypoint.time)"
        >
          <span aria-hidden="true" />
        </button>
      </template>

      <div
        class="timeline-playhead"
        :style="{ left: `${percentAt(clampedCurrentTime)}%` }"
        :title="`Playhead · ${formatTime(clampedCurrentTime)}`"
        aria-hidden="true"
      />

      <p v-if="layout.length === 0" class="empty-state">
        {{ canonicalPersonId ? 'No actions yet. Add one at the current playhead.' : 'Select an identified player to edit actions.' }}
      </p>
    </div>

    <div class="timeline-scale" aria-hidden="true">
      <span>{{ formatTime(0) }}</span>
      <strong>{{ formatTime(clampedCurrentTime) }}</strong>
      <span>{{ formatTime(safeDuration) }}</span>
    </div>

    <div v-if="selectedAction" class="action-editor" data-testid="selected-action-editor">
      <div class="editor-heading">
        <div>
          <p class="eyebrow">Selected interval</p>
          <strong>{{ playerActionLabel(selectedAction.type) }}</strong>
        </div>
        <div class="action-badges">
          <span :class="`status-${selectedAction.status}`">{{ selectedAction.status }}</span>
          <span>{{ selectedAction.source }}</span>
        </div>
      </div>

      <p v-if="!selectedActionEditable" class="review-only" role="note">
        Automatic suggestion · review-only. Accept it before making manual adjustments.
      </p>

      <div class="interval-fields">
        <label>
          <span>Action</span>
          <select :value="selectedAction.type" :disabled="interactionDisabled || !selectedActionEditable" @change="updateActionType">
            <option v-for="type in PLAYER_ACTION_TYPES" :key="type" :value="type">{{ playerActionLabel(type) }}</option>
          </select>
        </label>
        <label>
          <span>Start</span>
          <div class="time-input">
            <input
              :value="selectedAction.startTime.toFixed(3)"
              type="number"
              min="0"
              :max="Math.max(0, selectedAction.endTime - PLAYER_ACTION_TIME_STEP)"
              step="0.001"
              :disabled="interactionDisabled || !selectedActionEditable"
              aria-label="Action start time in seconds"
              @change="updateBoundary('start', $event)"
            />
            <small>s</small>
          </div>
        </label>
        <label>
          <span>End</span>
          <div class="time-input">
            <input
              :value="selectedAction.endTime.toFixed(3)"
              type="number"
              :min="Math.min(safeDuration, selectedAction.startTime + PLAYER_ACTION_TIME_STEP)"
              :max="safeDuration"
              step="0.001"
              :disabled="interactionDisabled || !selectedActionEditable"
              aria-label="Action end time in seconds"
              @change="updateBoundary('end', $event)"
            />
            <small>s</small>
          </div>
        </label>
        <button type="button" class="delete-action" :disabled="interactionDisabled || !selectedActionEditable" @click="removeSelected">
          Delete action
        </button>
      </div>

      <div class="phase-editor">
        <div class="phase-heading">
          <div>
            <strong>Significant phases</strong>
            <p>Contact, release, apex and recovery markers stay inside the action interval.</p>
          </div>
          <button type="button" :disabled="interactionDisabled || !selectedActionEditable" @click="addKeypoint">＋ Add phase</button>
        </div>

        <div v-if="selectedAction.keypoints.length" class="phase-list">
          <div v-for="(keypoint, index) in selectedAction.keypoints" :key="`${index}-${keypoint.kind}-${keypoint.time}`" class="phase-row">
            <span class="phase-index">{{ index + 1 }}</span>
            <label>
              <span class="sr-only">Phase {{ index + 1 }} kind</span>
              <select :value="keypoint.kind" :disabled="interactionDisabled || !selectedActionEditable" @change="updateKeypointKind(index, $event)">
                <option v-for="kind in PLAYER_ACTION_KEYPOINT_KINDS" :key="kind" :value="kind">{{ keypointLabel(kind) }}</option>
              </select>
            </label>
            <label class="time-input">
              <span class="sr-only">Phase {{ index + 1 }} time in seconds</span>
              <input
                :value="keypoint.time.toFixed(3)"
                type="number"
                :min="selectedAction.startTime"
                :max="selectedAction.endTime"
                step="0.001"
                :disabled="interactionDisabled || !selectedActionEditable"
                :aria-label="`Phase ${index + 1} time in seconds`"
                @change="updateKeypointTime(index, $event)"
              />
              <small>s</small>
            </label>
            <button
              type="button"
              class="remove-phase"
              :disabled="interactionDisabled || !selectedActionEditable"
              :aria-label="`Remove phase ${index + 1}`"
              @click="removeKeypoint(index)"
            >
              Remove
            </button>
          </div>
        </div>
        <p v-else class="no-phases">No significant phases. The interval can still be used as a body-state action.</p>
      </div>
    </div>

    <p v-else-if="layout.length" class="select-hint">Select an interval or phase marker to edit it.</p>
    <span v-if="saving" class="saving-state" role="status">Saving action…</span>
  </section>
</template>

<style scoped>
.player-action-timeline {
  --panel: rgba(8, 13, 18, .88);
  --line: rgba(171, 193, 213, .18);
  display: grid;
  gap: 10px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(180deg, rgba(104, 169, 255, .055), var(--panel));
  color: #edf4fa;
}

button,
input,
select {
  font: inherit;
}

button {
  color: inherit;
}

.action-header,
.header-actions,
.editor-heading,
.action-badges,
.phase-heading,
.phase-row,
.timeline-scale {
  display: flex;
  align-items: center;
}

.action-header,
.editor-heading,
.phase-heading,
.timeline-scale {
  justify-content: space-between;
}

.action-header {
  gap: 20px;
}

.action-header h3,
.action-header p,
.editor-heading p,
.phase-heading p,
.no-phases,
.select-hint {
  margin: 0;
}

.action-header h3 {
  margin: 2px 0 3px;
  font-size: 17px;
}

.action-header > div:first-child > p:last-child,
.phase-heading p,
.no-phases,
.select-hint {
  color: #93a3b0;
  font-size: 12px;
}

.eyebrow {
  color: #74b5ff;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .14em;
  text-transform: uppercase;
}

.header-actions {
  flex: 0 0 auto;
  gap: 10px;
}

.action-count,
.action-badges span {
  padding: 4px 7px;
  border: 1px solid var(--line);
  border-radius: 999px;
  color: #9fb0be;
  font-size: 11px;
  text-transform: capitalize;
}

.add-action,
.phase-heading button {
  padding: 7px 10px;
  border: 1px solid rgba(104, 169, 255, .42);
  border-radius: 6px;
  background: rgba(104, 169, 255, .14);
  cursor: pointer;
}

.action-track {
  position: relative;
  min-width: 0;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 6px;
  background-color: rgba(2, 6, 10, .55);
  background-image: linear-gradient(90deg, transparent 24.9%, rgba(255, 255, 255, .035) 25%, transparent 25.1%, transparent 49.9%, rgba(255, 255, 255, .035) 50%, transparent 50.1%, transparent 74.9%, rgba(255, 255, 255, .035) 75%, transparent 75.1%);
  cursor: crosshair;
}

.action-track.disabled {
  cursor: default;
  opacity: .68;
}

.timeline-rail {
  position: absolute;
  right: 0;
  bottom: 16px;
  left: 0;
  height: 1px;
  background: var(--line);
}

.action-interval {
  --action-color: #68a9ff;
  position: absolute;
  z-index: 2;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 14px;
  height: 26px;
  overflow: hidden;
  padding: 0 7px;
  border: 1px solid color-mix(in srgb, var(--action-color), white 18%);
  border-radius: 5px;
  background: color-mix(in srgb, var(--action-color) 33%, #101820);
  box-shadow: inset 0 0 10px rgba(255, 255, 255, .035);
  cursor: pointer;
  transform: translateX(0);
}

.action-interval strong {
  overflow: hidden;
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.action-interval.suggested {
  border-style: dashed;
}

.action-interval.rejected {
  opacity: .38;
  filter: saturate(.35);
  text-decoration: line-through;
}

.action-interval.selected {
  z-index: 4;
  outline: 2px solid #fff;
  outline-offset: 2px;
}

.boundary {
  position: absolute;
  top: 2px;
  bottom: 2px;
  width: 2px;
  border-radius: 2px;
  background: #fff;
  opacity: .88;
}

.boundary.start { left: 2px; }
.boundary.end { right: 2px; }

.action-keypoint-marker {
  --action-color: #68a9ff;
  position: absolute;
  z-index: 5;
  width: 16px;
  height: 30px;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: pointer;
  transform: translateX(-50%);
}

.action-keypoint-marker span {
  position: absolute;
  top: -3px;
  left: 3px;
  width: 8px;
  height: 8px;
  border: 2px solid #0b1117;
  border-radius: 2px;
  background: var(--action-color);
  box-shadow: 0 0 0 1px var(--action-color);
  transform: rotate(45deg);
}

.action-keypoint-marker.selected span {
  background: #fff;
}

.timeline-playhead {
  position: absolute;
  z-index: 7;
  top: 0;
  bottom: 0;
  width: 2px;
  background: #ff536f;
  box-shadow: 0 0 7px rgba(255, 83, 111, .62);
  pointer-events: none;
  transform: translateX(-1px);
}

.timeline-playhead::before {
  position: absolute;
  top: 0;
  left: -4px;
  width: 0;
  height: 0;
  border-top: 6px solid #ff536f;
  border-right: 5px solid transparent;
  border-left: 5px solid transparent;
  content: '';
}

.empty-state {
  position: absolute;
  top: 50%;
  left: 50%;
  width: min(420px, 80%);
  margin: 0;
  color: #8797a4;
  font-size: 12px;
  text-align: center;
  transform: translate(-50%, -50%);
}

.timeline-scale {
  color: #71818e;
  font-size: 10px;
}

.timeline-scale strong {
  color: #ff7a90;
  font-variant-numeric: tabular-nums;
}

.action-editor {
  display: grid;
  gap: 12px;
  margin-top: 3px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: rgba(255, 255, 255, .025);
}

.action-badges {
  gap: 5px;
}

.action-badges .status-confirmed {
  border-color: rgba(91, 209, 142, .35);
  color: #7ce6aa;
}

.action-badges .status-suggested {
  border-color: rgba(244, 191, 79, .35);
  color: #f4bf4f;
}

.action-badges .status-rejected {
  border-color: rgba(255, 122, 109, .35);
  color: #ff8c81;
}

.review-only {
  margin: 0;
  padding: 8px 10px;
  border: 1px solid rgba(244, 191, 79, .28);
  border-radius: 5px;
  background: rgba(244, 191, 79, .07);
  color: #f1cb78;
  font-size: 11px;
}

.interval-fields {
  display: grid;
  grid-template-columns: minmax(140px, 1.4fr) minmax(105px, 1fr) minmax(105px, 1fr) auto;
  gap: 9px;
  align-items: end;
}

.interval-fields label,
.phase-row label {
  display: grid;
  gap: 4px;
  min-width: 0;
}

.interval-fields label > span {
  color: #8fa0ad;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .06em;
  text-transform: uppercase;
}

input,
select {
  min-width: 0;
  height: 32px;
  padding: 0 8px;
  border: 1px solid rgba(171, 193, 213, .25);
  border-radius: 5px;
  background: #101820;
  color: #edf4fa;
  font-variant-numeric: tabular-nums;
}

.time-input {
  position: relative;
}

.time-input input {
  width: 100%;
  box-sizing: border-box;
  padding-right: 22px;
}

.time-input small {
  position: absolute;
  right: 8px;
  bottom: 8px;
  color: #72828e;
  pointer-events: none;
}

.delete-action,
.remove-phase {
  height: 32px;
  padding: 0 9px;
  border: 1px solid rgba(255, 122, 109, .3);
  border-radius: 5px;
  background: rgba(255, 122, 109, .08);
  color: #ff9a91;
  cursor: pointer;
}

.phase-editor {
  display: grid;
  gap: 8px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
}

.phase-list {
  display: grid;
  gap: 6px;
}

.phase-row {
  display: grid;
  grid-template-columns: 24px minmax(130px, 1fr) minmax(110px, .7fr) auto;
  gap: 7px;
}

.phase-index {
  display: grid;
  width: 22px;
  height: 22px;
  place-items: center;
  border: 1px solid var(--line);
  border-radius: 50%;
  color: #91a3b0;
  font-size: 10px;
}

.saving-state {
  color: #f4bf4f;
  font-size: 11px;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  clip-path: inset(50%);
}

button:disabled,
input:disabled,
select:disabled {
  cursor: not-allowed;
  opacity: .5;
}

@container (max-width: 720px) {
  .action-header,
  .header-actions,
  .phase-heading {
    align-items: flex-start;
    flex-direction: column;
  }

  .interval-fields {
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 660px) {
  .interval-fields,
  .phase-row {
    grid-template-columns: 1fr;
  }

  .phase-index {
    display: none;
  }
}
</style>
