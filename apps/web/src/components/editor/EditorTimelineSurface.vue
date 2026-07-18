<script setup lang="ts">
import { computed } from 'vue'
import ManualBallTimeline from '../ManualBallTimeline.vue'
import PlayerActionTimeline from '../PlayerActionTimeline.vue'
import type { useManualBallEditor } from '../../composables/useManualBallEditor'
import type { usePlayerActionEditor } from '../../composables/usePlayerActionEditor'
import type { useSegmentLayoutEditor } from '../../composables/useSegmentLayoutEditor'
import type { VideoSegment } from '../../types/media'
import type { SceneDocument, SceneVideoAsset } from '../../types/scene'

type ManualBallEditor = ReturnType<typeof useManualBallEditor>
type PlayerActionEditor = ReturnType<typeof usePlayerActionEditor>
type SegmentLayoutEditor = ReturnType<typeof useSegmentLayoutEditor>

const props = defineProps<{
  scene: SceneDocument
  timelineScene: SceneDocument | null
  sceneVideo: SceneVideoAsset | null
  playing: boolean
  timeLabel: string
  reconstructionRunning: boolean
  reconstructionMutationLocked: boolean
  selectedActionActorId: string | null
  selectedActionActorLabel: string
  showPlayerActionTimeline: boolean
  segmentLayout: SegmentLayoutEditor
  manualBall: ManualBallEditor
  playerActions: PlayerActionEditor
}>()

const currentTime = defineModel<number>('currentTime', { required: true })
const playbackRate = defineModel<number>('playbackRate', { required: true })
const emit = defineEmits<{
  seek: [time: number]
  'master-segment': [segment: VideoSegment]
  'timeline-input': []
  'toggle-play': []
}>()

const masterScene = computed(() => props.timelineScene ?? props.scene)
const masterVideo = computed(() => masterScene.value.payload.videoAsset ?? null)
const masterLayout = computed(() => masterVideo.value?.segmentLayout ?? null)
const canEditMasterTimeline = computed(() => masterScene.value.id === props.scene.id)
const masterCurrentTime = computed(() => (
  canEditMasterTimeline.value
    ? currentTime.value
    : (props.sceneVideo?.sourceStart ?? 0) + currentTime.value
))
const hasSegmentMap = computed(() => Boolean(
  masterLayout.value && masterVideo.value?.segments?.length,
))

function segmentRoleLabel(role?: string) {
  if (role === 'original') return 'Original'
  if (role === 'replay') return 'Replay'
  return 'Continuation'
}

function timelineTick(seconds: number) {
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.floor(seconds % 60)
  return `${minutes}:${String(remainder).padStart(2, '0')}`
}
</script>

<template>
  <div
    class="timeline-panel"
    :class="{
      'has-segment-map': hasSegmentMap,
      'has-ball-timeline': manualBall.mode.value === 'manual',
      'has-player-action-timeline': showPlayerActionTimeline,
    }"
  >
    <div v-if="masterLayout && masterVideo?.segments?.length" class="master-timeline">
      <div class="master-timeline-heading">
        <div>
          <strong>Full video timeline</strong>
          <span>{{ masterLayout.groups.length }} events · {{ canEditMasterTimeline && segmentLayout.groupEditing.value ? `${segmentLayout.selection.value.length} selected` : canEditMasterTimeline ? 'click any segment to seek' : 'click any segment to open' }}</span>
        </div>
        <div v-if="canEditMasterTimeline" class="master-timeline-actions">
          <button :class="{ active: segmentLayout.groupEditing.value }" @click="segmentLayout.toggleGroupEditing">{{ segmentLayout.groupEditing.value ? 'Close edit' : 'Edit groups' }}</button>
          <button v-if="segmentLayout.groupEditing.value" class="split" :disabled="!segmentLayout.canSplitSelection.value" @click="segmentLayout.splitSelection">Split selected</button>
          <button v-if="segmentLayout.groupEditing.value" class="save" @click="segmentLayout.saveGroupMap">Save map</button>
        </div>
      </div>
      <div class="master-timeline-track" :class="{ editing: segmentLayout.groupEditing.value }" aria-label="Full video event timeline">
        <button
          v-for="segment in masterVideo.segments"
          :key="segment.id"
          :class="[segment.layout?.role, { selected: canEditMasterTimeline && segmentLayout.selection.value.includes(segment.id) }]"
          :style="{
            width: `${(segment.duration / masterScene.duration) * 100}%`,
            backgroundColor: `${segmentLayout.segmentGroupColor(segment.layout?.group)}20`,
            borderColor: segmentLayout.segmentGroupColor(segment.layout?.group),
          }"
          :aria-label="`${segment.layout?.label ?? segment.label}, ${segmentRoleLabel(segment.layout?.role)}, ${segment.start.toFixed(2)} to ${segment.end.toFixed(2)} seconds`"
          @click="emit('master-segment', segment)"
        >
          <strong>{{ segment.layout?.label ?? segment.label }}</strong>
          <small>{{ segment.start.toFixed(1) }}–{{ segment.end.toFixed(1) }}s</small>
          <i>{{ segmentRoleLabel(segment.layout?.role) }}</i>
        </button>
        <span class="master-playhead" :style="{ left: `${(masterCurrentTime / masterScene.duration) * 100}%` }" />
      </div>
    </div>
    <ManualBallTimeline
      v-if="manualBall.mode.value === 'manual'"
      :duration="scene.duration"
      :current-time="currentTime"
      :keyframes="manualBall.manualKeyframes.value"
      :selected-time="manualBall.selectedKeyframeTime.value"
      :saving="manualBall.saving.value"
      :disabled="reconstructionRunning"
      @seek="emit('seek', $event)"
      @add="manualBall.addKeypoint"
      @select="manualBall.selectKeypoint"
      @remove="manualBall.removeKeypoint"
      @update-time="manualBall.updateKeypointTime"
    />
    <PlayerActionTimeline
      v-if="showPlayerActionTimeline && selectedActionActorId"
      :canonical-person-id="selectedActionActorId"
      :person-label="selectedActionActorLabel"
      :duration="scene.duration"
      :current-time="currentTime"
      :actions="playerActions.selectedActorActions.value"
      :selected-action-id="playerActions.selectedActionId.value"
      :saving="playerActions.saving.value"
      :disabled="reconstructionMutationLocked"
      @seek="playerActions.seek"
      @add="playerActions.addAt"
      @select="playerActions.select"
      @update="playerActions.update"
      @remove="playerActions.remove"
    />
    <div class="transport">
      <button class="play-button" @click="emit('toggle-play')">{{ playing ? 'Ⅱ' : '▶' }}</button>
      <span class="timecode">{{ timeLabel }}</span>
      <select v-model="playbackRate" aria-label="Playback speed">
        <option :value="0.25">0.25×</option>
        <option :value="0.5">0.5×</option>
        <option :value="1">1×</option>
        <option :value="2">2×</option>
      </select>
    </div>
    <div class="timeline-track">
      <input v-model.number="currentTime" type="range" min="0" :max="scene.duration" step="0.01" @input="emit('timeline-input')" />
      <div class="event-markers">
        <button
          v-for="(binding, index) in scene.payload.eventBindings"
          :key="`${binding.externalEventId}-${index}`"
          :style="{ left: `${(binding.sceneTime / scene.duration) * 100}%` }"
          :title="binding.label"
          @click="emit('seek', binding.sceneTime)"
        />
      </div>
      <div class="timeline-scale">
        <span>{{ timelineTick(0) }}</span>
        <span>{{ timelineTick(scene.duration / 3) }}</span>
        <span>{{ timelineTick((scene.duration * 2) / 3) }}</span>
        <span>{{ timelineTick(scene.duration) }}</span>
      </div>
    </div>
    <span class="duration">{{ scene.duration.toFixed(2) }}s</span>
  </div>
</template>
