<script setup lang="ts">
import type { ComponentPublicInstance } from 'vue'
import PathTrackingLegend from '../PathTrackingLegend.vue'
import ThreeViewport from '../ThreeViewport.vue'
import type { FrameAnalysis } from '../../types/analysis'
import type { SceneDocument } from '../../types/scene'
import type { PlayerActionPlaybackState } from '../../lib/playerActions'
import type { PathTrackingSubjectKind } from '../../lib/pathTracking'
import type { ThreeRenderQuality, ThreeViewOptions } from '../../lib/threeViewOptions'

type ViewportApi = {
  cameraPreset: (name: 'broadcast' | 'orbit' | 'tactical' | 'goal') => void
}

defineProps<{
  scene: SceneDocument
  currentTime: number
  selectedTrackId: string | null
  editMode: boolean
  ballEditMode: boolean
  selectedBallKeyframeTime: number | null
  ballSelected: boolean
  options: ThreeViewOptions
  renderQuality: ThreeRenderQuality
  frameAnalysis: FrameAnalysis | null
  activePlayerAction: PlayerActionPlaybackState | null
  pathSubject: {
    kind: PathTrackingSubjectKind
    label: string
    color: string
    sampleCount: number
  } | null
  hasDrawablePath: boolean
  unavailablePathLabel: string | null
}>()

const emit = defineEmits<{
  viewportElement: [viewport: ViewportApi | null]
  selectTrack: [trackId: string]
  selectBall: []
  moveTrack: [position: { x: number; z: number }]
  moveBall: [position: { x: number; z: number }]
}>()

function bindViewport(viewport: Element | ComponentPublicInstance | null) {
  emit(
    'viewportElement',
    viewport && 'cameraPreset' in viewport ? viewport as unknown as ViewportApi : null,
  )
}
</script>

<template>
  <div class="three-pane">
    <ThreeViewport
      :ref="bindViewport"
      :scene="scene"
      :current-time="currentTime"
      :selected-track-id="selectedTrackId"
      :edit-mode="editMode"
      :ball-edit-mode="ballEditMode"
      :selected-ball-keyframe-time="selectedBallKeyframeTime"
      :show-models="options.models"
      :show-trails="options.trajectory"
      :show-path-tracking="options.pathTracking"
      :ball-selected="ballSelected"
      :show-labels="options.labels"
      :show-ball="options.ball"
      :show-analysis-markers="options.analysisMarkers"
      :render-quality="renderQuality"
      :frame-analysis="frameAnalysis"
      :active-player-action="activePlayerAction"
      @select="emit('selectTrack', $event)"
      @select-ball="emit('selectBall')"
      @move-track="emit('moveTrack', $event)"
      @move-ball="emit('moveBall', $event)"
    />
    <PathTrackingLegend
      :enabled="options.pathTracking"
      :subject-kind="pathSubject?.kind"
      :subject-label="pathSubject?.label"
      :subject-color="pathSubject?.color"
      :sample-count="pathSubject?.sampleCount"
      :has-drawable-path="hasDrawablePath"
      :unavailable-label="unavailablePathLabel"
      surface-label="3D scene"
    />
  </div>
</template>
