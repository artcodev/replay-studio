<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { FrameAnalysis } from '../types/analysis'
import type { SceneDocument } from '../types/scene'
import type { PlayerActionPlaybackState } from '../lib/playerActions'
import { renderQualityProfile, type RenderQuality } from '../lib/renderQuality'
import { AnalysisMarkerLayer } from '../features/three-viewport/analysisMarkerLayer'
import { BallLayer } from '../features/three-viewport/ballLayer'
import { PitchLayer } from '../features/three-viewport/pitchLayer'
import { PlayerLayer } from '../features/three-viewport/playerLayer'
import { SelectionLayer } from '../features/three-viewport/selectionLayer'
import {
  resolveSelectedPathSource,
  SelectedPathLayer,
} from '../features/three-viewport/selectedPathLayer'
import {
  type CameraPreset,
  ViewportRenderSurface,
} from '../features/three-viewport/viewportRenderSurface'
import { ViewportPointerSelection } from '../features/three-viewport/viewportPointerSelection'
import PlayerActionPreviewHud from './PlayerActionPreviewHud.vue'

const props = withDefaults(defineProps<{
  scene: SceneDocument
  currentTime: number
  selectedTrackId: string | null
  editMode: boolean
  ballEditMode?: boolean
  selectedBallKeyframeTime?: number | null
  showTrails: boolean
  showLabels: boolean
  showModels?: boolean
  showBall?: boolean
  showAnalysisMarkers?: boolean
  showPathTracking?: boolean
  ballSelected?: boolean
  activePlayerAction?: PlayerActionPlaybackState | null
  frameAnalysis: FrameAnalysis | null
  renderQuality?: RenderQuality
}>(), {
  showModels: true,
  showBall: true,
  showAnalysisMarkers: true,
  showPathTracking: false,
  ballSelected: false,
  activePlayerAction: null,
  ballEditMode: false,
  selectedBallKeyframeTime: null,
  renderQuality: 'basic',
})

const emit = defineEmits<{
  select: [trackId: string]
  selectBall: []
  moveTrack: [position: { x: number; z: number }]
  moveBall: [position: { x: number; z: number }]
}>()

const host = ref<HTMLDivElement | null>(null)
const status = ref('Broadcast camera · drag to inspect')

let surface: ViewportRenderSurface | null = null
let pitchLayer: PitchLayer | null = null
let playerLayer: PlayerLayer | null = null
let ballLayer: BallLayer | null = null
let analysisMarkerLayer: AnalysisMarkerLayer | null = null
let selectedPathLayer: SelectedPathLayer | null = null
let selectionLayer: SelectionLayer | null = null
let pointerSelection: ViewportPointerSelection | null = null

const selectedPathSource = computed(() => (
  resolveSelectedPathSource(props.scene, props.selectedTrackId, props.ballSelected)
))

function rebuildPlayers() {
  playerLayer?.rebuild(props.scene.payload.tracks, {
    showModels: props.showModels,
    showLabels: props.showLabels,
    shadows: renderQualityProfile(props.renderQuality).shadows,
  })
  updateObjects()
}

function rebuildBallTrail() {
  ballLayer?.rebuildTrail(props.scene.payload.ball.keyframes)
  updateObjects()
}

function rebuildSelectedPath() {
  selectedPathLayer?.rebuild(selectedPathSource.value, props.showPathTracking, props.currentTime)
  updateObjects()
}

function updateObjects() {
  const tracks = props.scene.payload.tracks
  const ballFrames = props.scene.payload.ball.keyframes
  playerLayer?.update(tracks, props.currentTime, props.scene.duration)
  selectedPathLayer?.update(selectedPathSource.value, props.showPathTracking, props.currentTime)
  ballLayer?.update(
    ballFrames,
    props.currentTime,
    props.showBall,
    props.showTrails,
    selectedPathLayer?.replacesBallTrail ?? false,
  )
  selectionLayer?.updateBall(
    props.ballEditMode,
    props.selectedBallKeyframeTime ?? null,
    ballFrames,
  )
  selectionLayer?.updatePlayer(playerLayer?.selectedPosition(props.selectedTrackId) ?? null)
}

function applyRenderQuality() {
  if (!surface || !pitchLayer || !playerLayer || !ballLayer) return
  surface.applyQuality(props.renderQuality, [pitchLayer, playerLayer, ballLayer])
}

function cameraPreset(name: CameraPreset) {
  surface?.cameraPreset(name)
  status.value = `${name[0].toUpperCase()}${name.slice(1)} camera`
}

onMounted(() => {
  if (!host.value) return
  surface = new ViewportRenderSurface(host.value)
  const shadows = renderQualityProfile(props.renderQuality).shadows
  pitchLayer = new PitchLayer(surface.scene, props.scene.payload.pitch, shadows)
  playerLayer = new PlayerLayer(surface.scene)
  ballLayer = new BallLayer(surface.scene, shadows)
  analysisMarkerLayer = new AnalysisMarkerLayer(surface.scene)
  selectedPathLayer = new SelectedPathLayer(surface.scene)
  selectionLayer = new SelectionLayer(surface.scene)

  rebuildPlayers()
  rebuildBallTrail()
  rebuildSelectedPath()
  analysisMarkerLayer.rebuild(props.frameAnalysis)
  analysisMarkerLayer.setVisible(props.showAnalysisMarkers)
  applyRenderQuality()
  surface.setBallEditCursor(props.ballEditMode)

  pointerSelection = new ViewportPointerSelection(
    surface.renderer,
    surface.camera,
    () => ({
      pitch: pitchLayer?.surface ?? null,
      ball: ballLayer?.mesh ?? null,
      players: playerLayer?.raycastTargets ?? [],
      ballEditMode: props.ballEditMode,
      editMode: props.editMode,
      selectedTrackId: props.selectedTrackId,
    }),
    {
      selectTrack: (trackId) => emit('select', trackId),
      selectBall: () => emit('selectBall'),
      moveTrack: (position) => emit('moveTrack', position),
      moveBall: (position) => emit('moveBall', position),
    },
  )

  surface.startAnimationLoop((time) => {
    updateObjects()
    selectionLayer?.animate(time / 1000)
    selectedPathLayer?.animate(time / 1000)
  })
})

watch(
  () => props.scene.payload.tracks.map(
    (track) => `${track.id}:${track.number}:${track.label}:${track.color}`,
  ).join('|'),
  rebuildPlayers,
)
watch(() => JSON.stringify(props.scene.payload.ball.keyframes), rebuildBallTrail)
watch(
  () => [
    props.selectedTrackId,
    props.ballSelected,
    props.ballSelected
      ? JSON.stringify(props.scene.payload.ball.keyframes)
      : JSON.stringify(
        props.scene.payload.tracks.find((track) => track.id === props.selectedTrackId)?.keyframes ?? [],
      ),
    props.scene.payload.tracks.find((track) => track.id === props.selectedTrackId)?.color ?? '',
  ],
  rebuildSelectedPath,
)
watch(() => JSON.stringify(props.frameAnalysis), () => analysisMarkerLayer?.rebuild(props.frameAnalysis))
watch(() => [props.showModels, props.showLabels], () => {
  playerLayer?.setVisualOptions({
    showModels: props.showModels,
    showLabels: props.showLabels,
  })
})
watch(() => props.ballEditMode, (enabled) => {
  surface?.setBallEditCursor(enabled)
  updateObjects()
})
watch(
  () => [
    props.currentTime,
    props.selectedTrackId,
    props.selectedBallKeyframeTime,
    props.showBall,
    props.showTrails,
    props.showPathTracking,
  ],
  updateObjects,
)
watch(() => props.showAnalysisMarkers, (visible) => analysisMarkerLayer?.setVisible(visible))
watch(() => props.renderQuality, applyRenderQuality)

onBeforeUnmount(() => {
  pointerSelection?.dispose()
  selectionLayer?.dispose()
  selectedPathLayer?.dispose()
  analysisMarkerLayer?.dispose()
  ballLayer?.dispose()
  playerLayer?.dispose()
  pitchLayer?.dispose()
  surface?.dispose()
  pointerSelection = null
  selectionLayer = null
  selectedPathLayer = null
  analysisMarkerLayer = null
  ballLayer = null
  playerLayer = null
  pitchLayer = null
  surface = null
})

defineExpose({ cameraPreset })
</script>

<template>
  <div ref="host" class="three-host">
    <div class="viewport-hud"><span>{{ status }}</span></div>
    <div v-if="ballEditMode" class="edit-hint">
      Ball keypoint<span v-if="selectedBallKeyframeTime !== null"> · {{ selectedBallKeyframeTime.toFixed(2) }}s</span> — click the pitch to place it
    </div>
    <div v-else-if="editMode" class="edit-hint">Click the pitch to place the selected player</div>
    <PlayerActionPreviewHud v-if="activePlayerAction" :action="activePlayerAction" />
  </div>
</template>
