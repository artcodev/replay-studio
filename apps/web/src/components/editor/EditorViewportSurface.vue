<script setup lang="ts">
import { computed, type ComputedRef, type Ref, type ShallowRef } from 'vue'
import PathTrackingLegend from '../PathTrackingLegend.vue'
import VideoPathTrackingOverlay from '../VideoPathTrackingOverlay.vue'
import FrameDetectionOverlay from './FrameDetectionOverlay.vue'
import PitchCalibrationOverlay from './PitchCalibrationOverlay.vue'
import PitchCalibrationPanel from './PitchCalibrationPanel.vue'
import ReconstructionProgressPanel from './ReconstructionProgressPanel.vue'
import SelectedTrackProjectionOverlay from './SelectedTrackProjectionOverlay.vue'
import ThreeScenePane from './ThreeScenePane.vue'
import VideoReviewPane from './VideoReviewPane.vue'
import VideoOverlayMenu from '../VideoOverlayMenu.vue'
import type { VideoOverlayOptions } from '../../lib/videoOverlayOptions'
import VideoSelectionStatus from './VideoSelectionStatus.vue'
import { VIDEO_REVIEW_MAX_SCALE, VIDEO_REVIEW_MIN_SCALE } from '../../lib/videoReviewTransform'
import type { PathProjectionContext } from '../../lib/pathProjection'
import type { PathTrackingSegment, PathTrackingSubjectKind } from '../../lib/pathTracking'
import type { useFrameAnalysis } from '../../composables/useFrameAnalysis'
import type { useFrameAnnotations } from '../../composables/useFrameAnnotations'
import type { useManualBallEditor } from '../../composables/useManualBallEditor'
import type { usePitchCalibrationEditor } from '../../composables/usePitchCalibrationEditor'
import type { usePlayerActionEditor } from '../../composables/usePlayerActionEditor'
import type { useReconstructionController } from '../../composables/useReconstructionController'
import type { useVideoReviewViewport } from '../../composables/useVideoReviewViewport'
import type { FrameAnalysis } from '../../types/analysis'
import type { CalibrationFrameEvidence } from '../../types/calibration'
import type { CanonicalPerson } from '../../types/identity'
import type { SceneDocument, SceneVideoAsset } from '../../types/scene'
import type { Keyframe, Track } from '../../types/tracking'
import type { ThreeRenderQuality, ThreeViewOptions } from '../../lib/threeViewOptions'
import { projectPitchMarkings } from '../../lib/pitchProjection'

type ViewMode = 'video' | 'split' | '3d'
type ViewportApi = { cameraPreset: (name: 'broadcast' | 'orbit' | 'tactical' | 'goal') => void }
type PathSubject = {
  kind: PathTrackingSubjectKind
  label: string
  color: string
  sampleCount: number
} | null

const props = defineProps<{
  playback: {
    scene: ShallowRef<SceneDocument | null>
    sceneVideo: ComputedRef<SceneVideoAsset | null>
    currentTime: Ref<number>
    sourceVideo: Ref<HTMLVideoElement | null>
    seekTo: (time: number) => void
  }
  view: {
    mode: Ref<ViewMode>
    options: Ref<ThreeViewOptions>
    videoOverlayOptions: Ref<VideoOverlayOptions>
    renderQuality: Ref<ThreeRenderQuality>
    activeTab: Ref<'binding' | 'qa' | 'events'>
    viewport: Ref<ViewportApi | null>
  }
  selection: {
    selectedTrack: ComputedRef<Track | null>
    selectedCanonicalPerson: ComputedRef<CanonicalPerson | null>
    selectedTrackId: Ref<string | null>
    selectedFramePersonId: Ref<string | null>
    editMode: Ref<boolean>
    pathSubject: ComputedRef<PathSubject>
    pathKeyframes: ComputedRef<Keyframe[]>
    pathSegments: ComputedRef<PathTrackingSegment[]>
    unavailablePathSubjectLabel: ComputedRef<string | null>
    reconstructionPreviewScene: ComputedRef<SceneDocument | null>
  }
  analysis: {
    frameCount: ComputedRef<number>
    calibrationFrames: ComputedRef<CalibrationFrameEvidence[]>
    videoPathUsesReferenceCamera: ComputedRef<boolean>
    videoPathProjectionContext: ComputedRef<PathProjectionContext | null>
    videoPathUnavailableReason: ComputedRef<string | null>
    videoPathSurfaceNote: ComputedRef<string | null>
    calibrationLabel: ComputedRef<string>
    framePersonSelectionDescription: (person: FrameAnalysis['people'][number]) => string
  }
  reconstruction: ReturnType<typeof useReconstructionController>
  videoReview: ReturnType<typeof useVideoReviewViewport>
  calibration: ReturnType<typeof usePitchCalibrationEditor>
  frameAnalysis: ReturnType<typeof useFrameAnalysis>
  frameAnnotations: ReturnType<typeof useFrameAnnotations>
  manualBall: ReturnType<typeof useManualBallEditor>
  playerActions: ReturnType<typeof usePlayerActionEditor>
  moveSelected: (position: { x: number; z: number }) => void
}>()

const scene = computed(() => {
  const document = props.playback.scene.value
  if (!document) throw new Error('Editor viewport requires an active scene.')
  return document
})

const referenceCaption = computed(() => {
  const reconstruction = props.reconstruction
  // The calibration stage reports its progress inside its own modal, not here.
  if (reconstruction.running.value && reconstruction.stage.value !== 'calibration') {
    return `Original clip · ${reconstruction.progress.value?.overallPercent ?? 0}% · ${reconstruction.progress.value?.label ?? 'STARTING ANALYSIS'}`
  }
  return scene.value.payload.tracks.length
    ? `Original clip · AUTO ${scene.value.payload.tracks.length} · ${props.analysis.calibrationLabel.value}`
    : 'Original clip'
})

const storedCalibrationFrame = computed<CalibrationFrameEvidence | null>(() => {
  const frames = props.analysis.calibrationFrames.value
  if (!frames.length) return null
  return frames.reduce((nearest, frame) => (
    Math.abs(frame.sceneTime - props.playback.currentTime.value)
      < Math.abs(nearest.sceneTime - props.playback.currentTime.value)
      ? frame
      : nearest
  ))
})
const storedCalibrationFrameSize = computed(() => ({
  width: storedCalibrationFrame.value?.frameWidth
    ?? props.playback.sourceVideo.value?.videoWidth
    ?? 960,
  height: storedCalibrationFrame.value?.frameHeight
    ?? props.playback.sourceVideo.value?.videoHeight
    ?? 540,
}))
const storedCalibrationMarkings = computed(() => {
  const frame = storedCalibrationFrame.value
  if (!frame) return []
  if (frame.markings?.length) return frame.markings
  return projectPitchMarkings(
    frame.imageToPitch,
    storedCalibrationFrameSize.value.width,
    storedCalibrationFrameSize.value.height,
  )
})
const projectionDebugObservations = computed(() => (
  props.selection.selectedTrack.value?.observations
  ?? props.selection.selectedCanonicalPerson.value?.observations
  ?? []
))
const projectionDebugLabel = computed(() => (
  props.selection.selectedTrack.value?.label
  ?? props.selection.selectedCanonicalPerson.value?.displayName
  ?? null
))
const contactPointProfile = computed(() => (
  scene.value.payload.videoAsset?.reconstruction?.contactPointProfile ?? 'bbox-bottom'
))

function bindVideoReviewViewport(element: HTMLDivElement | null) {
  props.videoReview.viewport.value = element
}

function bindSourceVideo(element: HTMLVideoElement | null) {
  props.playback.sourceVideo.value = element
}

function bindCalibrationOverlay(element: SVGSVGElement | null) {
  props.calibration.overlay.value = element
}

function bindFrameAnalysisOverlay(element: SVGSVGElement | null) {
  props.frameAnnotations.overlay.value = element
}

function bindThreeViewport(value: ViewportApi | null) {
  props.view.viewport.value = value
}
</script>

<template>
  <div
    class="viewport-wrap"
    :class="{
      'split-view': view.mode.value === 'split' && playback.sceneVideo.value && !calibration.draft.value,
      'video-only': view.mode.value === 'video' && playback.sceneVideo.value && !calibration.draft.value,
      'three-only': (view.mode.value === '3d' || !playback.sceneVideo.value) && !calibration.draft.value,
      calibrating: Boolean(calibration.draft.value),
    }"
  >
    <ReconstructionProgressPanel
      v-if="reconstruction.running.value && reconstruction.stage.value !== 'calibration'"
      :progress="reconstruction.progress.value"
      :phases="reconstruction.phases.value"
      :frame-count="analysis.frameCount.value"
      :active-run-id="reconstruction.activeRunId.value"
      :can-cancel="reconstruction.canCancel.value"
      :cancelling="reconstruction.cancelling.value"
      @cancel="reconstruction.cancelActive"
    />

    <VideoReviewPane
      v-if="playback.sceneVideo.value && (view.mode.value !== '3d' || calibration.draft.value)"
      :asset="playback.sceneVideo.value"
      :transform="videoReview.transform.value"
      :transform-style="videoReview.style.value"
      :zoom-percent="videoReview.zoomPercent.value"
      :panning="videoReview.panning.value"
      :min-scale="VIDEO_REVIEW_MIN_SCALE"
      :max-scale="VIDEO_REVIEW_MAX_SCALE"
      :caption="referenceCaption"
      @viewport-element="bindVideoReviewViewport"
      @video-element="bindSourceVideo"
      @loaded-metadata="playback.seekTo(playback.currentTime.value)"
      @wheel="videoReview.onWheel"
      @pointer-down="videoReview.startPan"
      @pointer-move="videoReview.updatePan"
      @pointer-up="videoReview.finishPan"
      @pointer-cancel="videoReview.finishPan"
      @keydown="videoReview.onKeydown"
      @adjust-zoom="videoReview.adjustZoom"
      @reset="videoReview.reset"
    >
      <VideoPathTrackingOverlay
        :enabled="view.options.value.pathTracking && !calibration.draft.value && view.activeTab.value !== 'qa' && analysis.videoPathUsesReferenceCamera.value"
        :keyframes="selection.pathKeyframes.value"
        :projection-context="analysis.videoPathProjectionContext.value"
        :current-time="playback.currentTime.value"
        :subject-kind="selection.pathSubject.value?.kind"
        :color="selection.pathSubject.value?.color"
        :subject-label="selection.pathSubject.value?.label"
      />
      <PitchCalibrationOverlay
        v-if="view.videoOverlayOptions.value.pitchCalibration || calibration.draft.value"
        :draft="calibration.activeDraft.value"
        :diagnostics="calibration.diagnostics.value"
        :qa-frame="calibration.activeQaFrame.value ?? storedCalibrationFrame"
        :qa-frame-size="calibration.activeQaFrame.value ? calibration.qaFrameSize.value : storedCalibrationFrameSize"
        :qa-markings="calibration.activeQaFrame.value ? calibration.qaMarkings.value : storedCalibrationMarkings"
        @overlay-element="bindCalibrationOverlay"
        @update-drag="calibration.updateDraggedAnchor"
        @finish-drag="calibration.finishAnchorDrag"
        @start-anchor-drag="calibration.startAnchorDrag"
        @nudge-anchor="calibration.nudgeAnchor"
      />
      <SelectedTrackProjectionOverlay
        :enabled="view.videoOverlayOptions.value.projectionDebug"
        :label="projectionDebugLabel"
        :observations="projectionDebugObservations"
        :calibration-frames="analysis.calibrationFrames.value"
        :pitch="scene.payload.pitch"
        :current-time="playback.currentTime.value"
        :frame-size="storedCalibrationFrameSize"
        :contact-point-profile="contactPointProfile"
      />
      <FrameDetectionOverlay
        v-if="frameAnalysis.activeAnalysis.value && !calibration.draft.value && view.activeTab.value !== 'qa'"
        :analysis="frameAnalysis.activeAnalysis.value"
        :selected-person-id="selection.selectedFramePersonId.value"
        :labeling="frameAnnotations.mode.value"
        :draft="frameAnnotations.draft.value"
        :canonical-id="frameAnalysis.framePersonCanonicalId"
        :person-label="frameAnalysis.framePersonLabel"
        :selection-description="analysis.framePersonSelectionDescription"
        :overlay-options="view.videoOverlayOptions.value"
        @overlay-element="bindFrameAnalysisOverlay"
        @start-drag="frameAnnotations.startDrag"
        @update-drag="frameAnnotations.updateDrag"
        @finish-drag="frameAnnotations.finishDrag"
        @select-at-point="frameAnnotations.selectAtPoint"
        @select-person="frameAnnotations.selectDetectedPerson"
        @select-annotation="frameAnnotations.selectAnnotation"
      />

      <template #floating>
        <VideoSelectionStatus
          v-if="frameAnalysis.selectionStatus.value && !calibration.draft.value && view.activeTab.value !== 'qa'"
          :status="frameAnalysis.selectionStatus.value"
        />
        <PathTrackingLegend
          v-if="!calibration.draft.value"
          :enabled="view.options.value.pathTracking"
          :subject-kind="selection.pathSubject.value?.kind"
          :subject-label="selection.pathSubject.value?.label"
          :subject-color="selection.pathSubject.value?.color"
          :sample-count="selection.pathSubject.value?.sampleCount"
          :has-drawable-path="selection.pathSegments.value.length > 0"
          :unavailable-label="selection.unavailablePathSubjectLabel.value"
          :surface-unavailable-reason="analysis.videoPathUnavailableReason.value"
          :surface-note="analysis.videoPathSurfaceNote.value"
          align="left"
          top-offset="stacked"
          surface-label="video review"
        />
        <PitchCalibrationPanel
          v-if="calibration.draft.value"
          :draft="calibration.draft.value"
          :active-at-current-time="Boolean(calibration.activeDraft.value)"
          :diagnostics="calibration.diagnostics.value"
          :warnings="calibration.warnings.value"
          :preset="calibration.preset.value"
          :presets="calibration.presets"
          :loading="calibration.loading.value"
          :applying="calibration.applying.value"
          @change-preset="calibration.changePreset"
          @calibrate-again="calibration.open(calibration.preset.value)"
          @return-to-frame="playback.seekTo(calibration.draft.value!.sceneTime)"
          @apply="calibration.apply"
        />
      </template>
      <template #overlay-controls>
        <VideoOverlayMenu
          v-model="view.videoOverlayOptions.value"
          :disabled="Boolean(calibration.draft.value)"
        />
      </template>
    </VideoReviewPane>

    <ThreeScenePane
      v-show="view.mode.value !== 'video'"
      :scene="selection.reconstructionPreviewScene.value ?? scene"
      :current-time="playback.currentTime.value"
      :selected-track-id="selection.selectedTrackId.value"
      :edit-mode="selection.editMode.value"
      :ball-edit-mode="manualBall.mode.value === 'manual' && manualBall.placementMode.value"
      :selected-ball-keyframe-time="manualBall.selectedKeyframeTime.value"
      :ball-selected="manualBall.selected.value"
      :options="view.options.value"
      :render-quality="view.renderQuality.value"
      :frame-analysis="frameAnalysis.activeAnalysis.value"
      :active-player-action="selection.selectedTrack.value ? playerActions.activePlayback.value : null"
      :path-subject="selection.pathSubject.value"
      :has-drawable-path="selection.pathSegments.value.length > 0"
      :unavailable-path-label="selection.unavailablePathSubjectLabel.value"
      @viewport-element="bindThreeViewport"
      @select-track="frameAnalysis.selectTrackFromThree"
      @select-ball="manualBall.selectBall"
      @move-track="moveSelected"
      @move-ball="manualBall.move"
    />
  </div>
</template>
