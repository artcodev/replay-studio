<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { ComputedRef, Ref, ShallowRef } from 'vue'
import ToolbarDisclosure from '../ToolbarDisclosure.vue'
import ThreeViewMenu from '../ThreeViewMenu.vue'
import CalibrationReviewModal from './CalibrationReviewModal.vue'
import type { useFrameAnalysis } from '../../composables/useFrameAnalysis'
import type { useManualBallEditor } from '../../composables/useManualBallEditor'
import type { useModelComparison } from '../../composables/useModelComparison'
import type { usePitchCalibrationEditor } from '../../composables/usePitchCalibrationEditor'
import type { useReconstructionController } from '../../composables/useReconstructionController'
import type { BallDetectionBackend, ReconstructionModel } from '../../types/reconstruction'
import type { MultiPassSummary } from '../../types/media'
import type { SceneDocument, SceneVideoAsset } from '../../types/scene'
import type { ThreeRenderQuality, ThreeViewOptions } from '../../lib/threeViewOptions'

type CameraName = 'broadcast' | 'orbit' | 'tactical' | 'goal'
type ViewMode = 'video' | 'split' | '3d'
type ActivePass = MultiPassSummary['passes'][number] | null

const props = defineProps<{
  view: {
    scene: ShallowRef<SceneDocument | null>
    sceneVideo: ComputedRef<SceneVideoAsset | null>
    activeCamera: Ref<CameraName>
    viewMode: Ref<ViewMode>
    viewOptions: Ref<ThreeViewOptions>
    renderQuality: Ref<ThreeRenderQuality>
    activeTab: Ref<'binding' | 'qa' | 'events'>
    multiPass: ComputedRef<MultiPassSummary | null>
    activePass: ComputedRef<ActivePass>
  }
  controllers: {
    reconstruction: ReturnType<typeof useReconstructionController>
    frameAnalysis: ReturnType<typeof useFrameAnalysis>
    manualBall: ReturnType<typeof useManualBallEditor>
    modelComparison: ReturnType<typeof useModelComparison>
    calibration: ReturnType<typeof usePitchCalibrationEditor>
  }
  models: {
    reconstruction: Array<{ value: ReconstructionModel; label: string }>
    ballDetection: Array<{ value: BallDetectionBackend; label: string }>
  }
  commands: {
    cameraPresetChange: (event: Event) => void
    chooseSourcePass: (event: Event) => void
    passRelationLabel: (relation?: string) => string
  }
}>()

const {
  scene,
  sceneVideo,
  activeCamera,
  viewMode,
  viewOptions,
  renderQuality,
  activeTab,
  multiPass: multiPassAnalysis,
  activePass,
} = props.view
const reconstruction = props.controllers.reconstruction
const frameAnalysis = props.controllers.frameAnalysis
const manualBall = props.controllers.manualBall
const modelComparison = props.controllers.modelComparison
const calibration = props.controllers.calibration

// The calibration stage lives entirely in the modal: it opens on "Calibrate"
// and stays open across the run (progress → review) until the stage ends (reset
// clears it, or the operator reconstructs).
const reviewModalOpen = ref(false)
const calibrationStageActive = computed(() => (
  reconstruction.stage.value === 'calibration'
  && (reconstruction.running.value || Boolean(reconstruction.calibrationReview.value))
))
watch(calibrationStageActive, (active) => {
  if (active) reviewModalOpen.value = true
})

function openCalibrationStage() {
  reviewModalOpen.value = true
}

function reconstructFromReview() {
  reviewModalOpen.value = false
  void reconstruction.reconstruct('full')
}

function regenerateSourceFrames() {
  const confirmed = window.confirm(
    `Regenerate analysis frames at the native source cadence (${reconstruction.sourceFrameRate.value.toFixed(3)} FPS)? This removes the old calibration, detections, tracks and identities because their frame generation changes. Calibration will not start automatically.`,
  )
  if (confirmed) void reconstruction.regenerateAnalysisFrames()
}
</script>

<template>
  <div class="stage-toolbar">
    <div class="stage-view-controls">
      <label class="toolbar-select camera-selector">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7.5h10.5v9H4zM14.5 10l5-2.5v9l-5-2.5" /></svg>
        <span>Camera</span>
        <select :value="activeCamera" aria-label="3D camera preset" :disabled="viewMode === 'video' || Boolean(calibration.draft.value)" @change="commands.cameraPresetChange">
          <option value="broadcast">Broadcast</option><option value="orbit">Orbit</option><option value="tactical">Tactical</option><option value="goal">Goal line</option>
        </select>
      </label>
      <label v-if="sceneVideo" class="toolbar-select layout-selector">
        <span>Layout</span>
        <select v-model="viewMode" aria-label="Workspace layout" :disabled="Boolean(calibration.draft.value)">
          <option value="split">Video + 3D</option><option value="3d">3D only</option><option value="video">Video only</option>
        </select>
      </label>
    </div>
    <div class="stage-tools">
      <label v-if="multiPassAnalysis?.status === 'ready' && activePass" class="angle-switcher">
        <span>Source</span>
        <select :value="activePass.sceneId" aria-label="Replay angle" @change="commands.chooseSourcePass">
          <option v-for="item in multiPassAnalysis.passes.filter((pass) => pass.status === 'ready')" :key="item.sceneId" :value="item.sceneId">{{ item.label }} · {{ commands.passRelationLabel(item.relation) }}</option>
        </select>
      </label>
      <span v-if="multiPassAnalysis" class="multi-pass-badge">{{ multiPassAnalysis.status }} · {{ multiPassAnalysis.selectedSegmentIds.length }} angles</span>
      <button
        v-if="sceneVideo?.selectedSegmentId"
        class="tool-toggle frame-analysis-toggle primary-tool"
        :class="{ active: Boolean(frameAnalysis.activeAnalysis.value) }"
        :disabled="frameAnalysis.analyzing.value || reconstruction.reconstructing.value || reconstruction.status.value === 'processing' || reconstruction.status.value === 'queued'"
        @click="frameAnalysis.analyze"
      >{{ frameAnalysis.analyzing.value ? 'Reading frame…' : 'Analyze frame' }}</button>
      <button
        v-if="sceneVideo?.selectedSegmentId"
        class="tool-toggle calibration-frame-toggle"
        :class="{ active: reviewModalOpen }"
        :disabled="(reconstruction.reconstructing.value || reconstruction.running.value) && reconstruction.stage.value !== 'calibration'"
        title="Open the calibration workspace with preview, timeline and frame controls."
        @click="openCalibrationStage"
      >Calibration</button>
      <ToolbarDisclosure
        v-if="sceneVideo?.selectedSegmentId && !multiPassAnalysis"
        label="Reconstruction"
        :active="reconstruction.running.value || activeTab === 'qa' || Boolean(calibration.draft.value)"
      >
        <template #icon><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h14v14H5zM8 8h8v8H8zM3 9h2m14 0h2M3 15h2m14 0h2M9 3v2m6-2v2M9 19v2m6-2v2" /></svg></template>
        <template #default="{ closeMenu }">
          <div class="reconstruction-menu-content">
            <div class="reconstruction-menu-heading">
              <div><span>Scene reconstruction</span><strong>{{ scene?.payload.tracks.length ?? 0 }} tracked objects</strong></div>
              <b :class="`verdict-${reconstruction.qualityVerdict.value}`">{{ reconstruction.running.value ? `${reconstruction.progress.value?.overallPercent ?? 0}%` : reconstruction.qualityVerdict.value.toUpperCase() }}</b>
            </div>
            <div v-if="reconstruction.inputState.value === 'stale'" class="reconstruction-input-warning" role="status"><strong>Inputs changed</strong><span>Match data or a reviewed scene setting changed after this result was built. The current 3D output is preserved until you explicitly rebuild it.</span></div>
            <div v-else-if="reconstruction.resultState.value === 'calibration-only'" class="reconstruction-input-warning" role="status"><strong>Reconstruction invalidated</strong><span>The current calibration is ready for inspection, but tracks from the previous calibration were removed. Complete step 2 to build identities and 3D trajectories from this exact calibration artifact.</span></div>
            <label class="reconstruction-model-field"><span>People detector</span><select v-model="reconstruction.selectedModel.value" aria-label="Reconstruction model" :disabled="reconstruction.reconstructing.value || reconstruction.running.value"><option v-for="item in models.reconstruction" :key="item.value" :value="item.value">{{ item.label }}</option></select></label>
            <label class="reconstruction-model-field"><span>Ball detector</span><select v-model="reconstruction.selectedBallBackend.value" aria-label="Ball detection backend" :disabled="reconstruction.reconstructing.value || reconstruction.running.value"><option v-for="item in models.ballDetection" :key="item.value" :value="item.value">{{ item.label }}</option></select></label>
            <label class="reconstruction-model-field"><span>Ball trajectory</span><select :value="manualBall.mode.value" aria-label="Ball trajectory mode" :disabled="manualBall.saving.value || reconstruction.reconstructing.value || reconstruction.running.value" @change="manualBall.changeMode"><option value="automatic">Automatic detection</option><option value="manual">Manual keypoints</option></select></label>
            <label v-if="reconstruction.manualBallAuthoritative.value" class="reconstruction-model-field reconstruction-skip-ball"><span>Ball detection</span><span class="reconstruction-skip-ball-control"><input v-model="reconstruction.skipBallDetection.value" type="checkbox" aria-label="Skip automatic ball detection" :disabled="reconstruction.reconstructing.value || reconstruction.running.value" /> Skip dense detection — the manual trajectory stays authoritative and the run is much faster</span></label>
            <label class="reconstruction-model-field reconstruction-skip-ball"><span>Jersey OCR</span><span class="reconstruction-skip-ball-control"><input v-model="reconstruction.skipJerseyOcr.value" type="checkbox" aria-label="Skip jersey number OCR" :disabled="reconstruction.reconstructing.value || reconstruction.running.value" /> Skip shirt-number OCR — faster run, but fewer automatic track merges; bind the roster manually</span></label>
            <label class="reconstruction-model-field reconstruction-skip-ball"><span>Contact point</span><span class="reconstruction-skip-ball-control"><input v-model="reconstruction.poseContactPoint.value" type="checkbox" aria-label="Use pose feet as the ground contact point" :disabled="reconstruction.reconstructing.value || reconstruction.running.value" /> Pose feet (experimental) — project RTMPose feet instead of the bbox bottom; crops that fail the pose gate keep the bbox point</span></label>
            <div class="reconstruction-stage-buttons">
              <button class="reconstruction-menu-primary reconstruction-stage-calibrate" :disabled="(reconstruction.reconstructing.value || reconstruction.running.value) && reconstruction.stage.value !== 'calibration'" @click="closeMenu(); openCalibrationStage()">{{ reconstruction.running.value && reconstruction.stage.value === 'calibration' ? `1 · Calibrating · ${reconstruction.progress.value?.overallPercent ?? 0}%` : '1 · Calibrate scene' }}</button>
              <button class="reconstruction-menu-primary" :disabled="reconstruction.reconstructing.value || reconstruction.running.value || !reconstruction.calibrationCleared.value" :title="reconstruction.reviewPending.value ? 'Fix calibration or explicitly authorize image fallback before reconstructing' : !reconstruction.calibrationCleared.value ? 'Calibrate the scene before reconstructing' : undefined" @click="closeMenu(); reconstruction.reconstruct('full')">{{ reconstruction.running.value && reconstruction.stage.value !== 'calibration' ? `Analyzing · ${reconstruction.progress.value?.overallPercent ?? 0}%` : reconstruction.inputState.value === 'stale' ? 'Rebuild with current inputs' : scene?.payload.tracks.length ? '2 · Reconstruct scene' : '2 · Build scene' }}</button>
            </div>
            <button v-if="reconstruction.reviewPending.value" class="reconstruction-review-hint" @click="closeMenu(); reviewModalOpen = true"><strong>Calibration review pending</strong><span>{{ reconstruction.calibrationReview.value?.unresolvedFrames }} of {{ reconstruction.calibrationReview.value?.totalFrames }} frames need attention — inspect before reconstructing</span></button>
            <div class="reconstruction-menu-actions">
              <button v-if="reconstruction.activeRunId.value && reconstruction.running.value" class="cancel-reconstruction-action" :disabled="!reconstruction.canCancel.value || reconstruction.cancelling.value" @click="closeMenu(); reconstruction.cancelActive()"><span>{{ reconstruction.cancelling.value ? 'Cancelling analysis…' : 'Cancel current analysis' }}</span><small>The current run will stop at its next safe checkpoint</small></button>
              <button :class="{ active: Boolean(modelComparison.report.value) }" :disabled="modelComparison.queueing.value || modelComparison.running.value || reconstruction.reconstructing.value || reconstruction.running.value" @click="closeMenu(); modelComparison.compare()"><span>Compare detection models</span><small>{{ modelComparison.queueing.value ? 'Queueing comparison…' : modelComparison.running.value ? modelComparison.job.value?.progress.label || 'Comparing 26n and 26m…' : modelComparison.report.value ? 'Result ready' : '26n versus 26m' }}</small></button>
              <button :class="{ active: reviewModalOpen }" :disabled="reconstruction.running.value && reconstruction.stage.value !== 'calibration'" @click="closeMenu(); openCalibrationStage()"><span>Open calibration workspace</span><small>Preview, timeline, full run and per-frame corrections</small></button>
              <button v-if="reconstruction.stage.value === 'calibration' && reconstruction.calibrationReview.value" :class="{ active: reconstruction.reviewPending.value }" @click="closeMenu(); reviewModalOpen = true"><span>Open calibration review</span><small>{{ reconstruction.calibrationReview.value.resolvedFrames }}/{{ reconstruction.calibrationReview.value.totalFrames }} frames · {{ reconstruction.calibrationReview.value.status }}</small></button>
              <button v-if="sceneVideo?.reconstruction" @click="closeMenu(); activeTab = 'qa'"><span>Open calibration quality</span><small>View {{ calibration.visiblePitchSide.value.toUpperCase() }} · attack {{ calibration.attackingGoalSide.value.toUpperCase() }}</small></button>
            </div>
          </div>
        </template>
      </ToolbarDisclosure>
      <ThreeViewMenu v-model="viewOptions" v-model:render-quality="renderQuality" :disabled="Boolean(calibration.draft.value)" />
    </div>
    <CalibrationReviewModal
      :open="reviewModalOpen"
      :project-id="reconstruction.projectId()"
      :scene-id="scene?.id ?? null"
      :media-url="sceneVideo?.mediaUrl ?? null"
      :generation-key="sceneVideo?.generationKey ?? null"
      :frame-exclusions="sceneVideo?.frameExclusions ?? []"
      :source-start="sceneVideo?.sourceStart ?? 0"
      :duration="scene?.duration ?? 0"
      :source-fps="reconstruction.sourceFrameRate.value"
      :materialized-fps="reconstruction.materializedFrameRate.value"
      :frame-rate="reconstruction.selectedFrameRate.value"
      :direct-calibration-max-gap-seconds="reconstruction.selectedDirectCalibrationMaxGapSeconds.value"
      :frame-rate-requires-regeneration="reconstruction.frameRateRequiresRegeneration.value"
      :calibration-frame-rate-current="reconstruction.calibrationFrameRateCurrent.value"
      :calibration-direct-sampling-current="reconstruction.calibrationDirectSamplingCurrent.value"
      :calibration-review-current="reconstruction.calibrationReviewCurrent.value"
      :review="reconstruction.calibrationReview.value"
      :frame-input-ready="reconstruction.sourceFrameInputReady.value"
      :regenerating-frames="reconstruction.regeneratingFrames.value"
      :frame-generation-progress="reconstruction.frameGenerationJob.value ? {
        label: reconstruction.frameGenerationJob.value.progress.label,
        detail: reconstruction.frameGenerationJob.value.progress.detail,
        percent: reconstruction.frameGenerationJob.value.progress.percent,
      } : null"
      :busy="reconstruction.savingFrame.value"
      :updating-frame-exclusion="reconstruction.updatingFrameExclusion.value"
      :finalizing="reconstruction.finalizingCalibrationEdits.value"
      :calibration-trigger="scene?.payload.videoAsset?.reconstruction?.calibrationTrigger ?? null"
      :pending-edit-sample-indices="reconstruction.pendingCalibrationEditSession.value?.editedSampleIndices ?? []"
      :running="reconstruction.running.value"
      :confirming="reconstruction.confirmingReview.value"
      :resetting="reconstruction.resettingCalibration.value"
      :can-reconstruct="reconstruction.calibrationCleared.value"
      :progress="reconstruction.progress.value"
      :can-cancel="reconstruction.canCancel.value"
      :cancelling="reconstruction.cancelling.value"
      :save-frame="reconstruction.saveFrameCalibration"
      @close="reviewModalOpen = false"
      @confirm="reconstruction.confirmReview()"
      @reconstruct="reconstructFromReview()"
      @reset="reconstruction.resetCalibration()"
      @rerun="reconstruction.calibrate()"
      @finalize="reconstruction.finalizeCalibrationEdits()"
      @set-frame-excluded="reconstruction.setFrameExcluded($event.sourceFrameIndex, $event.excluded)"
      @regenerate-frames="regenerateSourceFrames"
      @update:frame-rate="reconstruction.selectedFrameRate.value = $event"
      @update:direct-calibration-max-gap-seconds="reconstruction.selectedDirectCalibrationMaxGapSeconds.value = $event"
      @cancel="reconstruction.cancelActive()"
    />
  </div>
</template>
