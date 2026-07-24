<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import PitchCalibrationOverlay from './PitchCalibrationOverlay.vue'
import { useCalibrationReview } from '../../composables/useCalibrationReview'
import { projectPitchMarkings } from '../../lib/pitchProjection'
import {
  calibrationFramesWithExclusions,
  calibrationReviewTimelineStatus,
} from '../../features/calibration/calibrationQaPresentation'
import { calibrationWorkflowSteps } from '../../features/calibration/calibrationProgressPresentation'
import { sceneClient } from '../../lib/api/scenes'
import type { CalibrationReview, CalibrationReviewSample, ReconstructionProgress } from '../../types/reconstruction'
import type { SceneFrameExclusion } from '../../types/scene'
import type {
  CalibrationDraftSource,
  PitchCalibrationAnchor,
  PitchCalibrationPreset,
} from '../../types/calibration'

const props = defineProps<{
  open: boolean
  projectId: string
  sceneId: string | null
  mediaUrl: string | null
  generationKey: string | null
  frameExclusions: SceneFrameExclusion[]
  sourceStart: number
  duration: number
  sourceFps: number
  materializedFps: number
  frameRate: number
  directCalibrationMaxGapSeconds: number
  frameRateRequiresRegeneration: boolean
  calibrationFrameRateCurrent: boolean
  calibrationDirectSamplingCurrent: boolean
  calibrationReviewCurrent: boolean
  review: CalibrationReview | null
  frameInputReady: boolean
  regeneratingFrames: boolean
  frameGenerationProgress: {
    label: string
    detail: string | null
    percent: number
  } | null
  /** A manual frame draft is being persisted. */
  busy: boolean
  updatingFrameExclusion: boolean
  finalizing: boolean
  calibrationTrigger: 'full-request' | 'manual-draft-finalize' | null
  pendingEditSampleIndices: number[]
  /** A calibrate job is running — the modal shows progress instead of the gate. */
  running: boolean
  confirming: boolean
  resetting: boolean
  /** The gate is cleared (100% resolved or the operator accepted the gaps). */
  canReconstruct: boolean
  progress: ReconstructionProgress | null
  canCancel: boolean
  cancelling: boolean
  saveFrame: (
    sceneTime: number,
    preset: PitchCalibrationPreset,
    anchors: PitchCalibrationAnchor[],
    source: CalibrationDraftSource,
    acceptQualityWarning?: boolean,
  ) => Promise<void>
}>()

const emit = defineEmits<{
  close: []
  confirm: []
  reconstruct: []
  reset: []
  rerun: []
  cancel: []
  regenerateFrames: []
  finalize: []
  'update:frameRate': [value: number]
  'update:directCalibrationMaxGapSeconds': [value: number]
  setFrameExcluded: [value: { sourceFrameIndex: number; excluded: boolean }]
}>()

const localError = ref<string | null>(null)
const exactFrameLoadError = ref(false)

const cal = useCalibrationReview({
  projectId: () => props.projectId,
  sceneId: () => props.sceneId,
  error: localError,
  saveFrame: props.saveFrame,
})

const allFrames = computed<CalibrationReviewSample[]>(() => (
  calibrationFramesWithExclusions(
    props.review?.frames ?? [],
    props.frameExclusions,
  )
))
const includedFrames = computed(() => allFrames.value.filter((frame) => !frame.excluded))
const unresolvedFrames = computed(() => (
  includedFrames.value.filter((frame) => !frame.resolved)
))
const excludedFrames = computed(() => allFrames.value.filter((frame) => frame.excluded))
const attentionFrames = computed(() => (
  allFrames.value.filter((frame) => frame.excluded || !frame.resolved)
))
const showAll = ref(false)
const listFrames = computed(() => (showAll.value ? allFrames.value : attentionFrames.value))
const includedResolvedCount = computed(() => (
  includedFrames.value.filter((frame) => frame.resolved).length
))
const ratioPercent = computed(() => Math.round(
  includedFrames.value.length
    ? includedResolvedCount.value / includedFrames.value.length * 100
    : 100,
))
const selectedFps = computed(() => (
  props.frameRate > 0 ? props.frameRate : props.sourceFps
))
const estimatedFrameCount = computed(() => (
  Math.max(1, Math.floor(props.duration * selectedFps.value) + 1)
))
const estimatedDirectFrameCount = computed(() => (
  props.directCalibrationMaxGapSeconds <= 0
    ? estimatedFrameCount.value
    : Math.min(
      estimatedFrameCount.value,
      Math.ceil(props.duration / props.directCalibrationMaxGapSeconds) + 1,
    )
))
const directCalibrationGapOptions = [0, 0.1, 0.25, 0.5, 1]
const frameRateOptions = computed(() => {
  const maximumReduced = Math.min(props.sourceFps, props.materializedFps)
  return [60, 50, 30, 25, 20, 15, 10, 5]
    .filter((fps) => fps < props.sourceFps - 1e-3 && fps <= maximumReduced + 1e-3)
})
const isReviewing = computed(() => (
  props.calibrationReviewCurrent && props.review?.status === 'review'
))
const editingLocked = computed(() => (
  props.busy
  || props.finalizing
  || props.resetting
  || props.updatingFrameExclusion
  || cal.saving.value
))
const stagedSampleIndices = computed(() => new Set(props.pendingEditSampleIndices))
const stagedCount = computed(() => props.pendingEditSampleIndices.length)
const nextStepReason = computed(() => {
  if (stagedCount.value) {
    return `${stagedCount.value} frame correction(s) are staged. Finalize them before Reconstruction.`
  }
  if (!props.review) return 'Run full calibration before Reconstruction.'
  if (!props.calibrationReviewCurrent) {
    return 'This timeline belongs to an earlier calibration input. The latest calibration did not publish a replacement; run full recalibration before Reconstruction.'
  }
  if (props.review.status === 'review') {
    return `${props.review.unresolvedFrames} frame(s) still have no accepted calibration. Fix them or explicitly authorize image fallback.`
  }
  if (!props.calibrationFrameRateCurrent) {
    return 'The selected FPS differs from the published calibration. Run full recalibration before Reconstruction.'
  }
  if (!props.calibrationDirectSamplingCurrent) {
    return 'The direct PnLCalib sampling differs from the published calibration. Run full recalibration before Reconstruction.'
  }
  if (!props.canReconstruct) return 'The calibration artifact is missing or stale for the current scene inputs.'
  return null
})

// While a calibrate job runs, show progress inside the modal instead of the gate.
const showProgress = computed(() => props.running || props.regeneratingFrames)
const visibleProgress = computed(() => (
  props.regeneratingFrames
    ? {
        label: props.frameGenerationProgress?.label ?? 'Extracting source-resolution frames…',
        detail: props.frameGenerationProgress?.detail ?? 'The published generation stays active until the new immutable frame set is complete.',
        percent: props.frameGenerationProgress?.percent ?? 0,
      }
    : {
        label: props.progress?.label ?? 'Calibrating the pitch…',
        detail: props.progress?.detail ?? 'Only pitch calibration runs at this stage — no detection, crops or tracking.',
        percent: props.progress?.overallPercent ?? 0,
      }
))
const workflowSteps = computed(() => calibrationWorkflowSteps(
  props.progress,
  props.calibrationTrigger,
))

function stripPercent(index: number, count: number): number {
  if (count <= 1) return 0
  return (index / (count - 1)) * 100
}

// The frame currently under inspection, tracked by sample index so it re-points
// to the fresh frame object whenever the gate updates (e.g. after a save).
function frameKey(frame: CalibrationReviewSample): string {
  return frame.sourceFrameIndex != null
    ? `source:${frame.sourceFrameIndex}`
    : `sample:${frame.sampleIndex}`
}

const selectedFrameKey = ref<string | null>(null)
const selectedFrame = computed<CalibrationReviewSample | null>(() => (
  allFrames.value.find((frame) => frameKey(frame) === selectedFrameKey.value) ?? null
))
const isEditing = computed(() => cal.draft.value != null)
const manualQaWarning = computed(() => cal.draft.value?.quality === 'poor')
const manualQaDetail = computed(() => {
  const metrics = cal.draft.value?.alignmentMetrics
  if (!metrics) return 'No reliable line-mask score is available.'
  return [
    `precision ${(metrics.precision * 100).toFixed(0)}%`,
    `recall ${(metrics.recall * 100).toFixed(0)}%`,
    `F1 ${(metrics.f1 * 100).toFixed(0)}%`,
    `p95 ${metrics.residualP95.toFixed(1)}px`,
  ].join(' · ')
})
// Read-only inspection overlay: the ACTUAL projected pitch lines of the selected
// frame's stored calibration. Shown until the operator chooses to recalibrate.
const inspectionSize = computed(() => ({
  width: selectedFrame.value?.frameWidth ?? 960,
  height: selectedFrame.value?.frameHeight ?? 540,
}))
const inspectionMarkings = computed(() => {
  const frame = selectedFrame.value
  if (
    !frame
    || frame.excluded
    || isEditing.value
    || !frame.resolved
    || !frame.imageToPitch
  ) return []
  return projectPitchMarkings(frame.imageToPitch, inspectionSize.value.width, inspectionSize.value.height)
})

const exactFrameUrl = computed(() => {
  const sourceFrameIndex = selectedFrame.value?.sourceFrameIndex
  if (
    !props.sceneId
    || !props.generationKey
    || sourceFrameIndex == null
  ) return null
  return sceneClient.exactAnalysisFrameUrl(
    props.projectId,
    props.sceneId,
    props.generationKey,
    sourceFrameIndex,
  )
})
watch(exactFrameUrl, () => { exactFrameLoadError.value = false })

// Selecting a frame is inspection only — no network, shows its stored overlay.
function selectFrame(frame: CalibrationReviewSample): void {
  if (frame.sceneTime == null) return
  localError.value = null
  cal.cancel()
  selectedFrameKey.value = frameKey(frame)
}

// Opt into editing the selected frame (auto-proposes anchors to adjust).
async function recalibrateSelected(): Promise<void> {
  const frame = selectedFrame.value
  if (!frame || frame.excluded || frame.sceneTime == null || editingLocked.value) return
  localError.value = null
  await cal.prepareStoredFrame(frame)
}

function autoFit(): void {
  if (cal.activeSceneTime.value != null) void cal.editFrame(cal.activeSceneTime.value)
}

const hasPreviousFrame = computed(() => {
  const current = selectedFrame.value
  return Boolean(current && allFrames.value.some((frame) => (
    frame.sceneTime != null
    && current.sceneTime != null
    && frame.sceneTime < current.sceneTime
    && !frame.excluded
    && frame.resolved
  )))
})
const hasNextFrame = computed(() => {
  const current = selectedFrame.value
  return Boolean(current && allFrames.value.some((frame) => (
    frame.sceneTime != null
    && current.sceneTime != null
    && frame.sceneTime > current.sceneTime
    && !frame.excluded
    && frame.resolved
  )))
})

function borrow(source: 'previous' | 'next' | 'interpolation'): void {
  const frame = selectedFrame.value
  if (!frame?.excluded && frame?.sceneTime != null) {
    void cal.borrowFrame(frame.sceneTime, source)
  }
}

function setSelectedFrameExcluded(excluded: boolean): void {
  const sourceFrameIndex = selectedFrame.value?.sourceFrameIndex
  if (sourceFrameIndex == null || editingLocked.value) return
  localError.value = null
  cal.cancel()
  emit('setFrameExcluded', { sourceFrameIndex, excluded })
}

// On open (and after a run settles) select the first unresolved frame for
// inspection — or the first frame if everything already resolved.
watch(
  [() => props.open, unresolvedFrames, () => props.running, () => props.busy],
  ([open, unresolved, running, busy]) => {
    if (!open) {
      cal.cancel()
      selectedFrameKey.value = null
      localError.value = null
      showAll.value = false
      return
    }
    if (running || busy || isEditing.value || selectedFrame.value) return
    const first = (unresolved as CalibrationReviewSample[])[0] ?? allFrames.value[0]
    if (first) selectFrame(first)
  },
  { immediate: true },
)

function onKeydown(event: KeyboardEvent): void {
  if (event.key === 'Escape') emit('close')
}

function changeFrameRate(event: Event): void {
  emit('update:frameRate', Number((event.target as HTMLSelectElement).value))
}

function changeDirectCalibrationGap(event: Event): void {
  emit(
    'update:directCalibrationMaxGapSeconds',
    Number((event.target as HTMLSelectElement).value),
  )
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="calibration-review-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Pitch calibration"
      tabindex="-1"
      @keydown="onKeydown"
      @click.self="emit('close')"
    >
      <div class="calibration-review-modal">
        <header class="crm-header">
          <div class="crm-title">
            <span>Pitch calibration</span>
            <template v-if="review && !showProgress">
              <strong :class="`crm-status crm-status-${calibrationReviewCurrent ? review.status : 'stale'}`">
                {{ calibrationReviewCurrent ? review.status.toUpperCase() : 'STALE · RECALIBRATE' }}
              </strong>
              <small>
                {{ includedResolvedCount }}/{{ includedFrames.length }} analyzed · {{ ratioPercent }}%
                <template v-if="excludedFrames.length"> · {{ excludedFrames.length }} excluded</template>
              </small>
            </template>
          </div>
          <button class="crm-close" aria-label="Close" @click="emit('close')">×</button>
        </header>

        <div v-if="frameInputReady && !showProgress" class="crm-fps-panel">
          <label class="crm-fps-field">
            <span>Calibration cadence</span>
            <select
              :value="frameRate"
              :disabled="busy || running"
              aria-label="Calibration frame rate"
              @change="changeFrameRate"
            >
              <option :value="0">Native · {{ sourceFps.toFixed(3) }} FPS</option>
              <option v-for="fps in frameRateOptions" :key="fps" :value="fps">Reduced · {{ fps }} FPS</option>
            </select>
          </label>
          <label class="crm-fps-field">
            <span>Direct PnLCalib sampling</span>
            <select
              :value="directCalibrationMaxGapSeconds"
              :disabled="busy || running"
              aria-label="Direct PnLCalib sampling"
              @change="changeDirectCalibrationGap"
            >
              <option
                v-for="gap in directCalibrationGapOptions"
                :key="gap"
                :value="gap"
              >{{ gap === 0 ? 'Every selected frame · default' : `Sparse · max ${gap}s gap` }}</option>
            </select>
          </label>
          <div class="crm-fps-summary">
            <strong>≈ {{ estimatedFrameCount }} frames</strong>
            <span>direct PnLCalib ≈ {{ estimatedDirectFrameCount }} frames</span>
            <span>source {{ sourceFps.toFixed(3) }} · generation {{ materializedFps.toFixed(3) }} FPS</span>
          </div>
          <div v-if="directCalibrationMaxGapSeconds > 0" class="crm-fps-warning">
            <span>Sparse direct calibration is a performance tradeoff. Frames between anchors rely on temporal recovery, so one rejected anchor can leave a continuous unresolved gap.</span>
          </div>
          <div v-if="frameRateRequiresRegeneration" class="crm-fps-warning">
            <span>The current immutable generation was capped at {{ materializedFps.toFixed(3) }} FPS. Native calibration needs a new source-rate generation.</span>
            <button class="crm-secondary" @click="emit('regenerateFrames')">Regenerate native frames</button>
          </div>
        </div>

        <!-- While the calibrate job runs this stage lives entirely here, not in
             the viewport pipeline panel. -->
        <div v-if="showProgress" class="crm-progress">
          <div class="crm-progress-bar"><span :style="{ width: `${visibleProgress.percent}%` }"></span></div>
          <div class="crm-progress-info">
            <strong>{{ visibleProgress.label }}</strong>
            <small>{{ visibleProgress.detail }}</small>
          </div>
          <ol class="crm-workflow">
            <li
              v-for="step in workflowSteps"
              :key="step.id"
              :class="step.state"
            >
              <span>{{ step.state === 'done' ? '✓' : step.state === 'current' ? '●' : '○' }}</span>
              <div>
                <strong>{{ step.label }}</strong>
                <small>{{ step.purpose }}</small>
              </div>
            </li>
          </ol>
          <div class="crm-progress-foot">
            <b>{{ visibleProgress.percent }}%</b>
            <button v-if="canCancel" class="crm-secondary" :disabled="cancelling" @click="emit('cancel')">{{ cancelling ? 'Cancelling…' : 'Cancel' }}</button>
          </div>
        </div>

        <template v-else-if="!frameInputReady">
          <div class="crm-body crm-body-initial">
            <div class="crm-editor">
              <div class="crm-stage">
                <video
                  v-if="mediaUrl"
                  class="crm-video"
                  :src="mediaUrl"
                  muted
                  playsinline
                  controls
                  preload="metadata"
                ></video>
                <div v-else class="crm-stage-empty">No video preview available</div>
                <div class="crm-stage-badge crm-stage-badge-bad">Legacy 1280px analysis generation</div>
              </div>
            </div>
            <div class="crm-initial-guide crm-source-upgrade">
              <strong>Source-resolution frames are required</strong>
              <span>This scene was published before source-resolution extraction existed. A Docker rebuild does not rewrite immutable media generations.</span>
              <span>The explicit cutover extracts new frames from the uploaded source without resize. Old calibration, detections, tracks and identities are removed because their pixel coordinates are no longer valid.</span>
              <span>It does not start calibration. After extraction completes, return here and run full calibration.</span>
            </div>
          </div>
          <footer class="crm-footer">
            <small class="crm-footer-note">This action changes the analysis pixel grid and intentionally invalidates the previous reconstruction result.</small>
            <div class="crm-actions">
              <button class="crm-primary" @click="emit('regenerateFrames')">Regenerate from source video</button>
              <button class="crm-primary" disabled>Run full calibration</button>
            </div>
          </footer>
        </template>

        <template v-else-if="review">
        <div class="crm-timeline" role="group" aria-label="Frame calibration overview — green direct, blue temporal, yellow manual, red unresolved, gray excluded">
          <button
            v-for="(frame, index) in allFrames"
            :key="frameKey(frame)"
            class="crm-strip"
            :class="[
              calibrationReviewTimelineStatus(frame),
              {
                active: selectedFrameKey === frameKey(frame),
                staged: stagedSampleIndices.has(frame.sampleIndex),
              },
            ]"
            :style="{ left: `${stripPercent(index, allFrames.length)}%` }"
            :title="frame.excluded
              ? `Frame #${frame.sourceFrameIndex ?? frame.sampleIndex} · excluded from calibration and reconstruction`
              : `Frame #${frame.sourceFrameIndex ?? frame.sampleIndex} · ${frame.solutionStatus} · ${frame.projectionSource ?? 'none'}`"
            @click="selectFrame(frame)"
          ></button>
        </div>
        <div class="crm-timeline-legend" aria-hidden="true">
          <span><i class="direct"></i>direct</span>
          <span><i class="temporal"></i>temporal</span>
          <span><i class="manual"></i>manual</span>
          <span><i class="staged"></i>staged correction</span>
          <span><i class="unresolved"></i>unresolved</span>
          <span><i class="excluded"></i>excluded</span>
        </div>

        <div class="crm-body">
          <!-- One window: inspect and calibrate the frame right here. -->
          <div class="crm-editor">
            <div class="crm-stage">
              <img
                v-if="exactFrameUrl && !exactFrameLoadError"
                :key="exactFrameUrl"
                class="crm-video"
                :src="exactFrameUrl"
                :alt="`Exact analysis frame #${selectedFrame?.sourceFrameIndex ?? ''}`"
                draggable="false"
                @load="exactFrameLoadError = false"
                @error="exactFrameLoadError = true"
              />
              <div v-else-if="exactFrameLoadError" class="crm-stage-empty">
                Exact analysis frame could not be loaded. No approximate video frame is substituted.
              </div>
              <div v-else class="crm-stage-empty">Select a frame with an exact source-frame index</div>
              <!-- Editing: draggable anchors. -->
              <PitchCalibrationOverlay
                v-if="isEditing"
                :draft="cal.draft.value"
                :diagnostics="null"
                :qa-frame="null"
                :qa-frame-size="{ width: 0, height: 0 }"
                :qa-markings="[]"
                @overlay-element="cal.bindOverlay"
                @update-drag="cal.updateDraggedAnchor"
                @finish-drag="cal.finishAnchorDrag"
                @start-anchor-drag="cal.startAnchorDrag"
                @nudge-anchor="cal.nudgeAnchor"
              />
              <!-- Inspecting: the frame's actual projected pitch lines (read-only). -->
              <svg
                v-else-if="inspectionMarkings.length"
                class="crm-inspect-overlay"
                :viewBox="`0 0 ${inspectionSize.width} ${inspectionSize.height}`"
                preserveAspectRatio="xMidYMid meet"
                aria-label="Projected pitch calibration for this frame"
              >
                <polyline
                  v-for="marking in inspectionMarkings"
                  :key="marking.id"
                  class="crm-inspect-line"
                  :class="marking.kind"
                  :points="marking.points.map((point) => `${point.x},${point.y}`).join(' ')"
                />
              </svg>
              <div v-if="cal.busy.value" class="crm-stage-badge">Preparing overlay…</div>
              <div v-else-if="busy" class="crm-stage-badge">Saving…</div>
              <div v-else-if="updatingFrameExclusion" class="crm-stage-badge">Updating frame set…</div>
              <div v-else-if="selectedFrame?.excluded" class="crm-stage-badge crm-stage-badge-excluded">Excluded from calibration and reconstruction</div>
              <div v-else-if="selectedFrame && !isEditing && !inspectionMarkings.length" class="crm-stage-badge crm-stage-badge-bad">No calibration for this frame</div>
              <div v-if="selectedFrame?.sourceFrameIndex != null" class="crm-exact-frame-badge">
                Exact analysis JPEG · #{{ selectedFrame.sourceFrameIndex }}
              </div>
            </div>

            <div class="crm-controls">
              <template v-if="isEditing">
                <label class="crm-field">
                  <span>Preset</span>
                  <select
                    :value="cal.preset.value"
                    :disabled="editingLocked || cal.busy.value"
                    aria-label="Calibration preset"
                    @change="cal.changePreset"
                  >
                    <option v-for="item in cal.presets" :key="item.value" :value="item.value">{{ item.label }}</option>
                  </select>
                </label>
                <button class="crm-secondary" :disabled="editingLocked || cal.busy.value" title="Run a new PnLCalib inference on this frame" @click="autoFit">Run PnLCalib</button>
                <div class="crm-borrow-actions">
                  <button class="crm-secondary" :disabled="editingLocked || cal.busy.value || !hasPreviousFrame" @click="borrow('previous')">Use previous</button>
                  <button class="crm-secondary" :disabled="editingLocked || cal.busy.value || !hasNextFrame" @click="borrow('next')">Use next</button>
                  <button class="crm-secondary" :disabled="editingLocked || cal.busy.value || !hasPreviousFrame || !hasNextFrame" @click="borrow('interpolation')">Interpolate neighbors</button>
                </div>
                <span class="crm-align">{{ cal.draft.value?.alignmentError == null ? 'no fit yet' : `${cal.draft.value?.alignmentError.toFixed(1)}px error` }}</span>
                <p v-if="manualQaWarning" class="crm-qa-warning">
                  Automatic line-mask QA does not support this fit. This can be a false negative on a partial or noisy pitch view. Check the projected lines before overriding the warning.
                  <strong>{{ manualQaDetail }}</strong>
                </p>
                <button class="crm-secondary" :disabled="cal.saving.value" @click="cal.cancel">Cancel</button>
                <button
                  v-if="manualQaWarning"
                  class="crm-primary crm-save crm-save-warning"
                  :disabled="!cal.draft.value || editingLocked || cal.busy.value"
                  @click="cal.saveFrame(true)"
                >{{ cal.saving.value || busy ? 'Saving draft…' : 'Save despite QA warning' }}</button>
                <button
                  v-else
                  class="crm-primary crm-save"
                  :disabled="!cal.draft.value || editingLocked || cal.busy.value"
                  @click="cal.saveFrame()"
                >{{ cal.saving.value || busy ? 'Saving draft…' : 'Save frame correction' }}</button>
              </template>
              <template v-else>
                <span
                  class="crm-inspect-status"
                  :class="selectedFrame?.excluded ? 'excluded' : selectedFrame?.resolved ? 'ok' : 'bad'"
                >
                  {{ selectedFrame
                    ? selectedFrame.excluded
                      ? 'Excluded · not sent to calibration or reconstruction'
                      : `${selectedFrame.resolved ? 'Calibrated' : 'Not calibrated'} · ${selectedFrame.solutionStatus}`
                    : 'Select a frame to inspect' }}
                </span>
                <button
                  v-if="selectedFrame?.excluded"
                  class="crm-secondary"
                  :disabled="editingLocked"
                  @click="setSelectedFrameExcluded(false)"
                >Restore frame to segment</button>
                <template v-else>
                <button class="crm-primary crm-save" :disabled="!selectedFrame || editingLocked" @click="recalibrateSelected">Recalibrate this frame</button>
                <button
                  class="crm-secondary crm-exclude"
                  :disabled="!selectedFrame?.sourceFrameIndex || editingLocked"
                  title="Remove this exact source frame from both calibration and reconstruction. A full recalibration will be required."
                  @click="setSelectedFrameExcluded(true)"
                >Exclude from segment</button>
                <div class="crm-borrow-actions">
                  <button class="crm-secondary" :disabled="editingLocked || !hasPreviousFrame" @click="borrow('previous')">Use previous</button>
                  <button class="crm-secondary" :disabled="editingLocked || !hasNextFrame" @click="borrow('next')">Use next</button>
                  <button class="crm-secondary" :disabled="editingLocked || !hasPreviousFrame || !hasNextFrame" @click="borrow('interpolation')">Interpolate</button>
                </div>
                </template>
              </template>
            </div>
            <p v-if="localError" class="crm-error" role="alert">{{ localError }}</p>
            <p class="crm-hint">
              <template v-if="isEditing">Drag the numbered anchors onto the pitch markings (or pick a preset), then save the correction. Saving only stages this frame; it does not start calibration. After all edits, use <strong>Finalize corrections</strong>.</template>
              <template v-else>Click any frame to inspect the exact JPEG used by analysis — never an approximate video seek. The green lines are the projected pitch. Excluding a frame removes it from both calibration and reconstruction after the required full recalibration.</template>
            </p>
          </div>

          <div class="crm-side">
            <div class="crm-side-head">
              <span>{{ showAll ? 'All frames' : 'Problem / excluded frames' }} · {{ listFrames.length }}</span>
              <button class="crm-toggle" @click="showAll = !showAll">{{ showAll ? 'Only problems / excluded' : 'Show all' }}</button>
            </div>
            <ul v-if="listFrames.length" class="crm-list">
              <li
                v-for="frame in listFrames"
                :key="frameKey(frame)"
                class="crm-list-item"
                :class="{ active: selectedFrameKey === frameKey(frame), excluded: frame.excluded }"
                @click="selectFrame(frame)"
              >
                <div class="crm-list-head">
                  <span class="crm-dot" :class="frame.excluded ? 'excluded' : frame.resolved ? 'ok' : 'bad'"></span>
                  <span class="crm-list-frame">#{{ frame.sourceFrameIndex ?? frame.sampleIndex }}</span>
                  <span class="crm-list-time">{{ (frame.sceneTime ?? 0).toFixed(2) }}s</span>
                  <span class="crm-list-status" :class="{ ok: frame.resolved, excluded: frame.excluded }">{{ frame.solutionStatus }}</span>
                </div>
                <div v-if="frame.excluded" class="crm-list-meta">
                  <span>omitted from calibration and reconstruction</span>
                </div>
                <div v-else-if="!frame.resolved && (frame.residualP95 != null || frame.rejectionReasons.length)" class="crm-list-meta">
                  <span v-if="frame.residualP95 != null">residual p95 {{ frame.residualP95.toFixed(1) }}px</span>
                  <span v-for="reason in frame.rejectionReasons" :key="reason" class="crm-reason">{{ reason }}</span>
                </div>
              </li>
            </ul>
            <div v-else class="crm-list crm-list-empty">All included frames resolved.</div>
          </div>
        </div>

        <div v-if="review.warnings.length" class="crm-warnings">
          <span v-for="warning in review.warnings" :key="warning">{{ warning }}</span>
        </div>

        <footer class="crm-footer">
          <small class="crm-footer-note">
            <template v-if="isReviewing">Fix every unresolved frame. If that is impossible, explicitly authorize image-space tracking only for the listed gaps.</template>
            <template v-else-if="canReconstruct">Calibration is cleared — you can continue to Reconstruction.</template>
            <template v-else>{{ nextStepReason }}</template>
          </small>
          <div class="crm-actions">
            <button class="crm-secondary crm-danger" :disabled="resetting || busy" @click="emit('reset')">{{ resetting ? 'Resetting…' : 'Reset' }}</button>
            <button class="crm-secondary" :disabled="busy || frameRateRequiresRegeneration" title="Re-run calibration over the complete sampled timeline" @click="emit('rerun')">Full recalibration</button>
            <button
              v-if="stagedCount"
              class="crm-primary"
              :disabled="busy || finalizing"
              title="Apply staged edits and recompute only their temporal dependency region"
              @click="emit('finalize')"
            >{{ finalizing ? 'Finalizing…' : `Finalize ${stagedCount} correction(s)` }}</button>
            <button
              v-if="isReviewing"
              class="crm-secondary"
              :disabled="confirming || busy"
              @click="emit('confirm')"
            >{{ confirming ? 'Authorizing…' : `Authorize image fallback · ${unresolvedFrames.length}` }}</button>
            <button
              class="crm-primary"
              :disabled="!canReconstruct || busy"
              :title="nextStepReason ?? 'Continue to Reconstruction'"
              @click="emit('reconstruct')"
            >2 · Continue to Reconstruction</button>
          </div>
        </footer>
        </template>

        <template v-else>
          <div class="crm-timeline crm-timeline-empty" role="group" aria-label="Calibration timeline is empty">
            <span>Timeline will appear after calibration</span>
          </div>
          <div class="crm-body crm-body-initial">
            <div class="crm-editor">
              <div class="crm-stage">
                <video
                  v-if="mediaUrl"
                  class="crm-video"
                  :src="mediaUrl"
                  muted
                  playsinline
                  controls
                  preload="metadata"
                ></video>
                <div v-else class="crm-stage-empty">No video preview available</div>
                <div class="crm-stage-badge">Calibration has not been run for the current inputs</div>
              </div>
              <p class="crm-hint">Review the source preview, then run full calibration. Per-frame correction becomes available directly on the timeline after the first artifact is published.</p>
            </div>
            <div class="crm-initial-guide">
              <strong>Calibration stage</strong>
              <span>1. Run direct PnLCalib on every selected frame by default.</span>
              <span>2. Inspect direct, temporal and unresolved frames.</span>
              <span>3. Correct individual frames or reset and start again.</span>
              <span>4. Continue only after the calibration gate is cleared.</span>
            </div>
          </div>
          <footer class="crm-footer">
            <small class="crm-footer-note">{{ nextStepReason }}</small>
            <div class="crm-actions">
              <button class="crm-secondary crm-danger" disabled>Reset</button>
              <button class="crm-secondary" :disabled="busy || frameRateRequiresRegeneration" @click="emit('rerun')">Run full calibration</button>
              <button class="crm-primary" disabled :title="nextStepReason ?? undefined">2 · Continue to Reconstruction</button>
            </div>
          </footer>
        </template>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.calibration-review-backdrop {
  position: fixed;
  inset: 0;
  z-index: 60;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: rgba(4, 6, 7, 0.68);
  backdrop-filter: blur(2px);
}
.calibration-review-modal {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: min(1040px, 100%);
  max-height: calc(100vh - 48px);
  padding: 18px 20px;
  background: var(--panel, #101314);
  border: 1px solid var(--line-strong, rgba(238, 243, 235, 0.2));
  border-radius: 14px;
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.5);
  color: #eef3eb;
}
.crm-header { display: flex; align-items: center; justify-content: space-between; }
.crm-title { display: flex; align-items: center; gap: 10px; font-size: 15px; font-weight: 600; }
.crm-title small { color: var(--muted, #8b9490); font-weight: 400; font-size: 11px; }
.crm-status { font-size: 11px; letter-spacing: 0.06em; padding: 2px 8px; border-radius: 999px; }
.crm-status-review { background: rgba(255, 104, 79, 0.18); color: var(--orange, #ff684f); }
.crm-status-ready { background: rgba(120, 220, 150, 0.16); color: #86e0a3; }
.crm-status-confirmed { background: rgba(255, 211, 106, 0.16); color: var(--accent, #ffd36a); }
.crm-status-stale { background: rgba(255, 104, 79, 0.18); color: var(--orange, #ff684f); }
.crm-close { background: none; border: none; color: var(--muted, #8b9490); font-size: 22px; line-height: 1; cursor: pointer; }
.crm-close:hover { color: #eef3eb; }
.crm-fps-panel { display: flex; flex-wrap: wrap; align-items: center; gap: 14px; padding: 10px 12px; border: 1px solid var(--line, rgba(238, 243, 235, 0.11)); border-radius: 8px; background: var(--panel-2, #141819); }
.crm-fps-field { display: flex; flex-direction: column; gap: 4px; min-width: 220px; color: var(--muted, #8b9490); font-size: 10px; }
.crm-fps-field select { padding: 6px 8px; border: 1px solid var(--line-strong, rgba(238, 243, 235, 0.2)); border-radius: 6px; background: #0d1011; color: #eef3eb; }
.crm-fps-summary { display: flex; flex-direction: column; gap: 2px; font-size: 11px; }
.crm-fps-summary span { color: var(--muted, #8b9490); }
.crm-fps-warning { display: flex; align-items: center; gap: 10px; margin-left: auto; color: var(--orange, #ff684f); font-size: 11px; }
.crm-fps-warning span { max-width: 390px; }
.crm-progress { display: flex; flex-direction: column; gap: 10px; padding: 32px 8px; }
.crm-progress-bar { height: 6px; border-radius: 999px; background: rgba(238, 243, 235, 0.1); overflow: hidden; }
.crm-progress-bar span { display: block; height: 100%; background: var(--accent, #ffd36a); transition: width 0.3s ease; }
.crm-progress-info { display: flex; flex-direction: column; gap: 4px; }
.crm-progress-info strong { font-size: 14px; }
.crm-progress-info small { color: var(--muted, #8b9490); font-size: 12px; }
.crm-progress-foot { display: flex; align-items: center; justify-content: space-between; }
.crm-progress-foot b { font-size: 20px; color: var(--accent, #ffd36a); }
.crm-workflow { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 7px; margin: 4px 0 0; padding: 0; list-style: none; }
.crm-workflow li { display: flex; gap: 7px; padding: 8px; border: 1px solid var(--line, rgba(238, 243, 235, 0.11)); border-radius: 7px; opacity: .48; }
.crm-workflow li.done { opacity: .72; color: #86e0a3; }
.crm-workflow li.current { opacity: 1; border-color: var(--accent, #ffd36a); color: var(--accent, #ffd36a); }
.crm-workflow li div { display: flex; flex-direction: column; gap: 3px; }
.crm-workflow li strong { font-size: 10px; }
.crm-workflow li small { color: var(--muted, #8b9490); font-size: 9px; line-height: 1.35; }
.crm-danger { border-color: rgba(255, 104, 79, 0.4); color: #f0a091; }
.crm-danger:hover:not(:disabled) { border-color: var(--orange, #ff684f); }
.crm-timeline { position: relative; height: 22px; flex: none; border-radius: 4px; overflow: hidden; background: rgba(238, 243, 235, 0.06); }
.crm-timeline-empty { display: grid; place-items: center; height: 30px; border: 1px dashed rgba(238, 243, 235, 0.15); color: var(--muted, #8b9490); font-size: 10px; }
.crm-strip { position: absolute; top: 0; bottom: 0; width: 3px; margin-left: -1px; padding: 0; border: 0; cursor: pointer; }
.crm-strip.direct { background: rgba(134, 224, 163, 0.72); }
.crm-strip.temporal { background: #65bfff; }
.crm-strip.manual { background: var(--accent, #ffd36a); }
.crm-strip.unresolved { background: var(--orange, #ff684f); }
.crm-strip.excluded { background: #6f7778; }
.crm-strip.active { width: 5px; margin-left: -2px; outline: 1px solid #fff4cc; outline-offset: -1px; filter: brightness(1.25); z-index: 1; }
.crm-strip.staged { width: 7px; margin-left: -3px; outline: 2px solid #d88cff; outline-offset: -2px; z-index: 2; }
.crm-timeline-legend { display: flex; align-items: center; gap: 12px; flex: none; color: var(--muted, #8b9490); font-size: 10px; }
.crm-timeline-legend span { display: inline-flex; align-items: center; gap: 5px; }
.crm-timeline-legend i { width: 8px; height: 8px; border-radius: 2px; }
.crm-timeline-legend i.direct { background: rgba(134, 224, 163, 0.72); }
.crm-timeline-legend i.temporal { background: #65bfff; }
.crm-timeline-legend i.manual { background: var(--accent, #ffd36a); }
.crm-timeline-legend i.staged { background: #d88cff; }
.crm-timeline-legend i.unresolved { background: var(--orange, #ff684f); }
.crm-timeline-legend i.excluded { background: #6f7778; }
.crm-body { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr); gap: 14px; min-height: 0; overflow: hidden; }
.crm-body-initial { align-items: start; }
.crm-initial-guide { display: flex; flex-direction: column; gap: 10px; padding: 16px; border: 1px solid var(--line, rgba(238, 243, 235, 0.11)); border-radius: 8px; background: var(--panel-2, #141819); color: var(--muted, #8b9490); font-size: 12px; line-height: 1.4; }
.crm-initial-guide strong { color: #eef3eb; font-size: 13px; }
.crm-editor { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.crm-stage { position: relative; border-radius: 8px; overflow: hidden; background: #000; aspect-ratio: 16 / 9; }
.crm-video { width: 100%; height: 100%; object-fit: contain; display: block; }
.crm-stage-empty { display: grid; place-items: center; height: 100%; color: var(--muted, #8b9490); font-size: 12px; }
.crm-stage :deep(.pitch-calibration-overlay) { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; }
.crm-inspect-overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.crm-inspect-line { fill: none; stroke: #86e0a3; stroke-width: 2; opacity: 0.9; }
.crm-inspect-line.curve { stroke-dasharray: 6 4; }
.crm-stage-badge { position: absolute; top: 8px; left: 8px; padding: 3px 8px; font-size: 11px; border-radius: 6px; background: rgba(0, 0, 0, 0.62); color: var(--accent, #ffd36a); }
.crm-stage-badge-bad { color: var(--orange, #ff684f); }
.crm-stage-badge-excluded { color: #d5dcdd; }
.crm-exact-frame-badge { position: absolute; right: 8px; bottom: 8px; padding: 3px 7px; border-radius: 5px; background: rgba(0, 0, 0, 0.66); color: #d5dcdd; font-size: 10px; pointer-events: none; }
.crm-inspect-status { font-size: 12px; font-weight: 600; }
.crm-inspect-status.ok { color: #86e0a3; }
.crm-inspect-status.bad { color: var(--orange, #ff684f); }
.crm-inspect-status.excluded { color: #b8c0c1; }
.crm-controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.crm-borrow-actions { display: flex; gap: 5px; flex-wrap: wrap; }
.crm-borrow-actions .crm-secondary { padding: 5px 8px; font-size: 10px; }
.crm-field { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--muted, #8b9490); }
.crm-field select { background: var(--panel-2, #141819); color: #eef3eb; border: 1px solid var(--line-strong, rgba(238, 243, 235, 0.2)); border-radius: 6px; padding: 5px 7px; font-size: 12px; }
.crm-align { font-size: 11px; color: var(--muted, #8b9490); }
.crm-qa-warning { flex-basis: 100%; margin: 0; padding: 8px 10px; border: 1px solid rgba(255, 166, 0, .45); border-radius: 6px; color: #ffc46b; background: rgba(255, 166, 0, .08); font-size: 11px; line-height: 1.45; }
.crm-qa-warning strong { display: block; margin-top: 3px; color: #ffd89b; font-variant-numeric: tabular-nums; }
.crm-save-warning { border-color: rgba(255, 166, 0, .75); background: rgba(167, 91, 0, .75); }
.crm-save { margin-left: auto; }
.crm-hint { margin: 0; font-size: 10px; color: var(--muted, #8b9490); line-height: 1.5; }
.crm-error { margin: 0; font-size: 11px; color: var(--orange, #ff684f); }
.crm-side { display: flex; flex-direction: column; min-height: 0; }
.crm-side-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; font-size: 11px; color: var(--muted, #8b9490); flex: none; }
.crm-toggle { padding: 3px 8px; font-size: 11px; border: 1px solid var(--line-strong, rgba(238, 243, 235, 0.2)); border-radius: 6px; background: none; color: #eef3eb; cursor: pointer; }
.crm-toggle:hover { border-color: var(--accent, #ffd36a); }
.crm-dot { width: 8px; height: 8px; border-radius: 999px; flex: none; }
.crm-dot.ok { background: #86e0a3; }
.crm-dot.bad { background: var(--orange, #ff684f); }
.crm-dot.excluded { background: #6f7778; }
.crm-list { list-style: none; margin: 0; padding: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
.crm-list-empty { display: grid; place-items: center; color: var(--muted, #8b9490); font-size: 12px; }
.crm-list-item { padding: 8px 10px; border: 1px solid var(--line, rgba(238, 243, 235, 0.11)); border-radius: 8px; cursor: pointer; background: var(--panel-2, #141819); }
.crm-list-item.active { border-color: var(--accent, #ffd36a); }
.crm-list-item.excluded { opacity: 0.82; }
.crm-list-head { display: flex; align-items: center; gap: 8px; font-size: 12px; }
.crm-list-frame { font-weight: 600; }
.crm-list-time { color: var(--muted, #8b9490); }
.crm-list-status { margin-left: auto; color: var(--orange, #ff684f); font-size: 11px; }
.crm-list-status.ok { color: #86e0a3; }
.crm-list-status.excluded { color: #b8c0c1; }
.crm-list-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; font-size: 11px; color: var(--muted, #8b9490); }
.crm-reason { padding: 1px 6px; border-radius: 4px; background: rgba(255, 104, 79, 0.12); color: #f0a091; }
.crm-warnings { display: flex; flex-direction: column; gap: 3px; font-size: 11px; color: var(--muted, #8b9490); }
.crm-footer { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding-top: 8px; border-top: 1px solid var(--line, rgba(238, 243, 235, 0.11)); flex: none; }
.crm-footer-note { color: var(--muted, #8b9490); font-size: 11px; max-width: 55%; }
.crm-actions { display: flex; gap: 8px; }
.crm-secondary, .crm-primary { padding: 7px 14px; font-size: 12px; border-radius: 8px; cursor: pointer; border: 1px solid var(--line-strong, rgba(238, 243, 235, 0.2)); }
.crm-secondary { background: none; color: #eef3eb; }
.crm-secondary:hover:not(:disabled) { border-color: var(--accent, #ffd36a); }
.crm-primary { background: var(--accent, #ffd36a); color: #1a1400; border-color: var(--accent, #ffd36a); font-weight: 600; }
.crm-primary:disabled, .crm-secondary:disabled { opacity: 0.45; cursor: default; }
</style>
