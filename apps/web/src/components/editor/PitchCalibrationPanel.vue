<script setup lang="ts">
import {
  calibrationLineResidualLabel,
  calibrationRejectionReasonLabel,
  type CalibrationFrameDiagnostics,
} from '../../lib/calibrationDiagnostics'
import type { PitchCalibrationDraft, PitchCalibrationPreset } from '../../types/calibration'

defineProps<{
  draft: PitchCalibrationDraft
  activeAtCurrentTime: boolean
  diagnostics: CalibrationFrameDiagnostics | null
  warnings: string[]
  preset: PitchCalibrationPreset
  presets: Array<{ value: PitchCalibrationPreset; label: string }>
  loading: boolean
  applying: boolean
}>()

const emit = defineEmits<{
  changePreset: [event: Event]
  calibrateAgain: []
  returnToFrame: []
  apply: []
}>()

function percent(value: number | null | undefined) {
  return value === null || value === undefined ? '—' : `${Math.round(value * 100)}%`
}

function pixels(value: number | null | undefined) {
  return value === null || value === undefined ? '—' : `${value.toFixed(1)} px`
}

function visibleSideLabel(
  side: 'left' | 'right' | 'unknown' | null | undefined,
  trusted: boolean | null | undefined,
) {
  if (!side || side === 'unknown') return 'UNKNOWN'
  return trusted ? side.toUpperCase() : `CANDIDATE ${side.toUpperCase()} · UNTRUSTED`
}

function rawLineLabel(line: CalibrationFrameDiagnostics['lines'][number], index: number) {
  return line.label ?? line.name ?? line.family ?? `Semantic line ${index + 1}`
}
</script>

<template>
  <div class="pitch-calibration-panel" :class="{ left: preset.endsWith('right') }" @click.stop>
    <div class="calibration-panel-heading">
      <div>
        <span>Current frame calibration</span>
        <strong>{{ draft.sceneTime.toFixed(2) }}s · frame {{ draft.frameIndex }}</strong>
      </div>
      <i :class="diagnostics?.status ?? draft.quality">{{ diagnostics?.status ?? draft.quality }}</i>
    </div>
    <div v-if="diagnostics" class="calibration-diagnostics" aria-live="polite">
      <div><span>Method</span><strong>{{ diagnostics.method }}</strong></div>
      <div><span>Confidence</span><strong>{{ percent(draft.confidence) }}</strong></div>
      <div><span>Detected keypoints</span><strong>{{ diagnostics.keypointCount ?? '—' }}</strong></div>
      <div><span>Inliers</span><strong>{{ diagnostics.inlierCount ?? '—' }}<template v-if="diagnostics.inlierRatio !== null"> · {{ percent(diagnostics.inlierRatio) }}</template></strong></div>
      <div><span>Projected markings · p50</span><strong>{{ pixels(diagnostics.residualP50) }}</strong></div>
      <div><span>Projected markings · p95</span><strong>{{ pixels(diagnostics.residualP95) }}</strong></div>
      <div><span>Projected markings · precision</span><strong>{{ percent(diagnostics.precision) }}</strong></div>
      <div><span>Projected markings · recall</span><strong>{{ percent(diagnostics.recall) }}</strong></div>
      <div><span>Projected markings · F1</span><strong>{{ percent(diagnostics.f1) }}</strong></div>
      <div v-if="diagnostics.sourceStatus"><span>Previous rebuild evidence</span><strong>{{ diagnostics.sourceStatus }}</strong></div>
    </div>
    <div v-if="diagnostics?.lines.length" class="calibration-source-line-list">
      <strong>Observed semantic lines · {{ diagnostics.lines.length }}</strong>
      <span v-for="(line, index) in diagnostics.lines.slice(0, 8)" :key="`line-label-${line.id ?? index}`">
        <i :class="{ accepted: line.accepted ?? line.inlier, rejected: line.accepted === false || line.inlier === false }" />
        <span>{{ rawLineLabel(line, index) }}</span>
        <b v-if="(line.confidence !== null && line.confidence !== undefined) || calibrationLineResidualLabel(line)">
          <template v-if="line.confidence !== null && line.confidence !== undefined">{{ percent(line.confidence) }}</template>
          <em v-if="calibrationLineResidualLabel(line)">{{ calibrationLineResidualLabel(line) }}</em>
        </b>
      </span>
      <small v-if="diagnostics.lines.length > 8">+{{ diagnostics.lines.length - 8 }} more lines</small>
    </div>
    <label>
      <span>Visible landmark</span>
      <select
        :value="preset"
        :disabled="loading"
        aria-label="Pitch landmark preset"
        @change="emit('changePreset', $event)"
      >
        <option v-for="option in presets" :key="option.value" :value="option.value">{{ option.label }}</option>
      </select>
    </label>
    <div v-if="!diagnostics" class="calibration-score">
      <span>Projected-markings residual p50</span>
      <strong>{{ draft.alignmentError === null ? 'NO SCORE' : `${draft.alignmentError.toFixed(1)} px` }}</strong>
    </div>
    <div class="calibration-score">
      <span>Visible pitch side</span>
      <strong :class="{ untrusted: diagnostics && !diagnostics.visibleSideTrusted }">
        {{ visibleSideLabel(diagnostics?.visibleSide, diagnostics?.visibleSideTrusted) }}
      </strong>
    </div>
    <div v-if="diagnostics?.rejectionReasons.length" class="calibration-rejections">
      <strong>{{ diagnostics.status === 'rejected' ? 'Rejection reasons' : 'Source candidate rejection reasons' }}</strong>
      <span v-for="reason in diagnostics.rejectionReasons" :key="reason">{{ calibrationRejectionReasonLabel(reason) }}</span>
    </div>
    <p v-if="!activeAtCurrentTime">Move the playhead back to {{ draft.sceneTime.toFixed(2) }}s to edit these anchors.</p>
    <p v-else>Yellow lines are the projected pitch. Colored dots are detected source points; red vectors end at their projected positions. Drag numbered anchors to refine this frame manually.</p>
    <small v-for="warning in warnings" :key="warning">{{ warning }}</small>
    <small class="calibration-preview-note">Preview only · diagnostics are recorded, but anchors and tracks stay unchanged until Apply & rebuild.</small>
    <div class="calibration-actions">
      <button :disabled="loading" @click="emit('calibrateAgain')">{{ loading ? 'Updating…' : 'Calibrate again' }}</button>
      <button v-if="!activeAtCurrentTime" @click="emit('returnToFrame')">Return to frame</button>
      <button class="apply" :disabled="loading || applying" @click="emit('apply')">{{ applying ? 'Applying…' : 'Apply & rebuild' }}</button>
    </div>
  </div>
</template>
