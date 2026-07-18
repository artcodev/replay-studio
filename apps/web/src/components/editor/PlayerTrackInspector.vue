<script setup lang="ts">
import { computed } from 'vue'
import { interpolateKeyframes } from '../../lib/interpolate'
import type { SceneVideoAsset } from '../../types/scene'
import type { Track } from '../../types/tracking'

const props = defineProps<{
  track: Track
  teamName: string | null
  currentTime: number
  editMode: boolean
  processingStatus: string
  qualityVerdict: string
  calibrationLabel: string
  visiblePitchSide: string
  attackingGoalSide: string
  pitchCalibrationStatus: string | null
  pitchCalibrationSupportedLines: number | null
  pitchCalibrationRectangle: string | null
  pitchCalibrationReason: string | null
  sceneVideo: SceneVideoAsset | null
  analysisFrameCount: number
  identityValidationLabel: string
}>()

const emit = defineEmits<{
  'update:editMode': [value: boolean]
  'update-label': [value: string]
  'update-number': [value: number]
  'update-position': [axis: 'x' | 'z', value: string]
}>()

const position = computed(() => interpolateKeyframes(props.track.keyframes, props.currentTime))
const frameConfidence = computed(() => Math.round(position.value.confidence * 100))
</script>

<template>
  <div class="player-identity">
    <span class="large-jersey" :style="{ background: track.color }">{{ track.number }}</span>
    <div><p>Selected player</p><h2>{{ track.label }}</h2><small>{{ teamName }}</small></div>
  </div>

  <div class="field-grid">
    <div class="field-group"><label>Display name</label><input :value="track.label" @input="emit('update-label', ($event.target as HTMLInputElement).value)" /></div>
    <div class="field-group"><label>Number</label><input :value="track.number" type="number" min="0" max="99" @input="emit('update-number', Number(($event.target as HTMLInputElement).value))" /></div>
  </div>

  <div class="field-group">
    <div class="label-row"><label>Position at {{ currentTime.toFixed(2) }}s</label><span>metres</span></div>
    <div class="position-grid">
      <label>X <input type="number" step="0.1" :value="position.x.toFixed(2)" @change="emit('update-position', 'x', ($event.target as HTMLInputElement).value)" /></label>
      <label>Z <input type="number" step="0.1" :value="position.z.toFixed(2)" @change="emit('update-position', 'z', ($event.target as HTMLInputElement).value)" /></label>
    </div>
    <button class="wide-action" :class="{ active: editMode }" @click="emit('update:editMode', !editMode)">◎ {{ editMode ? 'Click position on pitch…' : 'Place on pitch' }}</button>
  </div>

  <slot name="presence" />

  <div class="quality-card">
    <div><span>Frame confidence</span><strong>{{ frameConfidence }}%</strong></div>
    <div class="quality-bar"><i :style="{ width: `${frameConfidence}%` }" /></div>
    <small>{{ track.keyframes.length }} keyframes · linear interpolation</small>
    <div><span>Compute status</span><strong>{{ processingStatus.toUpperCase() }}</strong></div>
    <div><span>Quality verdict</span><strong :class="`quality-${qualityVerdict}`">{{ qualityVerdict.toUpperCase() }}</strong></div>
    <div><span>Pitch calibration</span><strong>{{ calibrationLabel }}</strong></div>
    <div><span>Visible pitch side</span><strong>{{ visiblePitchSide.toUpperCase() }}</strong></div>
    <div><span>Attacking goal</span><strong>{{ attackingGoalSide.toUpperCase() }}</strong></div>
    <small v-if="pitchCalibrationStatus === 'ready'">{{ pitchCalibrationSupportedLines }} markings · {{ pitchCalibrationRectangle }}</small>
    <small v-else>{{ pitchCalibrationReason || 'Screen-relative coordinates' }}</small>
    <small v-if="sceneVideo?.reconstruction?.diagnostics" class="reconstruction-diagnostics">
      {{ analysisFrameCount }} frames · {{ sceneVideo.reconstruction.diagnostics.meanPersonDetections }} detections/frame ·
      {{ sceneVideo.reconstruction.diagnostics.rawTrackCount }} → {{ sceneVideo.reconstruction.diagnostics.stableTrackCount }} → {{ sceneVideo.reconstruction.diagnostics.acceptedTrackCount }} tracks
    </small>
    <small v-if="sceneVideo?.reconstruction?.diagnostics?.identityObservationCoverage != null" class="reconstruction-diagnostics">
      Video observations retained {{ Math.round(sceneVideo.reconstruction.diagnostics.identityObservationCoverage * 100) }}% (not identity accuracy) ·
      metric {{ Math.round((sceneVideo.reconstruction.diagnostics.metricObservationCoverage ?? 0) * 100) }}% ·
      {{ sceneVideo.reconstruction.diagnostics.discardedProjectedObservationCount ?? 0 }} rejected metric observations
    </small>
    <small class="reconstruction-diagnostics">{{ identityValidationLabel }}</small>
    <small v-if="sceneVideo?.reconstruction?.diagnostics?.identity" class="reconstruction-diagnostics">
      ReID {{ sceneVideo.reconstruction.diagnostics.identity.reidSelectedIndependentSampleCount ?? 0 }} independent / {{ sceneVideo.reconstruction.diagnostics.identity.reidUsableObservationCount ?? 0 }} usable crops ·
      jersey OCR {{ sceneVideo.reconstruction.diagnostics.identity.jerseyReliablePersonCount ?? 0 }} reliable, {{ sceneVideo.reconstruction.diagnostics.identity.jerseyConflictPersonCount ?? 0 }} conflicts ·
      association p10 {{ sceneVideo.reconstruction.diagnostics.identity.associationConfidenceP10 == null ? 'n/a' : `${Math.round(sceneVideo.reconstruction.diagnostics.identity.associationConfidenceP10 * 100)}%` }}
    </small>
  </div>
</template>
