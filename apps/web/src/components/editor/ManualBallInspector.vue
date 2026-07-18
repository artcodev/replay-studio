<script setup lang="ts">
import type { Keyframe } from '../../types/tracking'

defineProps<{
  mode: 'automatic' | 'manual'
  saving: boolean
  reconstructionRunning: boolean
  selectedKeyframe: Keyframe | null
  currentTime: number
  placementMode: boolean
  manualCount: number
  automaticCount: number
}>()

const emit = defineEmits<{
  'change-mode': [event: Event]
  'update-coordinate': [axis: 'x' | 'z', value: string]
  'add-keypoint': [time: number]
  'toggle-placement': []
}>()
</script>

<template>
  <div class="manual-ball-inspector">
    <div class="player-identity">
      <span class="large-jersey ball-inspector-icon">●</span>
      <div>
        <p>Selected object</p>
        <h2>Match ball</h2>
        <small>{{ mode === 'manual' ? 'Human-authored trajectory' : 'Detector trajectory' }}</small>
      </div>
    </div>

    <div class="field-group">
      <label>Trajectory source</label>
      <select :value="mode" :disabled="saving || reconstructionRunning" aria-label="Ball trajectory source" @change="emit('change-mode', $event)">
        <option value="automatic">Automatic detection</option>
        <option value="manual">Manual keypoints</option>
      </select>
      <small>Both versions are retained, so switching modes never destroys the other trajectory.</small>
    </div>

    <template v-if="mode === 'manual'">
      <div class="field-group">
        <div class="label-row">
          <label>{{ selectedKeyframe ? `Keypoint at ${selectedKeyframe.t.toFixed(3)}s` : `Playhead at ${currentTime.toFixed(2)}s` }}</label>
          <span>metres</span>
        </div>
        <div v-if="selectedKeyframe" class="position-grid">
          <label>X <input type="number" step="0.1" :value="selectedKeyframe.x.toFixed(2)" :disabled="saving" @change="emit('update-coordinate', 'x', ($event.target as HTMLInputElement).value)" /></label>
          <label>Z <input type="number" step="0.1" :value="selectedKeyframe.z.toFixed(2)" :disabled="saving" @change="emit('update-coordinate', 'z', ($event.target as HTMLInputElement).value)" /></label>
        </div>
        <small v-else>Add a keypoint at the playhead, or enable placement and click the 3D pitch. The first click creates the keypoint automatically.</small>
        <button class="wide-action" :disabled="saving || reconstructionRunning" @click="emit('add-keypoint', currentTime)">
          + Add keypoint at {{ currentTime.toFixed(2) }}s
        </button>
        <button class="wide-action" :class="{ active: placementMode }" :disabled="saving || reconstructionRunning" @click="emit('toggle-placement')">
          ◎ {{ placementMode ? 'Click the 3D pitch…' : 'Place ball on pitch' }}
        </button>
      </div>
    </template>

    <div class="quality-card ball-trajectory-summary">
      <div><span>Active source</span><strong>{{ mode.toUpperCase() }}</strong></div>
      <div><span>Manual keypoints</span><strong>{{ manualCount }}</strong></div>
      <div><span>Automatic samples</span><strong>{{ automaticCount }}</strong></div>
      <small v-if="mode === 'manual'">Positions are linearly interpolated between consecutive manual keypoints.</small>
      <small v-else>The latest detector result is active. Your manual keypoints remain stored.</small>
    </div>
  </div>
</template>
