<script setup lang="ts">
import type { ModelComparisonReport } from '../../types/analysis'

defineProps<{ comparison: ModelComparisonReport }>()
</script>

<template>
  <div class="model-comparison-card">
    <div class="model-comparison-heading">
      <div><span>Recognition benchmark</span><strong>{{ comparison.frameCount }} identical frames</strong></div>
      <i :class="comparison.comparison.verdict">
        {{ comparison.comparison.verdict === 'candidate' ? 'M LEADS' : comparison.comparison.verdict === 'baseline' ? 'N LEADS' : 'REVIEW' }}
      </i>
    </div>
    <div class="model-run-grid">
      <div>
        <b>{{ comparison.baseline.model.replace('.pt', '') }}</b><small>BASELINE</small>
        <strong>{{ comparison.baseline.meanDetectionsPerFrame }}</strong><span>people / frame</span>
        <dl>
          <div><dt>In pitch</dt><dd>{{ comparison.baseline.inPitchDetections }}</dd></div>
          <div><dt>Outside</dt><dd>{{ comparison.baseline.outsidePitchDetections }}</dd></div>
          <div><dt>Stable → accepted</dt><dd>{{ comparison.baseline.stableTrackCount }} → {{ comparison.baseline.acceptedTrackCount }}</dd></div>
          <div><dt>Boundary risk</dt><dd>{{ comparison.baseline.boundaryRiskTrackCount }}</dd></div>
          <div><dt>Inference</dt><dd>{{ comparison.baseline.inferenceSeconds.toFixed(1) }}s</dd></div>
        </dl>
      </div>
      <div class="candidate">
        <b>{{ comparison.candidate.model.replace('.pt', '') }}</b><small>CANDIDATE</small>
        <strong>{{ comparison.candidate.meanDetectionsPerFrame }}</strong><span>people / frame</span>
        <dl>
          <div><dt>In pitch</dt><dd>{{ comparison.candidate.inPitchDetections }}</dd></div>
          <div><dt>Outside</dt><dd>{{ comparison.candidate.outsidePitchDetections }}</dd></div>
          <div><dt>Stable → accepted</dt><dd>{{ comparison.candidate.stableTrackCount }} → {{ comparison.candidate.acceptedTrackCount }}</dd></div>
          <div><dt>Boundary risk</dt><dd>{{ comparison.candidate.boundaryRiskTrackCount }}</dd></div>
          <div><dt>Inference</dt><dd>{{ comparison.candidate.inferenceSeconds.toFixed(1) }}s</dd></div>
        </dl>
      </div>
    </div>
    <div class="model-comparison-deltas">
      <span><strong>{{ comparison.comparison.sharedDetections }}</strong> shared</span>
      <span><strong>+{{ comparison.comparison.candidateOnlyInPitchDetections }}</strong> M-only in field</span>
      <span><strong>{{ comparison.comparison.baselineOnlyInPitchDetections }}</strong> N-only in field</span>
    </div>
    <p>
      In-field delta <strong>{{ comparison.comparison.inPitchObservationGain >= 0 ? '+' : '' }}{{ comparison.comparison.inPitchObservationGain }}</strong>
      · outside delta <strong>{{ comparison.comparison.outsidePitchDetectionDelta >= 0 ? '+' : '' }}{{ comparison.comparison.outsidePitchDetectionDelta }}</strong>
      · stable tracks <strong>{{ comparison.comparison.stableTrackDelta >= 0 ? '+' : '' }}{{ comparison.comparison.stableTrackDelta }}</strong>
    </p>
    <small>{{ comparison.warnings[0] }}</small>
  </div>
</template>
