<script setup lang="ts">
import { computed } from 'vue'
import { calibrationRejectionReasonLabel } from '../lib/calibrationDiagnostics'
import {
  calibrationBallBackendSummary,
  calibrationFrameStatus as frameStatus,
  calibrationFrameStatusLabel as frameStatusLabel,
  calibrationFrameUncertainty as frameUncertainty,
  calibrationMotionLabel as motionLabel,
  calibrationObservationLabel as observationLabel,
  calibrationPersonSupport as personSupport,
  calibrationProcessingLabel as processingLabel,
  calibrationReportReasons,
  calibrationSolutionLabel as solutionLabel,
  calibrationTemporalAnchors as temporalAnchors,
  calibrationTemporalDirection as temporalDirection,
  calibrationTemporalGap as temporalGap,
  calibrationTemporalMotionConfidence as temporalMotionConfidence,
  calibrationVerdictLabel as verdictLabel,
  formatCalibrationNumber as number,
  formatCalibrationPercent as percent,
  isAmbiguousCalibrationFrame as isAmbiguous,
  isTemporalCalibrationProjection as isTemporalProjection,
  nearestCalibrationFrame,
  normalizeCalibrationGates,
} from '../features/calibration/calibrationQaPresentation'
import type { CalibrationEvidence, CalibrationFrameEvidence } from '../types/calibration'
import type { ProcessingStatus, QualityVerdict, ReconstructionQuality } from '../types/reconstruction'

type Side = 'left' | 'right' | 'unknown'

const props = defineProps<{
  processingStatus: ProcessingStatus
  qualityVerdict: QualityVerdict
  coordinateSpace?: string | null
  quality?: ReconstructionQuality | null
  calibration?: CalibrationEvidence | null
  ballDetection?: {
    status: string
    requestedBackend: string
    frameCount: number
    candidateCount: number
    framesWithCandidates: number
    fallbackFrameCount?: number
    failedFrameCount?: number
    observedFrameCount: number
    inferredFrameCount: number
    occludedFrameCount: number
    observedCoverage?: number | null
    publishedCoverage?: number | null
    backendCounts?: Record<string, number>
    frameSource?: {
      detectionCacheHit?: boolean
      detectionCacheStored?: boolean
      detectionCacheWriteError?: string
    }
  } | null
  ballDiagnostics?: {
    status?: string
    pathCostMargin?: number | null
    worldProjectionStatus?: string
    gaps?: {
      longestGapSeconds?: number | null
    }
  } | null
  frames?: CalibrationFrameEvidence[]
  currentTime: number
  visiblePitchSide: Side
  visiblePitchSideSource: string
  attackingGoal: Side
  directionSaving?: boolean
}>()

const emit = defineEmits<{
  seek: [sceneTime: number]
  calibrate: [sceneTime: number]
  changeAttackingGoal: [side: 'left' | 'right']
}>()

const evidenceFrames = computed(() => props.calibration?.frameEvidence ?? props.frames ?? [])
const selectedFrame = computed(() => nearestCalibrationFrame(evidenceFrames.value, props.currentTime))
const effectiveBallBackends = computed(() => calibrationBallBackendSummary(props.ballDetection?.backendCounts))
const reportReasons = computed(() => calibrationReportReasons(props.quality))
const normalizedGates = computed(() => normalizeCalibrationGates(props.quality))

function chooseAttackingGoal(event: Event) {
  const side = (event.target as HTMLSelectElement).value
  if (side === 'left' || side === 'right') emit('changeAttackingGoal', side)
}
</script>

<template>
  <section class="calibration-qa" aria-label="Calibration quality assurance">
    <header class="calibration-qa-header">
      <div>
        <span>CALIBRATION QA</span>
        <strong>Metric projection evidence</strong>
        <small>{{ coordinateSpace || 'coordinate space not declared' }}</small>
      </div>
      <div class="qa-status-pills">
        <i :class="`processing-${processingStatus}`">{{ processingLabel(processingStatus) }}</i>
        <i :class="`verdict-${qualityVerdict}`">{{ verdictLabel(qualityVerdict) }}</i>
      </div>
    </header>

    <div class="qa-orientation-grid">
      <label title="Observed from pitch markings. This is evidence about the camera view, not attack direction.">
        <span>Visible pitch side</span>
        <select :value="visiblePitchSide" disabled aria-label="Visible pitch side from calibration">
          <option value="unknown">Unknown</option>
          <option value="left">← Left goal visible</option>
          <option value="right">Right goal visible →</option>
        </select>
        <small>{{ visiblePitchSideSource }} evidence · read-only</small>
      </label>
      <label title="Match semantics selected by the editor. This never changes calibration or mirrors coordinates.">
        <span>Team attacks toward</span>
        <select :value="attackingGoal" :disabled="directionSaving" aria-label="Attacking goal direction" @change="chooseAttackingGoal">
          <option value="unknown" disabled>Choose direction</option>
          <option value="left">← Left goal</option>
          <option value="right">Right goal →</option>
        </select>
        <small>manual match meaning · independent</small>
      </label>
    </div>

    <section v-if="ballDetection || ballDiagnostics" class="qa-ball-card" aria-label="Ball trajectory quality">
      <div class="qa-section-title">
        <div><strong>Ball trajectory</strong><small>{{ ballDetection?.requestedBackend || 'backend unknown' }} · {{ ballDetection?.status || ballDiagnostics?.status || 'unknown' }}</small></div>
        <i :class="ballDiagnostics?.worldProjectionStatus === 'published' ? 'published' : 'withheld'">
          {{ ballDiagnostics?.worldProjectionStatus === 'published' ? '3D PUBLISHED' : '3D WITHHELD' }}
        </i>
      </div>
      <div class="qa-summary-grid qa-ball-grid">
        <div><span>Observed coverage</span><strong>{{ percent(ballDetection?.observedCoverage) }}</strong></div>
        <div><span>Published coverage</span><strong>{{ percent(ballDetection?.publishedCoverage) }}</strong></div>
        <div><span>Observed / inferred</span><strong>{{ ballDetection?.observedFrameCount ?? 0 }} / {{ ballDetection?.inferredFrameCount ?? 0 }}</strong></div>
        <div><span>Occluded</span><strong>{{ ballDetection?.occludedFrameCount ?? 0 }}</strong></div>
        <div><span>Candidate frames</span><strong>{{ ballDetection?.framesWithCandidates ?? 0 }} / {{ ballDetection?.frameCount ?? 0 }}</strong></div>
        <div><span>Candidates</span><strong>{{ ballDetection?.candidateCount ?? 0 }}</strong></div>
        <div><span>Fallback / failed</span><strong>{{ ballDetection?.fallbackFrameCount ?? 0 }} / {{ ballDetection?.failedFrameCount ?? 0 }}</strong></div>
        <div><span>Detection cache</span><strong>{{ ballDetection?.frameSource?.detectionCacheHit ? 'HIT' : ballDetection?.frameSource?.detectionCacheStored ? 'STORED' : ballDetection?.frameSource?.detectionCacheWriteError ? 'ERROR' : 'MISS' }}</strong></div>
        <div><span>Longest unresolved gap</span><strong>{{ number(ballDiagnostics?.gaps?.longestGapSeconds, 's') }}</strong></div>
        <div><span>Path margin</span><strong>{{ number(ballDiagnostics?.pathCostMargin) }}</strong></div>
      </div>
      <small class="qa-ball-backends">Effective: {{ effectiveBallBackends }}</small>
    </section>

    <template v-if="calibration">
      <div class="qa-summary-grid">
        <div><span>Direct coverage</span><strong>{{ percent(calibration.summary.directCoverage) }}</strong></div>
        <div><span>Usable coverage</span><strong>{{ percent(calibration.summary.usableCoverage) }}</strong></div>
        <div><span>Accepted</span><strong>{{ calibration.summary.acceptedFrameCount }} / {{ calibration.summary.sampledFrameCount }}</strong></div>
        <div><span>Rejected / missing</span><strong>{{ calibration.summary.rejectedFrameCount }} / {{ calibration.summary.missingFrameCount }}</strong></div>
        <div><span>Reprojection p50</span><strong>{{ number(calibration.summary.reprojectionP50, ' px') }}</strong></div>
        <div><span>Reprojection p95</span><strong>{{ number(calibration.summary.reprojectionP95, ' px') }}</strong></div>
        <div><span>Longest gap</span><strong>{{ number(calibration.summary.maxGapSeconds, 's') }}</strong></div>
        <div><span>Side agreement</span><strong>{{ percent(calibration.summary.sideAgreement) }}</strong></div>
        <div><span>Temporal recovered</span><strong>{{ calibration.summary.temporalRecoveredFrameCount ?? 0 }}</strong></div>
        <div><span>Temporal ambiguous</span><strong>{{ calibration.summary.temporalAmbiguousFrameCount ?? 0 }}</strong></div>
        <div><span>Temporal uncertainty p95</span><strong>{{ number(calibration.summary.temporalUncertaintyP95Metres, 'm') }}</strong></div>
        <div><span>Motion reliability</span><strong>{{ percent(calibration.summary.cameraMotionReliability) }}</strong></div>
      </div>

      <div class="qa-timeline-heading">
        <div><strong>Calibration frames</strong><small>click a sample to inspect the exact projection</small></div>
        <div class="qa-timeline-legend" aria-hidden="true">
          <i class="direct" /> direct
          <i class="recovered" /> temporal
          <i class="propagated" /> manual
          <i class="ambiguous" /> ambiguous
          <i class="rejected" /> rejected
          <i class="missing" /> missing
        </div>
      </div>
      <div class="qa-frame-timeline" role="list" aria-label="Per-frame calibration quality timeline">
        <button
          v-for="frame in evidenceFrames"
          :key="`${frame.sourceFrameIndex}-${frame.sampleIndex}`"
          role="listitem"
          :class="[
            frameStatus(frame),
            {
              current: selectedFrame?.sourceFrameIndex === frame.sourceFrameIndex,
              'motion-cut': frame.cameraMotion?.status === 'cut',
            },
          ]"
          :aria-label="`Frame ${frame.sourceFrameIndex}, ${frameStatusLabel(frame)}`"
          :title="`#${frame.sourceFrameIndex} · ${frame.sceneTime.toFixed(2)}s · ${frameStatusLabel(frame)} · motion ${frame.cameraMotion?.status || 'not recorded'}`"
          @click="emit('seek', frame.sceneTime)"
        />
      </div>

      <article v-if="selectedFrame" class="qa-frame-detail">
        <div class="qa-frame-heading">
          <div>
            <span>SELECTED SAMPLE</span>
            <strong>{{ selectedFrame.sceneTime.toFixed(2) }}s · source #{{ selectedFrame.sourceFrameIndex }}</strong>
          </div>
          <i :class="frameStatus(selectedFrame)">{{ frameStatusLabel(selectedFrame) }}</i>
        </div>
        <dl>
          <div><dt>Observation</dt><dd>{{ observationLabel(selectedFrame) }}</dd></div>
          <div><dt>Resolved solution</dt><dd>{{ solutionLabel(selectedFrame) }}</dd></div>
          <div><dt>Projection</dt><dd>{{ selectedFrame.projectionSource }}</dd></div>
          <div><dt>Backend</dt><dd>{{ selectedFrame.backend || selectedFrame.source || '—' }}</dd></div>
          <div v-if="selectedFrame.pnlcalibAttempts"><dt>PnLCalib attempts</dt><dd>{{ selectedFrame.pnlcalibAttempts.attemptCount }} / {{ selectedFrame.pnlcalibAttempts.maximumAttempts }} · accepted {{ selectedFrame.pnlcalibAttempts.acceptedAttempt ?? 'no' }}</dd></div>
          <div><dt>Confidence</dt><dd>{{ percent(selectedFrame.confidence) }}</dd></div>
          <div><dt>Reprojection p50 / p95</dt><dd>{{ number(selectedFrame.reprojectionError, ' px') }} / {{ number(selectedFrame.reprojectionP95, ' px') }}</dd></div>
          <div><dt>Keypoint ground p50 / p95</dt><dd>{{ number(selectedFrame.groundErrorP50Metres, ' m') }} / {{ number(selectedFrame.groundErrorP95Metres, ' m') }}</dd></div>
          <div><dt>Keypoints / inliers</dt><dd>{{ selectedFrame.keypointCount ?? '—' }} / {{ selectedFrame.inlierCount ?? '—' }}</dd></div>
          <div><dt>Inlier ratio</dt><dd>{{ percent(selectedFrame.inlierRatio) }}</dd></div>
          <div><dt>Visible side</dt><dd>{{ selectedFrame.visiblePitchSide || 'unknown' }}</dd></div>
          <div><dt>Person support</dt><dd>{{ personSupport(selectedFrame.personSupport) }}</dd></div>
          <template v-if="isTemporalProjection(selectedFrame.projectionSource) || selectedFrame.hypotheses?.length">
            <div><dt>Temporal direction</dt><dd>{{ temporalDirection(selectedFrame) }}</dd></div>
            <div><dt>Anchor source frame</dt><dd>{{ temporalAnchors(selectedFrame) }}</dd></div>
            <div><dt>Anchor distance</dt><dd>{{ number(temporalGap(selectedFrame), 's') }}</dd></div>
            <div><dt>Position uncertainty p95</dt><dd>{{ number(frameUncertainty(selectedFrame), 'm') }}</dd></div>
            <div><dt>Motion path confidence</dt><dd>{{ percent(temporalMotionConfidence(selectedFrame)) }}</dd></div>
            <div><dt>Ambiguity margin</dt><dd>{{ selectedFrame.ambiguityMargin == null ? '—' : selectedFrame.ambiguityMargin.toFixed(3) }}</dd></div>
          </template>
          <div><dt>Camera motion</dt><dd :class="`motion-${selectedFrame.cameraMotion?.status || 'missing'}`">{{ motionLabel(selectedFrame) }}</dd></div>
          <div><dt>Motion tracks / inliers</dt><dd>{{ selectedFrame.cameraMotion?.metrics?.trackedCount ?? '—' }} / {{ selectedFrame.cameraMotion?.metrics?.inlierCount ?? '—' }}</dd></div>
          <div><dt>Motion residual p95</dt><dd>{{ number(selectedFrame.cameraMotion?.metrics?.residualP95Px, ' px') }}</dd></div>
          <div><dt>Motion coverage</dt><dd>{{ percent(selectedFrame.cameraMotion?.metrics?.coverageRatio) }}</dd></div>
        </dl>

        <section v-if="selectedFrame.hypotheses?.length" class="qa-hypotheses" aria-label="Ranked camera calibration hypotheses">
          <div class="qa-hypotheses-heading">
            <strong>Camera hypotheses</strong>
            <small v-if="isAmbiguous(selectedFrame)">conflict — metric projection withheld</small>
            <small v-else>ranked by direct evidence, motion and temporal distance</small>
          </div>
          <ol>
            <li
              v-for="hypothesis in selectedFrame.hypotheses"
              :key="hypothesis.id"
              :class="{
                selected: hypothesis.selected || hypothesis.id === selectedFrame.selectedHypothesisId,
                ambiguous: isAmbiguous(selectedFrame),
              }"
            >
              <span class="qa-hypothesis-rank">#{{ hypothesis.rank }}</span>
              <span class="qa-hypothesis-main">
                <strong>{{ hypothesis.origin }}</strong>
                <small>
                  anchor {{ hypothesis.anchorFrameIndices.map((index) => `#${index}`).join(' + ') || '—' }}
                  · {{ number(hypothesis.temporalDistanceSeconds, 's') }}
                  · ±{{ number(hypothesis.uncertaintyP95Metres, 'm') }} p95
                </small>
              </span>
              <span class="qa-hypothesis-score">
                <b>{{ percent(hypothesis.score) }}</b>
                <small v-if="hypothesis.selected || hypothesis.id === selectedFrame.selectedHypothesisId">SELECTED</small>
                <small v-else-if="isAmbiguous(selectedFrame)">CONFLICT</small>
                <small v-else>ALTERNATE</small>
              </span>
              <p v-if="hypothesis.disagreementMetres != null || hypothesis.rejectionReasons?.length">
                <span v-if="hypothesis.disagreementMetres != null">candidate disagreement {{ number(hypothesis.disagreementMetres, 'm') }}</span>
                <span v-for="reason in hypothesis.rejectionReasons" :key="reason">{{ calibrationRejectionReasonLabel(reason) }}</span>
              </p>
            </li>
          </ol>
        </section>
        <p v-if="selectedFrame.rejectionReasons?.length" class="qa-reasons">
          <span v-for="reason in selectedFrame.rejectionReasons" :key="reason">{{ calibrationRejectionReasonLabel(reason) }}</span>
        </p>
        <p v-else-if="selectedFrame.status === 'accepted'" class="qa-evidence-note">
          The video overlay uses this sample's stored homography. Reprojection is a frame-local fit, not a temporal-continuity score. Recovered samples name their anchor and uncertainty; motion cuts are hard temporal boundaries.
        </p>
        <button class="qa-calibrate-button" @click="emit('calibrate', selectedFrame.sceneTime)">Adjust this frame manually</button>
      </article>
    </template>

    <div v-else class="qa-no-evidence">
      <strong>Per-frame evidence was not recorded for this run.</strong>
      <p>This run cannot prove calibration coverage or temporal stability. Rebuild the scene to generate reviewable evidence.</p>
    </div>

    <div v-if="normalizedGates.length" class="qa-gates">
      <div class="qa-section-title"><strong>Quality gates</strong><small>completion and acceptance are independent</small></div>
      <div v-for="gate in normalizedGates" :key="gate.id" class="qa-gate" :class="gate.status">
        <i>{{ gate.status === 'pass' ? '✓' : gate.status === 'fail' ? '!' : gate.status === 'review' ? '△' : '—' }}</i>
        <span><strong>{{ gate.label }}</strong><small>{{ gate.detail || gate.threshold || 'No threshold evidence' }}</small></span>
        <b>{{ gate.value || gate.status.toUpperCase() }}</b>
      </div>
    </div>

    <p v-if="reportReasons.length" class="qa-report-reasons">
      <span v-for="reason in reportReasons" :key="reason">{{ reason }}</span>
    </p>
  </section>
</template>

<style scoped>
.calibration-qa {
  container: calibration-qa / inline-size;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 18px;
  color: #cbd2ce;
  font-size: 12px;
  line-height: 1.45;
}
.calibration-qa-header, .qa-frame-heading, .qa-section-title, .qa-timeline-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; }
.calibration-qa-header > div:first-child, .qa-frame-heading > div, .qa-timeline-heading > div:first-child { min-width: 0; display: flex; flex-direction: column; gap: 5px; }
.calibration-qa-header span, .qa-frame-heading span, .qa-orientation-grid label > span { color: #8a9590; font: 600 11px/1.3 'DM Mono'; letter-spacing: .09em; text-transform: uppercase; }
.calibration-qa-header strong { color: #f0f3ef; font-size: 16px; line-height: 1.25; }
.calibration-qa-header small { color: #84908a; font: 500 11px/1.4 'DM Mono'; overflow-wrap: anywhere; }
.qa-status-pills { display: flex; flex: 0 0 auto; flex-direction: column; align-items: flex-end; gap: 6px; }
.qa-status-pills i, .qa-frame-heading > i { padding: 5px 7px; border: 1px solid currentColor; color: #8a9690; font: 600 11px/1.2 'DM Mono'; font-style: normal; white-space: nowrap; }
.qa-status-pills .processing-completed, .qa-status-pills .verdict-pass { color: #71e2aa; }
.qa-status-pills .processing-processing, .qa-status-pills .processing-queued, .qa-status-pills .verdict-review, .qa-status-pills .verdict-pending { color: var(--accent); }
.qa-status-pills .processing-failed, .qa-status-pills .verdict-reject { color: #ff7867; }
.qa-status-pills .verdict-unknown { color: #929c97; }
.qa-ball-card { display: flex; flex-direction: column; gap: 10px; padding: 12px; border: 1px solid rgba(101,191,255,.25); background: rgba(101,191,255,.025); }
.qa-ball-card .qa-section-title > div { display: flex; min-width: 0; flex-direction: column; gap: 4px; }
.qa-ball-card .qa-section-title > i { flex: 0 0 auto; padding: 4px 6px; border: 1px solid currentColor; color: #ff7867; font: 600 10px/1.2 'DM Mono'; font-style: normal; }
.qa-ball-card .qa-section-title > i.published { color: #71e2aa; }
.qa-ball-card .qa-section-title > i.withheld { color: var(--accent); }
.qa-ball-backends { color: #84908a; font: 500 10px/1.45 'DM Mono'; overflow-wrap: anywhere; }
.qa-ball-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.qa-orientation-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px; }
.qa-orientation-grid label { min-width: 0; display: flex; flex-direction: column; gap: 8px; padding: 12px; border: 1px solid var(--line); background: rgba(255,255,255,.018); }
.qa-orientation-grid select { min-width: 0; width: 100%; height: 40px; border: 1px solid var(--line-strong); border-radius: 3px; background: #090c0c; color: #e2e7e3; padding: 0 10px; font: 500 12px 'DM Mono'; }
.qa-orientation-grid select:disabled { opacity: .78; }
.qa-orientation-grid small { color: #818c86; font: 500 11px/1.45 'DM Mono'; }
.qa-summary-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); border: 1px solid var(--line); }
.qa-summary-grid > div { min-width: 0; min-height: 68px; padding: 11px 12px; display: flex; flex-direction: column; justify-content: center; gap: 7px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.qa-summary-grid > div:nth-child(2n) { border-right: 0; }
.qa-summary-grid > div:nth-last-child(-n+2) { border-bottom: 0; }
.qa-summary-grid span { color: #8b9690; font: 500 11px/1.3 'DM Mono'; overflow-wrap: anywhere; }
.qa-summary-grid strong { color: #eef2ee; font: 600 14px/1.25 'DM Mono'; }
.qa-timeline-heading { flex-direction: column; }
.qa-timeline-heading strong, .qa-section-title strong { color: #d0d7d2; font: 600 12px/1.3 'DM Mono'; text-transform: uppercase; letter-spacing: .07em; }
.qa-timeline-heading small, .qa-section-title small { color: #838e88; font-size: 11px; line-height: 1.45; }
.qa-timeline-legend { display: flex; align-items: center; justify-content: flex-start; flex-wrap: wrap; gap: 7px 9px; color: #89938e; font: 500 11px/1.3 'DM Mono'; }
.qa-timeline-legend i { width: 11px; height: 11px; margin-left: 4px; border-radius: 2px; }
.qa-timeline-legend i:first-child { margin-left: 0; }
.qa-timeline-legend .direct, .qa-frame-timeline .direct { background: #71e2aa; }
.qa-timeline-legend .recovered, .qa-frame-timeline .recovered { background: #65bfff; }
.qa-timeline-legend .propagated, .qa-frame-timeline .propagated { background: var(--accent); }
.qa-timeline-legend .ambiguous, .qa-frame-timeline .ambiguous { background: #ba83ff; }
.qa-timeline-legend .rejected, .qa-frame-timeline .rejected { background: #ff6857; }
.qa-timeline-legend .missing, .qa-frame-timeline .missing { background: #414845; }
.qa-frame-timeline { min-height: 48px; display: flex; align-items: stretch; gap: 4px; padding: 7px; overflow-x: auto; overscroll-behavior-x: contain; scrollbar-color: #626d67 #111514; scrollbar-width: thin; border: 1px solid var(--line-strong); background: #090c0c; }
.qa-frame-timeline button { position: relative; flex: 0 0 16px; width: 16px; min-width: 16px; min-height: 32px; padding: 0; border: 0; border-radius: 2px; opacity: .72; cursor: pointer; transition: opacity .12s ease, filter .12s ease, transform .12s ease; }
.qa-frame-timeline button.motion-cut::after { content: ''; position: absolute; inset: -7px auto -7px -3px; width: 2px; background: #ff7867; box-shadow: 0 0 5px rgba(255,120,103,.8); }
.qa-frame-timeline button:hover { opacity: 1; filter: brightness(1.35); }
.qa-frame-timeline button:focus-visible { opacity: 1; outline: 3px solid #fff4cc; outline-offset: 2px; z-index: 2; }
.qa-frame-timeline button.current { opacity: 1; outline: 3px solid #fff4cc; outline-offset: 2px; transform: scaleY(1.06); z-index: 1; }
.qa-frame-detail { padding: 14px; border: 1px solid rgba(255,211,106,.3); background: rgba(255,211,106,.035); }
.qa-frame-heading { margin-bottom: 12px; }
.qa-frame-heading strong { color: #edf1ed; font: 600 14px/1.3 'DM Mono'; }
.qa-frame-heading > i.direct { color: #71e2aa; }
.qa-frame-heading > i.recovered { color: #65bfff; }
.qa-frame-heading > i.propagated { color: var(--accent); }
.qa-frame-heading > i.ambiguous { color: #ba83ff; }
.qa-frame-heading > i.rejected { color: #ff7867; }
.qa-frame-heading > i.missing { color: #929b97; }
.qa-frame-detail dl, .qa-no-evidence dl { margin: 0; display: grid; grid-template-columns: minmax(0, 1fr); }
.qa-frame-detail dl > div, .qa-no-evidence dl > div { min-width: 0; min-height: 43px; padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,.07); display: grid; grid-template-columns: minmax(110px, 42%) minmax(0, 1fr); align-items: baseline; gap: 10px; }
.qa-frame-detail dt, .qa-no-evidence dt { color: #87928c; font: 500 11px/1.4 'DM Mono'; }
.qa-frame-detail dd, .qa-no-evidence dd { min-width: 0; margin: 0; color: #dbe1dd; font: 500 13px/1.4 'DM Mono'; overflow-wrap: anywhere; }
.qa-frame-detail dd.motion-estimated { color: #71e2aa; }
.qa-frame-detail dd.motion-unreliable, .qa-frame-detail dd.motion-unestimated { color: var(--accent); }
.qa-frame-detail dd.motion-cut { color: #ff7867; }
.qa-hypotheses { margin-top: 14px; padding-top: 13px; border-top: 1px solid rgba(255,255,255,.09); }
.qa-hypotheses-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 9px; }
.qa-hypotheses-heading strong { color: #c5cdc8; font: 600 12px/1.3 'DM Mono'; letter-spacing: .07em; text-transform: uppercase; }
.qa-hypotheses-heading small { color: #85908a; font: 500 11px/1.4 'DM Mono'; text-align: right; }
.qa-hypotheses ol { display: flex; flex-direction: column; gap: 7px; margin: 0; padding: 0; list-style: none; }
.qa-hypotheses li { display: grid; grid-template-columns: 28px minmax(0, 1fr) auto; align-items: center; gap: 9px; min-width: 0; padding: 10px; border: 1px solid var(--line); background: rgba(255,255,255,.012); }
.qa-hypotheses li.selected { border-color: rgba(101,191,255,.4); background: rgba(101,191,255,.045); }
.qa-hypotheses li.ambiguous:not(.selected) { border-color: rgba(186,131,255,.22); }
.qa-hypothesis-rank { color: #89948e; font: 600 11px 'DM Mono'; }
.qa-hypothesis-main, .qa-hypothesis-score { min-width: 0; display: flex; flex-direction: column; gap: 4px; }
.qa-hypothesis-main strong { color: #dfe5e1; font: 600 12px/1.35 'DM Mono'; overflow-wrap: anywhere; }
.qa-hypothesis-main small { color: #86918b; font: 500 11px/1.45 'DM Mono'; overflow-wrap: anywhere; }
.qa-hypothesis-score { align-items: flex-end; }
.qa-hypothesis-score b { color: #e2e8e4; font: 600 14px 'DM Mono'; }
.qa-hypothesis-score small { color: #8b9690; font: 600 11px 'DM Mono'; letter-spacing: .05em; }
.qa-hypotheses li.selected .qa-hypothesis-score small { color: #65bfff; }
.qa-hypotheses li.ambiguous .qa-hypothesis-score small { color: #ba83ff; }
.qa-hypotheses li > p { grid-column: 2 / -1; display: flex; flex-direction: column; gap: 4px; margin: 0; color: #b698cb; font-size: 11px; line-height: 1.45; }
.qa-reasons, .qa-report-reasons { display: flex; flex-direction: column; gap: 7px; margin: 12px 0 0; }
.qa-reasons span, .qa-report-reasons span { padding: 3px 0 3px 10px; border-left: 3px solid #ff7867; color: #e2aaa1; font-size: 11px; line-height: 1.5; overflow-wrap: anywhere; }
.qa-evidence-note { margin: 12px 0 0; color: #919c96; font-size: 11px; line-height: 1.55; }
.qa-calibrate-button { width: 100%; min-height: 44px; margin-top: 12px; border: 1px solid rgba(255,211,106,.42); border-radius: 3px; background: rgba(255,211,106,.07); color: var(--accent); font: 600 12px 'DM Mono'; cursor: pointer; }
.qa-calibrate-button:hover { border-color: var(--accent); }
.qa-no-evidence { padding: 14px; border: 1px solid rgba(255,211,106,.28); background: rgba(255,211,106,.04); }
.qa-no-evidence > strong { color: #f0e0ad; font-size: 14px; line-height: 1.4; }
.qa-no-evidence > p { margin: 9px 0 13px; color: #afa58a; font-size: 12px; line-height: 1.55; }
.qa-gates { display: flex; flex-direction: column; gap: 8px; }
.qa-gate { min-height: 58px; display: grid; grid-template-columns: 28px minmax(0, 1fr) auto; gap: 10px; align-items: center; padding: 10px; border: 1px solid var(--line); }
.qa-gate > i { width: 27px; height: 27px; display: grid; place-items: center; border: 1px solid currentColor; border-radius: 50%; color: #87918c; font: 700 12px 'DM Mono'; font-style: normal; }
.qa-gate > span { min-width: 0; display: flex; flex-direction: column; gap: 4px; }
.qa-gate > span strong { color: #d2d8d4; font-size: 12px; line-height: 1.35; }
.qa-gate > span small { color: #838e88; font-size: 11px; line-height: 1.45; overflow-wrap: anywhere; }
.qa-gate > b { color: #b7c0bb; font: 600 13px 'DM Mono'; text-align: right; }
.qa-gate.pass { border-color: rgba(113,226,170,.2); }
.qa-gate.pass > i, .qa-gate.pass > b { color: #71e2aa; }
.qa-gate.review { border-color: rgba(255,211,106,.25); }
.qa-gate.review > i, .qa-gate.review > b { color: var(--accent); }
.qa-gate.fail { border-color: rgba(255,104,87,.3); background: rgba(255,104,87,.025); }
.qa-gate.fail > i, .qa-gate.fail > b { color: #ff7867; }
@container calibration-qa (min-width: 560px) {
  .qa-orientation-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .qa-summary-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .qa-summary-grid > div:nth-child(2n) { border-right: 1px solid var(--line); }
  .qa-summary-grid > div:nth-child(3n) { border-right: 0; }
  .qa-summary-grid > div:nth-last-child(-n+3) { border-bottom: 0; }
  .qa-frame-detail dl, .qa-no-evidence dl { grid-template-columns: repeat(2, minmax(0, 1fr)); column-gap: 18px; }
}
</style>
