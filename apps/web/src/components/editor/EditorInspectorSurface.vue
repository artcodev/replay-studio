<script setup lang="ts">
import CalibrationQaPanel from '../CalibrationQaPanel.vue'
import IdentityReviewPanel from '../IdentityReviewPanel.vue'
import TrackPresenceCard from '../TrackPresenceCard.vue'
import ManualBallInspector from './ManualBallInspector.vue'
import FrameAnalysisInspector from './FrameAnalysisInspector.vue'
import ModelComparisonCard from './ModelComparisonCard.vue'
import PlayerTrackInspector from './PlayerTrackInspector.vue'
import TrackProjectionDebugCard from './TrackProjectionDebugCard.vue'
import { frameMetricBadge } from '../../lib/videoTrackSelection'
import { identityValidationSummary } from '../../lib/reconstructionUi'
import type {
  IdentityReviewCandidateDecision,
  IdentityReviewInspectFrame,
  IdentityReviewObservation,
  IdentityReviewWorkerState,
} from '../../lib/identityReview'
import type { useFrameAnalysis } from '../../composables/useFrameAnalysis'
import type { useFrameAnnotations } from '../../composables/useFrameAnnotations'
import type { useIdentityReviewEditor } from '../../composables/useIdentityReviewEditor'
import type { useManualBallEditor } from '../../composables/useManualBallEditor'
import type { usePitchCalibrationEditor } from '../../composables/usePitchCalibrationEditor'
import type { useReconstructionController } from '../../composables/useReconstructionController'
import type { useSegmentLayoutEditor } from '../../composables/useSegmentLayoutEditor'
import type { useCompositionEditor } from '../../composables/useCompositionEditor'
import type { CalibrationEvidence, CalibrationFrameEvidence } from '../../types/calibration'
import type { CanonicalPerson } from '../../types/identity'
import type { FrameAnalysis, ModelComparisonReport } from '../../types/analysis'
import type { MultiPassSummary } from '../../types/media'
import type { SceneDocument, SceneVideoAsset, Team } from '../../types/scene'
import type { Track } from '../../types/tracking'
import type { CanonicalMatch, CanonicalMatchEvent, ExternalPlayer } from '../../types/match'

type Controllers = {
  manualBall: ReturnType<typeof useManualBallEditor>
  identityReview: ReturnType<typeof useIdentityReviewEditor>
  frameAnalysis: ReturnType<typeof useFrameAnalysis>
  frameAnnotations: ReturnType<typeof useFrameAnnotations>
  reconstruction: ReturnType<typeof useReconstructionController>
  pitchCalibration: ReturnType<typeof usePitchCalibrationEditor>
  segmentLayout: ReturnType<typeof useSegmentLayoutEditor>
  composition: ReturnType<typeof useCompositionEditor>
}

type SelectionView = {
  track: Track | null
  canonicalPerson: CanonicalPerson | null
  identityPerson: CanonicalPerson | null
  identityObservations: IdentityReviewObservation[] | null
  identityWorkers: IdentityReviewWorkerState[] | null
  dedicatedUnbindActive: boolean
  framePersonId: string | null
  team: Team | null
}

type AnalysisView = {
  sceneVideo: SceneVideoAsset | null
  calibrationEvidence: CalibrationEvidence | null
  calibrationFrames: CalibrationFrameEvidence[]
  calibrationLabel: string
  analysisFrameCount: number
  modelComparison: ModelComparisonReport | null
  multiPass: MultiPassSummary | null
}

type MatchView = {
  match: CanonicalMatch | null
  rosterPlayers: ExternalPlayer[]
  refreshAvailable: boolean
  refreshing: boolean
}

type Commands = {
  loadIdentityReview: (sceneId: string) => void | Promise<void>
  confirmRoster: (payload: { canonicalPersonId: string; externalPlayerId: string }) => void | Promise<void>
  rejectCandidate: (payload: IdentityReviewCandidateDecision) => void | Promise<void>
  inspectIdentityFrame: (payload: IdentityReviewInspectFrame) => void
  unbindRoster: (payload: { canonicalPersonId: string }) => void | Promise<void>
  clearRosterBinding: (payload: { canonicalPersonId: string }) => void | Promise<void>
  updateTrackLabel: (value: string) => void
  updateTrackNumber: (value: number) => void
  updateTrackPosition: (axis: 'x' | 'z', value: string) => void
  framePersonLabel: (person: FrameAnalysis['people'][number]) => string
  framePersonCanonicalId: (person: FrameAnalysis['people'][number]) => string | null
  selectDetectedPerson: (person: FrameAnalysis['people'][number]) => void
  seek: (time: number) => void
  refreshMatch: () => void | Promise<void>
  openMatchWorkspace: () => void | Promise<void>
  addEvent: (event: CanonicalMatchEvent) => void
  removeEvent: (index: number) => void
  reconstruct: () => void | Promise<void>
  rebuildLayout: () => void | Promise<void>
  confirmLayout: () => void | Promise<void>
  splitSelection: () => void | Promise<void>
  importClip: () => void
  passRelationLabel: (relation?: string) => string
}

const props = defineProps<{
  scene: SceneDocument
  currentTime: number
  saving: boolean
  editMode: boolean
  selection: SelectionView
  analysis: AnalysisView
  matchView: MatchView
  controllers: Controllers
  commands: Commands
}>()

const activeTab = defineModel<'binding' | 'qa' | 'events'>('activeTab', { required: true })
const emit = defineEmits<{
  'update:currentTime': [value: number]
  'update:editMode': [value: boolean]
}>()

const {
  manualBall,
  identityReview,
  frameAnalysis,
  frameAnnotations,
  reconstruction,
  pitchCalibration,
  segmentLayout,
  composition,
} = props.controllers
</script>

<template>
  <aside class="panel inspector-panel">
    <div class="inspector-tabs">
      <button :class="{ active: activeTab === 'binding' }" @click="activeTab = 'binding'">Inspector</button>
      <button v-if="analysis.sceneVideo?.reconstruction" :class="[`verdict-${reconstruction.qualityVerdict.value}`, { active: activeTab === 'qa' }]" @click="activeTab = 'qa'">Quality <span>{{ reconstruction.qualityVerdict.value === 'unknown' ? '?' : reconstruction.qualityVerdict.value.toUpperCase() }}</span></button>
      <button :class="{ active: activeTab === 'events' }" @click="activeTab = 'events'">Events <span>{{ scene.payload.eventBindings.length }}</span></button>
    </div>

    <div v-if="activeTab === 'binding' && (manualBall.selected.value || selection.canonicalPerson || selection.track || frameAnalysis.analysis.value || analysis.modelComparison || analysis.multiPass)" class="inspector-body">
      <ManualBallInspector
        v-if="manualBall.selected.value"
        :mode="manualBall.mode.value"
        :saving="manualBall.saving.value"
        :reconstruction-running="reconstruction.running.value"
        :selected-keyframe="manualBall.selectedKeyframe.value"
        :current-time="currentTime"
        :placement-mode="manualBall.placementMode.value"
        :manual-count="manualBall.manualKeyframes.value.length"
        :automatic-count="manualBall.automaticKeyframes.value.length"
        @change-mode="manualBall.changeMode"
        @update-coordinate="manualBall.updateCoordinate"
        @add-keypoint="manualBall.addKeypoint"
        @toggle-placement="manualBall.togglePlacement"
      />
      <div v-if="selection.identityPerson" class="canonical-identity-panel">
        <div v-if="identityReview.loading.value" class="identity-review-load-state" role="status">Loading identity crops and worker readiness…</div>
        <div v-else-if="identityReview.error.value" class="identity-review-load-state error" role="alert">
          <span>{{ identityReview.error.value }}</span>
          <button type="button" @click="commands.loadIdentityReview(scene.id)">Retry identity review</button>
        </div>
        <IdentityReviewPanel
          :identity="selection.identityPerson"
          :roster-players="matchView.rosterPlayers"
          :observations="selection.identityObservations"
          :worker-states="selection.identityWorkers"
          :dedicated-unbind-active="selection.dedicatedUnbindActive"
          :disabled="saving || frameAnnotations.saving.value || identityReview.rosterBindingSaving.value || identityReview.decisionSaving.value || reconstruction.reconstructing.value || reconstruction.running.value"
          @bind-candidate="commands.confirmRoster"
          @reject-candidate="commands.rejectCandidate"
          @inspect-frame="commands.inspectIdentityFrame"
          @unbind-roster="commands.unbindRoster"
          @clear-roster-binding="commands.clearRosterBinding"
        />
        <div v-if="!selection.track" class="identity-projection-note" role="status">
          <strong>Not projected in 3D</strong>
          <small>The video identity is preserved, but no trajectory passed metric projection QA for this person.</small>
        </div>
      </div>
      <PlayerTrackInspector
        v-if="selection.track"
        :edit-mode="editMode"
        :track="selection.track"
        :team-name="selection.team?.name ?? null"
        :current-time="currentTime"
        :processing-status="reconstruction.processingStatus.value"
        :quality-verdict="reconstruction.qualityVerdict.value"
        :calibration-label="analysis.calibrationLabel"
        :visible-pitch-side="pitchCalibration.visiblePitchSide.value"
        :attacking-goal-side="pitchCalibration.attackingGoalSide.value"
        :pitch-calibration-status="pitchCalibration.pitchCalibration.value?.status ?? null"
        :pitch-calibration-supported-lines="pitchCalibration.pitchCalibration.value?.supportedLines ?? null"
        :pitch-calibration-rectangle="pitchCalibration.pitchCalibration.value?.rectangle ?? null"
        :pitch-calibration-reason="pitchCalibration.pitchCalibration.value?.reason ?? null"
        :scene-video="analysis.sceneVideo"
        :analysis-frame-count="analysis.analysisFrameCount"
        :identity-validation-label="identityValidationSummary(analysis.sceneVideo?.reconstruction?.quality?.identityValidation)"
        @update:edit-mode="emit('update:editMode', $event)"
        @update-label="commands.updateTrackLabel"
        @update-number="commands.updateTrackNumber"
        @update-position="commands.updateTrackPosition"
      >
        <template #presence><TrackPresenceCard :track="selection.track" :current-time="currentTime" /></template>
      </PlayerTrackInspector>
      <TrackProjectionDebugCard
        v-if="selection.track?.observations?.length || selection.canonicalPerson?.observations?.length"
        :label="selection.track?.label ?? selection.canonicalPerson?.displayName ?? 'Selected identity'"
        :observations="selection.track?.observations ?? selection.canonicalPerson?.observations"
        :tracks="scene.payload.tracks"
        :calibration-frames="analysis.calibrationFrames"
        :pitch="scene.payload.pitch"
        :current-time="currentTime"
        :contact-point-profile="scene.payload.videoAsset?.reconstruction?.contactPointProfile ?? 'bbox-bottom'"
        :calibration-busy="pitchCalibration.loading.value || pitchCalibration.applying.value || reconstruction.running.value"
        @seek="commands.seek"
        @recalibrate="pitchCalibration.calibrateQaFrame"
      />
      <FrameAnalysisInspector
        v-if="frameAnalysis.analysis.value"
        :analysis="frameAnalysis.analysis.value"
        :active="Boolean(frameAnalysis.activeAnalysis.value)"
        :annotation-mode="frameAnnotations.mode.value"
        :draft="frameAnnotations.draft.value"
        :identity-actions="frameAnnotations.actions"
        :annotation-kinds="frameAnnotations.kinds"
        :merge-targets="frameAnnotations.mergeTargets.value"
        :split-preview="frameAnnotations.splitPreview.value"
        :scene-duration="scene.duration"
        :selected-person-id="selection.framePersonId"
        :saving="frameAnnotations.saving.value"
        :reconstructing="reconstruction.reconstructing.value"
        :reconstruction-status="reconstruction.status.value"
        :save-disabled="frameAnnotations.saveDisabled.value"
        :person-label="commands.framePersonLabel"
        :person-canonical-id="commands.framePersonCanonicalId"
        :metric-badge="frameMetricBadge"
        @toggle-mode="frameAnnotations.toggleMode"
        @action-change="frameAnnotations.onActionChange"
        @delete="frameAnnotations.remove"
        @save="frameAnnotations.save"
        @select-person="commands.selectDetectedPerson($event); commands.seek(frameAnalysis.analysis.value!.sceneTime)"
      />
      <ModelComparisonCard v-if="analysis.modelComparison" :comparison="analysis.modelComparison" />
      <div v-if="analysis.multiPass" class="multi-pass-card">
        <div><span>Reconstruction evidence</span><strong>{{ Math.round((analysis.multiPass.consensus?.evidenceScore ?? 0) * 100) }}%</strong></div>
        <small>{{ analysis.multiPass.consensus?.passesAnalyzed ?? 0 }} passes · {{ analysis.multiPass.consensus?.metricPasses ?? 0 }} metric · {{ analysis.multiPass.consensus?.ballPasses ?? 0 }} with ball</small>
        <small v-if="analysis.multiPass.ballSupport" class="aligned-support">{{ analysis.multiPass.consensus?.overlappingPasses ?? 0 }} aligned replay · {{ analysis.multiPass.ballSupport.supportedSamples }}/{{ analysis.multiPass.ballSupport.referenceSamples }} ball samples supported</small>
        <div class="pass-list">
          <span v-for="item in analysis.multiPass.passes" :key="item.segmentId" :class="{ reference: item.sceneId === analysis.multiPass.referenceSceneId }">
            {{ item.label }} · {{ Math.round(item.quality * 100) }}% <i>{{ commands.passRelationLabel(item.relation).toUpperCase() }} · QA {{ item.qualityVerdict.toUpperCase() }}</i>
          </span>
        </div>
        <small class="evidence-note">Motion alignment verifies overlapping replays. Continuation shots extend the event but are not fused into the reference trajectories.</small>
      </div>
    </div>

    <div v-else-if="activeTab === 'qa'" class="inspector-body calibration-qa-body">
      <CalibrationQaPanel
        :processing-status="reconstruction.processingStatus.value"
        :quality-verdict="reconstruction.qualityVerdict.value"
        :coordinate-space="analysis.sceneVideo?.reconstruction?.coordinateSpace ?? null"
        :quality="analysis.sceneVideo?.reconstruction?.quality ?? null"
        :calibration="analysis.calibrationEvidence"
        :ball-detection="analysis.sceneVideo?.reconstruction?.ballDetection ?? null"
        :ball-diagnostics="scene.payload.ball.diagnostics ?? null"
        :frames="analysis.calibrationFrames"
        :current-time="currentTime"
        :visible-pitch-side="pitchCalibration.visiblePitchSide.value"
        :visible-pitch-side-source="pitchCalibration.visiblePitchSideSource.value"
        :attacking-goal="pitchCalibration.attackingGoalSide.value"
        :direction-saving="pitchCalibration.pitchSideSaving.value || Boolean(pitchCalibration.draft.value) || reconstruction.running.value"
        @seek="commands.seek"
        @calibrate="pitchCalibration.calibrateQaFrame"
        @change-attacking-goal="pitchCalibration.changeAttackingGoal"
      />
    </div>

    <div v-else-if="activeTab === 'events'" class="inspector-body events-body">
      <div v-if="matchView.match" class="bound-match">
        <p>Project match</p>
        <strong>{{ matchView.match.homeTeam.name }} {{ matchView.match.score.home ?? '–' }} — {{ matchView.match.score.away ?? '–' }} {{ matchView.match.awayTeam.name }}</strong>
        <small>{{ matchView.match.kickoffAt || 'Kickoff unknown' }} · {{ matchView.match.competition || 'Competition unknown' }} · {{ matchView.match.status || 'status unknown' }}</small>
        <button v-if="matchView.refreshAvailable" type="button" :disabled="matchView.refreshing || reconstruction.mutationLocked.value" @click="commands.refreshMatch">{{ matchView.refreshing ? 'Refreshing…' : 'Refresh project snapshot' }}</button>
      </div>
      <button v-else class="empty-bind" @click="commands.openMatchWorkspace">＋ Set project match data</button>
      <div v-if="matchView.match?.events.length" class="source-events">
        <p class="section-label">Match timeline</p>
        <button v-for="item in matchView.match.events" :key="item.id" @click="commands.addEvent(item)"><span>{{ item.minute ?? '—' }}′</span><strong>{{ item.label }}</strong><i>＋</i></button>
      </div>
      <div v-if="matchView.match?.sync.warnings.length" class="source-warnings"><p v-for="warning in matchView.match.sync.warnings" :key="warning">{{ warning }}</p></div>
      <div class="scene-events">
        <p class="section-label">Scene markers</p>
        <div v-for="(item, index) in scene.payload.eventBindings" :key="`${item.externalEventId}-${index}`">
          <button class="marker-time" @click="emit('update:currentTime', item.sceneTime)">{{ item.sceneTime.toFixed(2) }}s</button>
          <span>{{ item.label }}</span><button class="remove" @click="commands.removeEvent(index)">×</button>
        </div>
        <small v-if="!scene.payload.eventBindings.length">Add source events at the current playhead position.</small>
      </div>
    </div>

    <div v-else class="inspector-body empty-reconstruction">
      <span>{{ reconstruction.running.value ? `AI ANALYSIS · ${reconstruction.progress.value?.overallPercent ?? 0}%` : reconstruction.status.value === 'failed' ? 'ANALYSIS NEEDS REVIEW' : segmentLayout.layout.value ? 'EVENT MAP READY' : 'FRAME SET READY' }}</span>
      <h2>{{ reconstruction.running.value ? reconstruction.progress.value?.label ?? (analysis.multiPass ? `Analyzing ${analysis.multiPass.selectedSegmentIds.length} camera angles…` : 'Preparing analysis…') : segmentLayout.layout.value ? 'Review detected events' : 'No reconstructed tracks yet' }}</h2>
      <p v-if="reconstruction.running.value">{{ reconstruction.progress.value?.detail ?? (analysis.multiPass ? `Pass ${analysis.multiPass.currentPass || 1} of ${analysis.multiPass.selectedSegmentIds.length}: reconstructing each angle before choosing a canonical view.` : 'Preparing sampled frames for the detector.') }}</p>
      <p v-else>{{ segmentLayout.layout.value ? 'Check which shots belong to the same event, correct replay roles, then confirm the proposed map.' : analysis.sceneVideo?.reconstruction?.error || 'Run automatic reconstruction to populate player and ball tracks from the extracted frames.' }}</p>
      <div v-if="analysis.sceneVideo" class="frame-summary"><strong>{{ analysis.analysisFrameCount }}</strong><small>{{ analysis.sceneVideo.selectedSegmentId ? 'analysis frames' : 'detector frames' }}</small><strong>{{ analysis.sceneVideo.fps.toFixed(2) }}</strong><small>source FPS</small></div>
      <div v-if="segmentLayout.layout.value && analysis.sceneVideo?.segments?.length" class="layout-editor-card">
        <div class="layout-editor-heading"><div><span>Suggested event map</span><strong>{{ segmentLayout.layout.value.groups.length }} events · {{ Math.round(segmentLayout.layout.value.confidence * 100) }}%</strong></div><i :class="segmentLayout.layout.value.status">{{ segmentLayout.layout.value.status }}</i></div>
        <p>{{ segmentLayout.layout.value.method === 'scoreboard-change+motion-dtw' ? `Score changes at ${segmentLayout.layout.value.scoreChangeTimes.map((time) => `${time.toFixed(0)}s`).join(', ')}; replay roles use motion alignment.` : 'No stable scoreboard found; review the order-based grouping.' }}</p>
        <div class="layout-editor-actions"><button :disabled="segmentLayout.rebuilding.value" @click="commands.rebuildLayout">{{ segmentLayout.rebuilding.value ? 'Analyzing…' : '↻ Rebuild' }}</button><button class="confirm" @click="commands.confirmLayout">✓ Confirm map</button><button class="split" :disabled="!segmentLayout.canSplitSelection.value" @click="commands.splitSelection">＋ Split {{ segmentLayout.selection.value.length || 'selected' }} into new event</button></div>
      </div>
      <button v-if="analysis.sceneVideo?.selectedSegmentId" class="wide-action" :disabled="reconstruction.reconstructing.value || reconstruction.status.value === 'processing' || reconstruction.status.value === 'queued'" @click="commands.reconstruct">{{ reconstruction.status.value === 'processing' || reconstruction.status.value === 'queued' ? 'Analyzing…' : '◎ Build automatic tracks' }}</button>
      <div v-if="analysis.sceneVideo?.segments?.length" class="shot-candidates">
        <div class="label-row"><label>Multi-angle passes</label><span>{{ segmentLayout.selection.value.length }}/6 selected</span></div>
        <p class="multi-pass-copy">Select a continuous tail such as 1-B + 1-C and split it into a new event, or select 2–6 variants for reconstruction.</p>
        <div v-for="segment in analysis.sceneVideo.segments" :key="segment.id" class="shot-candidate" :class="{ recommended: segment.recommended, selected: segmentLayout.selection.value.includes(segment.id) }">
          <div class="shot-candidate-main"><label class="shot-selector"><input v-model="segmentLayout.selection.value" type="checkbox" :value="segment.id" :disabled="!segmentLayout.selection.value.includes(segment.id) && segmentLayout.selection.value.length >= 6" /><b v-if="segment.layout" class="segment-layout-label" :style="{ borderColor: segmentLayout.segmentGroupColor(segment.layout.group), color: segmentLayout.segmentGroupColor(segment.layout.group) }">{{ segment.layout.label }}</b><span><strong>{{ segment.recommended ? '★ ' : '' }}{{ segment.label }}</strong><small>{{ segment.start.toFixed(2) }}–{{ segment.end.toFixed(2) }}s</small></span></label>
            <div v-if="segment.layout" class="shot-layout-controls"><select :value="segment.layout.group" :aria-label="`Event for ${segment.layout.label}`" @change="segmentLayout.assignGroup(segment, ($event.target as HTMLSelectElement).value)"><option v-for="group in segmentLayout.groupOptions.value" :key="group" :value="group">Event {{ group }}</option></select><select :value="segment.layout.role" :aria-label="`Role for ${segment.layout.label}`" @change="segmentLayout.assignRole(segment, ($event.target as HTMLSelectElement).value)"><option value="original">Original</option><option value="replay">Replay</option><option value="continuation">Continuation</option></select></div>
          </div>
          <button :disabled="composition.segmentCreating.value === segment.id" @click="composition.createSceneFromSegment(segment)">{{ composition.segmentCreating.value === segment.id ? '…' : segment.sceneId ? 'OPEN' : '→' }}</button>
        </div>
        <button class="wide-action multi-pass-action" :disabled="segmentLayout.selection.value.length < 2 || composition.multiPassStarting.value" @click="composition.startMultiPass">{{ composition.multiPassStarting.value ? 'Starting analysis…' : `◎ Analyze ${segmentLayout.selection.value.length || 'selected'} angles` }}</button>
      </div>
      <button class="wide-action" @click="commands.importClip">Import another clip</button>
    </div>
  </aside>
</template>
