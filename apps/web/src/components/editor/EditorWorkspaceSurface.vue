<script setup lang="ts">
import { computed } from 'vue'
import StudioTopbar from '../shell/StudioTopbar.vue'
import VideoIngestDrawer from '../VideoIngestDrawer.vue'
import EditorInspectorSurface from './EditorInspectorSurface.vue'
import EditorSidebar from './EditorSidebar.vue'
import EditorStageToolbar from './EditorStageToolbar.vue'
import EditorTimelineSurface from './EditorTimelineSurface.vue'
import EditorViewportSurface from './EditorViewportSurface.vue'
import { injectEditorContexts } from '../../features/editor/editorContexts'
import type { VideoSegment } from '../../types/media'

const { session, viewport, analysis, composition, identity } = injectEditorContexts()
const {
  scene,
  timelineScene,
  selectedProject,
  workspaceProjects,
  workspaceAssets,
  workspaceSegments,
  workspaceMatch,
  projectLoading,
  saveState,
  error,
  saving,
  videoIngestOpen,
  projects,
  activeProjectId,
  internalSceneLabel,
  navigateEditorScene,
  updateSceneTitle,
  openTimelineSegment,
  openProcessedVideo,
  returnToProjects,
} = session
const {
  sceneVideo,
  selectedTrackId,
  selectedCanonicalPersonId,
  selectedFramePersonId,
  trackQuery,
  editMode,
  viewOptions,
  videoOverlayOptions,
  renderQuality,
  viewMode,
  activeCamera,
  activeTab,
  currentTime,
  playbackRate,
  playing,
  sourceVideo,
  viewport: viewportApi,
  seekTo,
  onTimelineInput,
  togglePlay,
  timeLabel,
  chooseSourcePass,
  onCameraPresetChange,
  multiPassPlayback,
} = viewport
const {
  frameAnalysis: frameAnalysisController,
  frameAnnotations,
  reconstruction: reconstructionController,
  modelComparison: modelComparisonController,
  pitchCalibration: pitchCalibrationEditor,
  videoReview,
  calibrationEvidence,
  calibrationFrames,
  calibrationLabel,
  reconstructionModels,
  ballDetectionBackends,
  framePersonSelectionDescription,
} = analysis
const {
  manualBall: manualBallEditor,
  selection,
  playerActions: playerActionEditor,
  segmentLayout: segmentLayoutEditor,
  composition: compositionEditor,
  selectedTeam,
  showPlayerActionTimeline,
  videoPathUsesReferenceCamera,
  videoPathProjectionContext,
  videoPathUnavailableReason,
  videoPathSurfaceNote,
  moveSelected,
  updateSelectedTrackMetadata,
  updateTrackPosition,
  trackQualityFor,
} = composition
const {
  rosterPlayers,
  matchSnapshotRefreshAvailable,
  projectMatchContext,
  projectMatchTeams,
  identityReview: identityReviewEditor,
  identityPresentation,
  projectMatch: projectMatchEditor,
  openProjectMatchWorkspace,
  inspectIdentityFrame,
  addEventBinding,
  removeEventBinding,
} = identity

const {
  selectedTrack,
  selectedCanonicalPerson,
  selectedActionActorId,
  selectedActionActorLabel,
  selectedPathSubject,
  selectedPathKeyframes,
  selectedPathSegments,
  unavailablePathSubjectLabel,
  filteredTracks,
  canonicalPeopleWithoutRender,
  filteredCanonicalPeopleWithoutRender,
  ballMatchesTrackQuery,
  reconstructionPreviewScene,
} = selection
const {
  selected: ballSelected,
  mode: ballTrajectoryMode,
  selectBall: selectBallObject,
} = manualBallEditor
const {
  analysis: multiPassAnalysis,
  activePass,
  relationLabel: passRelationLabel,
} = multiPassPlayback
const {
  running: reconstructionRunning,
  mutationLocked: reconstructionMutationLocked,
  reconstruct: reconstructCurrentScene,
} = reconstructionController
const {
  selectTrack,
  selectCanonicalPerson,
  framePersonLabel,
  framePersonCanonicalId,
} = frameAnalysisController
const { selectDetectedPerson } = frameAnnotations
const { report: modelComparison, frameCount: analysisFrameCount } = modelComparisonController
const {
  splitSelection: splitSelectedIntoNewEvent,
  confirm: confirmSegmentLayout,
  rebuild: rebuildSegmentLayout,
} = segmentLayoutEditor
const {
  person: selectedIdentityReviewPerson,
  observations: selectedIdentityReviewObservations,
  workers: identityReviewWorkers,
  dedicatedUnbindActive: selectedCanonicalDedicatedUnbindActive,
} = identityPresentation
const {
  load: loadIdentityReview,
  confirm: confirmCanonicalRoster,
  reject: rejectIdentityCandidate,
  unbind: unbindCanonicalRoster,
  clearBinding: clearCanonicalRosterBinding,
} = identityReviewEditor
const {
  refreshing: matchSnapshotRefreshing,
  importing: manualRosterImporting,
  importError: manualRosterImportError,
  refresh: refreshMatchSnapshot,
  importFile: importManualRosterFile,
} = projectMatchEditor
const timelineSceneVideo = computed(() => timelineScene.value?.payload.videoAsset ?? null)
const hasMasterTimeline = computed(() => Boolean(
  timelineSceneVideo.value?.segmentLayout
  && timelineSceneVideo.value.segments?.length,
))

function handleMasterTimelineSegment(segment: VideoSegment) {
  if (timelineScene.value?.id === scene.value?.id) {
    segmentLayoutEditor.handleSegment(segment)
    return
  }
  void openTimelineSegment(segment)
}

const viewportPlaybackContext = { scene, sceneVideo, currentTime, sourceVideo, seekTo }
const viewportViewContext = {
  mode: viewMode,
  options: viewOptions,
  videoOverlayOptions,
  renderQuality,
  activeTab,
  viewport: viewportApi,
}
const viewportSelectionContext = {
  selectedTrack,
  selectedCanonicalPerson,
  selectedTrackId,
  selectedFramePersonId,
  editMode,
  pathSubject: selectedPathSubject,
  pathKeyframes: selectedPathKeyframes,
  pathSegments: selectedPathSegments,
  unavailablePathSubjectLabel,
  reconstructionPreviewScene,
}
const viewportAnalysisContext = {
  frameCount: analysisFrameCount,
  calibrationFrames,
  videoPathUsesReferenceCamera,
  videoPathProjectionContext,
  videoPathUnavailableReason,
  videoPathSurfaceNote,
  calibrationLabel,
  framePersonSelectionDescription,
}
const stageToolbarView = {
  scene,
  sceneVideo,
  activeCamera,
  viewMode,
  viewOptions,
  renderQuality,
  activeTab,
  multiPass: multiPassAnalysis,
  activePass,
}
const stageToolbarControllers = {
  reconstruction: reconstructionController,
  frameAnalysis: frameAnalysisController,
  manualBall: manualBallEditor,
  modelComparison: modelComparisonController,
  calibration: pitchCalibrationEditor,
}
const stageToolbarModels = {
  reconstruction: reconstructionModels,
  ballDetection: ballDetectionBackends,
}
const stageToolbarCommands = {
  cameraPresetChange: onCameraPresetChange,
  chooseSourcePass,
  passRelationLabel,
}
</script>

<template>
  <div class="editor-workspace-surface">
    <StudioTopbar
      surface="editor"
      :scene-title="scene?.title ?? null"
      :project="selectedProject"
      :project-count="workspaceProjects.length"
      :asset-count="workspaceAssets.length"
      :segment-count="workspaceSegments.length"
      :save-state="saveState"
      :project-loading="projectLoading"
      @update:scene-title="updateSceneTitle"
      @open-import="videoIngestOpen = true"
      @return-projects="returnToProjects"
    />

    <section v-if="error && !scene" class="fatal-state">
      <div class="fatal-card">
        <span class="fatal-code">API OFFLINE</span>
        <h2>The studio could not reach its workspace.</h2>
        <p>{{ error }}</p>
        <code>uvicorn app.main:app --app-dir apps/api</code>
        <button class="button primary" @click="returnToProjects">Back to project</button>
      </div>
    </section>

    <section v-else-if="scene" class="studio-grid">
      <EditorSidebar
        :projects="projects"
        :active-project-id="activeProjectId"
        :internal-scene-label="internalSceneLabel"
        :scene-title="scene.title"
        :home-team="projectMatchTeams.home"
        :away-team="projectMatchTeams.away"
        :match-context-label="projectMatchContext.label"
        :match="workspaceMatch"
        :match-refresh-available="matchSnapshotRefreshAvailable"
        :match-refreshing="matchSnapshotRefreshing"
        :roster-importing="manualRosterImporting"
        :mutation-locked="reconstructionMutationLocked"
        :roster-import-error="manualRosterImportError"
        :tracked-object-count="scene.payload.tracks.length + canonicalPeopleWithoutRender.length + 1"
        :track-query="trackQuery"
        :tracks="filteredTracks"
        :identities="filteredCanonicalPeopleWithoutRender"
        :selected-track-id="selectedTrackId"
        :selected-canonical-person-id="selectedCanonicalPersonId"
        :ball-matches-query="ballMatchesTrackQuery"
        :ball-selected="ballSelected"
        :ball-trajectory-mode="ballTrajectoryMode"
        :ball-keyframe-count="scene.payload.ball.keyframes.length"
        :track-quality="trackQualityFor"
        @navigate-scene="navigateEditorScene"
        @refresh-roster="refreshMatchSnapshot"
        @import-roster="importManualRosterFile"
        @update:track-query="trackQuery = $event"
        @select-track="selectTrack"
        @select-identity="selectCanonicalPerson"
        @select-ball="selectBallObject"
      />

      <section
        class="stage-column"
        :class="{
          'has-segment-map': hasMasterTimeline,
          'has-ball-timeline': ballTrajectoryMode === 'manual',
          'has-player-action-timeline': showPlayerActionTimeline,
        }"
      >
        <EditorStageToolbar
          :view="stageToolbarView"
          :controllers="stageToolbarControllers"
          :models="stageToolbarModels"
          :commands="stageToolbarCommands"
        />
        <EditorViewportSurface
          :playback="viewportPlaybackContext"
          :view="viewportViewContext"
          :selection="viewportSelectionContext"
          :analysis="viewportAnalysisContext"
          :reconstruction="reconstructionController"
          :video-review="videoReview"
          :calibration="pitchCalibrationEditor"
          :frame-analysis="frameAnalysisController"
          :frame-annotations="frameAnnotations"
          :manual-ball="manualBallEditor"
          :player-actions="playerActionEditor"
          :move-selected="moveSelected"
        />

        <EditorTimelineSurface
          v-model:current-time="currentTime"
          v-model:playback-rate="playbackRate"
          :scene="scene"
          :timeline-scene="timelineScene"
          :scene-video="sceneVideo"
          :playing="playing"
          :time-label="timeLabel"
          :reconstruction-running="reconstructionRunning"
          :reconstruction-mutation-locked="reconstructionMutationLocked"
          :selected-action-actor-id="selectedActionActorId"
          :selected-action-actor-label="selectedActionActorLabel"
          :show-player-action-timeline="showPlayerActionTimeline"
          :segment-layout="segmentLayoutEditor"
          :manual-ball="manualBallEditor"
          :player-actions="playerActionEditor"
          @seek="seekTo"
          @master-segment="handleMasterTimelineSegment"
          @timeline-input="onTimelineInput"
          @toggle-play="togglePlay"
        />
      </section>

      <EditorInspectorSurface
        v-model:active-tab="activeTab"
        :scene="scene"
        :current-time="currentTime"
        :saving="saving"
        :edit-mode="editMode"
        :selection="{
          track: selectedTrack,
          canonicalPerson: selectedCanonicalPerson,
          identityPerson: selectedIdentityReviewPerson,
          identityObservations: selectedIdentityReviewObservations,
          identityWorkers: identityReviewWorkers,
          dedicatedUnbindActive: selectedCanonicalDedicatedUnbindActive,
          framePersonId: selectedFramePersonId,
          team: selectedTeam,
        }"
        :analysis="{
          sceneVideo,
          calibrationEvidence,
          calibrationFrames,
          calibrationLabel,
          analysisFrameCount,
          modelComparison,
          multiPass: multiPassAnalysis,
        }"
        :match-view="{
          match: workspaceMatch,
          rosterPlayers,
          refreshAvailable: matchSnapshotRefreshAvailable,
          refreshing: matchSnapshotRefreshing,
        }"
        :controllers="{
          manualBall: manualBallEditor,
          identityReview: identityReviewEditor,
          frameAnalysis: frameAnalysisController,
          frameAnnotations,
          reconstruction: reconstructionController,
          pitchCalibration: pitchCalibrationEditor,
          segmentLayout: segmentLayoutEditor,
          composition: compositionEditor,
        }"
        :commands="{
          loadIdentityReview,
          confirmRoster: confirmCanonicalRoster,
          rejectCandidate: rejectIdentityCandidate,
          inspectIdentityFrame,
          unbindRoster: unbindCanonicalRoster,
          clearRosterBinding: clearCanonicalRosterBinding,
          updateTrackLabel: (value: string) => updateSelectedTrackMetadata('label', value),
          updateTrackNumber: (value: number) => updateSelectedTrackMetadata('number', value),
          updateTrackPosition,
          framePersonLabel,
          framePersonCanonicalId,
          selectDetectedPerson,
          seek: seekTo,
          refreshMatch: refreshMatchSnapshot,
          openMatchWorkspace: openProjectMatchWorkspace,
          addEvent: addEventBinding,
          removeEvent: removeEventBinding,
          reconstruct: reconstructCurrentScene,
          rebuildLayout: rebuildSegmentLayout,
          confirmLayout: confirmSegmentLayout,
          splitSelection: splitSelectedIntoNewEvent,
          importClip: () => { videoIngestOpen = true },
          passRelationLabel,
        }"
        @update:current-time="currentTime = $event"
        @update:edit-mode="editMode = $event"
      />
    </section>

    <VideoIngestDrawer
      :open="videoIngestOpen"
      :project-id="selectedProject?.id"
      :project-title="selectedProject?.title"
      @close="videoIngestOpen = false"
      @ready="openProcessedVideo"
    />

    <div v-if="error && scene" class="toast" @click="error = null"><strong>Something went wrong</strong><span>{{ error }}</span><button>×</button></div>
  </div>
</template>
