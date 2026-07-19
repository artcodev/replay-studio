import { computed, onScopeDispose, watch } from 'vue'
import { isAnalysisJobTerminalTransition } from '../../../lib/analysisJobs'
import { useFrameAnalysis } from '../../../composables/useFrameAnalysis'
import { useFrameAnnotations } from '../../../composables/useFrameAnnotations'
import { useModelComparison } from '../../../composables/useModelComparison'
import { usePitchCalibrationEditor } from '../../../composables/usePitchCalibrationEditor'
import { useReconstructionController } from '../../../composables/useReconstructionController'
import { useVideoReviewViewport } from '../../../composables/useVideoReviewViewport'
import type { FrameAnalysis } from '../../../types/analysis'
import type { CalibrationFrameEvidence } from '../../../types/calibration'
import type { AnalysisJob } from '../../../types/project'
import type {
  BallDetectionBackend,
  ReconstructionModel,
} from '../../../types/reconstruction'
import type { EditorSessionContext } from '../session/useEditorSessionContext'
import type { EditorViewportContext } from '../viewport/useEditorViewportContext'

/** Analysis jobs, source-frame evidence and camera calibration for one scene. */
export function useEditorAnalysisContext(
  session: EditorSessionContext,
  viewport: EditorViewportContext,
) {
  let frameAnnotations!: ReturnType<typeof useFrameAnnotations>

  const frameAnalysis = useFrameAnalysis({
    scene: session.scene,
    currentTime: viewport.currentTime,
    selectedTrackId: viewport.selectedTrackId,
    selectedCanonicalPersonId: viewport.selectedCanonicalPersonId,
    selectedFramePersonId: viewport.selectedFramePersonId,
    activeTab: viewport.activeTab,
    playing: viewport.playing,
    sourceVideo: viewport.sourceVideo,
    saveState: session.saveState,
    error: session.error,
    projectId: session.editorProjectId,
    seekTo: viewport.seekTo,
    clearAnnotations: () => frameAnnotations.clear(),
  })

  const reconstruction = useReconstructionController({
    scene: session.scene,
    jobs: session.projectAnalysisJobs,
    cancelingJobIds: session.projectCancelingJobIds,
    refreshJobs: session.analysisJobs.refresh,
    cancelJob: session.analysisJobs.cancel,
    saveState: session.saveState,
    error: session.error,
    projectId: session.editorProjectId,
    clearFrameAnalysis: frameAnalysis.clear,
    seekTo: viewport.seekTo,
  })

  frameAnnotations = useFrameAnnotations({
    scene: session.scene,
    analysis: frameAnalysis.analysis,
    activeAnalysis: frameAnalysis.activeAnalysis,
    selectedTrackId: viewport.selectedTrackId,
    selectedCanonicalPersonId: viewport.selectedCanonicalPersonId,
    selectedFramePersonId: viewport.selectedFramePersonId,
    activeTab: viewport.activeTab,
    playing: viewport.playing,
    sourceVideo: viewport.sourceVideo,
    viewMode: viewport.viewMode,
    mutationLocked: reconstruction.mutationLocked,
    saveState: session.saveState,
    error: session.error,
    projectId: session.editorProjectId,
    analyzeFrame: frameAnalysis.analyze,
    clearFrameAnalysis: frameAnalysis.clear,
    seekTo: viewport.seekTo,
    queueRebuild: reconstruction.queueIdentityCorrectionRebuild,
    canonicalPersonById: frameAnalysis.canonicalPersonById,
    framePersonCanonicalId: frameAnalysis.framePersonCanonicalId,
    framePersonLabel: frameAnalysis.framePersonLabel,
  })

  const modelComparison = useModelComparison({
    projectId: session.editorProjectId,
    scene: session.scene,
    sceneVideo: viewport.sceneVideo,
    jobs: session.projectAnalysisJobs,
    selectedTrackId: viewport.selectedTrackId,
    selectedCanonicalPersonId: viewport.selectedCanonicalPersonId,
    playing: viewport.playing,
    sourceVideo: viewport.sourceVideo,
    saveState: session.saveState,
    error: session.error,
    refreshJobs: session.analysisJobs.refresh,
  })

  const calibrationEvidence = computed(() => (
    viewport.sceneVideo.value?.reconstruction?.calibration ?? null
  ))
  const calibrationFrames = computed<CalibrationFrameEvidence[]>(() => (
    calibrationEvidence.value?.frameEvidence ?? []
  ))
  const pitchCalibration = usePitchCalibrationEditor({
    scene: session.scene,
    currentTime: viewport.currentTime,
    activeTab: viewport.activeTab,
    playing: viewport.playing,
    sourceVideo: viewport.sourceVideo,
    viewMode: viewport.viewMode,
    reconstructing: reconstruction.reconstructing,
    selectedTrackId: viewport.selectedTrackId,
    selectedCanonicalPersonId: viewport.selectedCanonicalPersonId,
    calibrationFrames,
    saveState: session.saveState,
    error: session.error,
    projectId: session.editorProjectId,
    seekTo: viewport.seekTo,
    clearFrameAnalysis: frameAnalysis.clear,
    startReconstructionPolling: reconstruction.startPolling,
  })
  const videoReview = useVideoReviewViewport({
    isZoomBlocked: () => Boolean(
      pitchCalibration.draggedAnchor.value || frameAnnotations.drag.value,
    ),
    isPanBlocked: () => Boolean(
      frameAnnotations.mode.value || pitchCalibration.draggedAnchor.value,
    ),
  })

  const reconstructionModels: Array<{ value: ReconstructionModel; label: string }> = [
    { value: 'yolo26n.pt', label: '26n · fast' },
    { value: 'yolo26s.pt', label: '26s' },
    { value: 'yolo26m.pt', label: '26m · balanced' },
    { value: 'yolo26l.pt', label: '26l' },
    { value: 'yolo26x.pt', label: '26x · max' },
  ]
  const ballDetectionBackends: Array<{ value: BallDetectionBackend; label: string }> = [
    { value: 'dedicated-ultralytics', label: 'Roboflow · tiled' },
    { value: 'wasb-service', label: 'WASB · temporal' },
    { value: 'generic-ultralytics', label: 'COCO · fallback' },
  ]
  const calibrationLabel = computed(() => {
    const calibration = pitchCalibration.pitchCalibration.value
    if (reconstruction.qualityVerdict.value === 'pending') return 'QUALITY PENDING'
    if (reconstruction.qualityVerdict.value === 'reject') return 'QUALITY REJECTED'
    if (reconstruction.qualityVerdict.value === 'review') return 'METRIC · REVIEW'
    if (reconstruction.qualityVerdict.value === 'pass') return 'METRIC · PASS'
    return calibration?.status === 'ready'
      ? `CALIBRATED ${Math.round((calibration.confidence ?? 0) * 100)}%`
      : calibration?.status === 'approximate'
        ? `HALF-PITCH ${Math.round((calibration.confidence ?? 0) * 100)}%`
        : '2.5D FALLBACK'
  })
  const busy = computed(() => Boolean(
    reconstruction.reconstructing.value
    || reconstruction.running.value
    || frameAnnotations.saving.value,
  ))

  function framePersonSelectionDescription(person: FrameAnalysis['people'][number]) {
    const canonicalPersonId = frameAnalysis.framePersonCanonicalId(person)
    if (canonicalPersonId && frameAnalysis.renderTrackForCanonicalPerson(canonicalPersonId)) {
      return 'select linked video and 3D player'
    }
    if (canonicalPersonId) return 'select canonical person; not projected in 3D'
    return 'select unresolved video detection'
  }

  watch(() => ({
    sceneId: session.scene.value?.id ?? null,
    job: reconstruction.currentJob.value,
  }), (current, previous) => {
    const currentJob = current.job
    const previousJob = previous?.job
    if (
      !current.sceneId
      || current.sceneId !== previous?.sceneId
      || !currentJob
      || !isAnalysisJobTerminalTransition(previousJob, currentJob)
    ) return
    void reconstruction.syncAfterTerminal(currentJob)
  })

  watch(modelComparison.job, (currentJob, previousJob) => {
    if (!isAnalysisJobTerminalTransition(previousJob, currentJob)) return
    void modelComparison.syncAfterTerminal(currentJob as AnalysisJob)
  })

  watch(() => session.scene.value?.id, (sceneId) => {
    frameAnalysis.clear()
    modelComparison.reset()
    pitchCalibration.reset()
    videoReview.reset()
    reconstruction.reset()
    reconstruction.resetModelSelection()
    if (sceneId) reconstruction.resumePolling()
  })
  watch(viewport.activePassSceneId, videoReview.reset)

  onScopeDispose(() => {
    frameAnalysis.clear()
    reconstruction.stop()
  })

  return {
    frameAnalysis,
    frameAnnotations,
    reconstruction,
    modelComparison,
    pitchCalibration,
    videoReview,
    calibrationEvidence,
    calibrationFrames,
    calibrationLabel,
    reconstructionModels,
    ballDetectionBackends,
    busy,
    framePersonSelectionDescription,
  }
}

export type EditorAnalysisContext = ReturnType<typeof useEditorAnalysisContext>
