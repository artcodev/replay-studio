import { computed, watch } from 'vue'
import { hasActiveDedicatedUnbindForOwner } from '../../../lib/identityCorrections'
import { interpolateKeyframes, upsertKeyframe } from '../../../lib/interpolate'
import {
  pathHasVisibleProjection,
  resolvePathProjectionContext,
} from '../../../lib/pathProjection'
import { useCompositionEditor } from '../../../composables/useCompositionEditor'
import { useEditorObjectSelection } from '../../../composables/useEditorObjectSelection'
import { useManualBallEditor } from '../../../composables/useManualBallEditor'
import { usePlayerActionEditor } from '../../../composables/usePlayerActionEditor'
import { useSegmentLayoutEditor } from '../../../composables/useSegmentLayoutEditor'
import type { Track } from '../../../types/tracking'
import type { EditorAnalysisContext } from '../analysis/useEditorAnalysisContext'
import type { EditorSessionContext } from '../session/useEditorSessionContext'
import type { EditorViewportContext } from '../viewport/useEditorViewportContext'

/** Editable trajectories, actions, object selection and segment composition. */
export function useEditorCompositionContext(
  session: EditorSessionContext,
  viewport: EditorViewportContext,
  analysis: EditorAnalysisContext,
) {
  const manualBall = useManualBallEditor({
    scene: session.scene,
    currentTime: viewport.currentTime,
    playing: viewport.playing,
    sourceVideo: viewport.sourceVideo,
    selectedTrackId: viewport.selectedTrackId,
    selectedCanonicalPersonId: viewport.selectedCanonicalPersonId,
    selectedFramePersonId: viewport.selectedFramePersonId,
    editMode: viewport.editMode,
    activeTab: viewport.activeTab,
    viewMode: viewport.viewMode,
    viewOptions: viewport.viewOptions,
    saveState: session.saveState,
    error: session.error,
    projectId: session.editorProjectId,
    seekTo: viewport.seekTo,
    notifySceneMutation: session.notifySceneMutation,
  })
  const selection = useEditorObjectSelection({
    scene: session.scene,
    selectedTrackId: viewport.selectedTrackId,
    selectedCanonicalPersonId: viewport.selectedCanonicalPersonId,
    trackQuery: viewport.trackQuery,
    ballSelected: manualBall.selected,
  })
  const playerActions = usePlayerActionEditor({
    scene: session.scene,
    currentTime: viewport.currentTime,
    selectedActorId: selection.selectedActionActorId,
    mutationLocked: analysis.reconstruction.mutationLocked,
    playing: viewport.playing,
    sourceVideo: viewport.sourceVideo,
    saveState: session.saveState,
    error: session.error,
    projectId: session.editorProjectId,
    seekTo: viewport.seekTo,
    notifySceneMutation: session.notifySceneMutation,
  })
  const segmentLayout = useSegmentLayoutEditor({
    scene: session.scene,
    sceneVideo: viewport.sceneVideo,
    selectedTrackId: viewport.selectedTrackId,
    selectedCanonicalPersonId: viewport.selectedCanonicalPersonId,
    currentTime: viewport.currentTime,
    saveState: session.saveState,
    error: session.error,
    projectId: session.editorProjectId,
    saveScene: session.saveScene,
    seekTo: viewport.seekTo,
    writeRouteTime: (time) => session.replaceEditorRouteView({ time }),
    notifySceneMutation: session.notifySceneMutation,
  })
  const composition = useCompositionEditor({
    projectId: session.editorProjectId,
    scene: session.scene,
    sceneVideo: viewport.sceneVideo,
    projectSegments: session.workspaceSegments,
    multiPassSelection: segmentLayout.selection,
    selectedTrackId: viewport.selectedTrackId,
    selectedCanonicalPersonId: viewport.selectedCanonicalPersonId,
    currentTime: viewport.currentTime,
    reconstructing: analysis.reconstruction.reconstructing,
    saveState: session.saveState,
    error: session.error,
    navigateToScene: session.navigateEditorScene,
    startReconstructionPolling: analysis.reconstruction.startPolling,
    seekTo: viewport.seekTo,
  })

  const selectedTeam = computed(() => {
    if (!session.scene.value) return null
    const teamId = selection.selectedTrack.value?.teamId
      ?? selection.selectedCanonicalPerson.value?.teamId
    return session.scene.value.payload.teams.find((team) => team.id === teamId) ?? null
  })
  const showPlayerActionTimeline = computed(() => (
    playerActions.visible.value && !manualBall.selected.value
  ))
  const busy = computed(() => manualBall.saving.value || playerActions.saving.value)

  const videoPathUsesReferenceCamera = computed(() => (
    !viewport.multiPassPlayback.analysis.value
    || viewport.multiPassPlayback.activePass.value?.sceneId
      === viewport.multiPassPlayback.analysis.value.referenceSceneId
  ))
  const videoPathProjectionContext = computed(() => (
    videoPathUsesReferenceCamera.value
      ? resolvePathProjectionContext(
        analysis.calibrationFrames.value,
        viewport.currentTime.value,
      )
      : null
  ))
  const videoPathUnavailableReason = computed<string | null>(() => {
    const subject = selection.selectedPathSubject.value
    if (!subject || !selection.selectedPathSegments.value.length) return null
    if (viewport.activeTab.value === 'qa') return 'Hidden while calibration QA is open'
    if (!videoPathUsesReferenceCamera.value) {
      return 'Unavailable for this replay angle · switch to the reference camera'
    }
    if (!videoPathProjectionContext.value) {
      return 'No trusted calibration for this frame · calibrate or move the playhead'
    }
    if (!pathHasVisibleProjection(
      videoPathProjectionContext.value,
      selection.selectedPathSegments.value,
    )) return 'Path is outside the current camera view'
    return null
  })
  const videoPathSurfaceNote = computed<string | null>(() => {
    const context = videoPathProjectionContext.value
    if (!context || videoPathUnavailableReason.value) return null
    const notes: string[] = []
    if (context.mode === 'interpolated') {
      notes.push(`Bounded camera interpolation · ${(context.interpolationIntervalSeconds * 1000).toFixed(0)} ms`)
    } else if (context.mode === 'nearest') {
      notes.push(`Nearest camera sample · Δ ${(context.timeOffsetSeconds * 1000).toFixed(0)} ms`)
    }
    if (context.uncertaintyMetres !== null) {
      notes.push(`camera uncertainty ±${context.uncertaintyMetres.toFixed(1)} m`)
    }
    if (selection.selectedPathSubject.value?.kind === 'ball') {
      notes.push('ground projection on video · height remains in 3D')
    }
    return notes.length ? notes.join(' · ') : null
  })

  function moveSelected(position: { x: number; z: number }) {
    const trackId = selection.selectedTrack.value?.id
    if (!trackId) return
    session.mutateScene((document) => {
      const track = document.payload.tracks.find((item) => item.id === trackId)
      if (!track) return
      track.keyframes = upsertKeyframe(track.keyframes, {
        t: Number(viewport.currentTime.value.toFixed(2)),
        x: Number(position.x.toFixed(2)),
        z: Number(position.z.toFixed(2)),
        confidence: 1,
      })
    })
    session.saveState.value = 'Unsaved changes'
  }

  function updateSelectedTrackMetadata(field: 'label' | 'number', value: string | number) {
    const trackId = selection.selectedTrack.value?.id
    if (!trackId) return
    session.mutateScene((document) => {
      const track = document.payload.tracks.find((item) => item.id === trackId)
      if (!track) return
      if (field === 'label') track.label = String(value)
      else track.number = Number(value)
    })
    session.saveState.value = 'Unsaved changes'
  }

  function updateTrackPosition(axis: 'x' | 'z', value: string) {
    if (!selection.selectedTrack.value) return
    const position = interpolateKeyframes(
      selection.selectedTrack.value.keyframes,
      viewport.currentTime.value,
    )
    moveSelected({
      x: axis === 'x' ? Number(value) : position.x,
      z: axis === 'z' ? Number(value) : position.z,
    })
  }

  function trackQualityFor(track: Track) {
    const observed = track.keyframes.filter(
      (keyframe) => keyframe.observed !== false && keyframe.confidence > 0.12,
    )
    if (!observed.length) return 0
    return Math.round(
      observed.reduce((total, keyframe) => total + keyframe.confidence, 0)
      / observed.length * 100,
    )
  }

  function canonicalHasActiveDedicatedUnbind(canonicalPersonId: string) {
    const currentScene = session.scene.value
    if (!currentScene) return false
    const tracks = currentScene.payload.tracks.filter(
      (track) => track.canonicalPersonId === canonicalPersonId,
    )
    return hasActiveDedicatedUnbindForOwner(
      currentScene.payload.videoAsset?.reconstruction?.frameAnnotations ?? [],
      [canonicalPersonId, ...tracks.map((track) => track.id)],
      {
        canonicalPeople: currentScene.payload.canonicalPeople,
        tracks: currentScene.payload.tracks,
      },
    )
  }

  watch(viewport.currentTime, manualBall.syncPlayhead)
  watch(
    [viewport.selectedTrackId, viewport.selectedCanonicalPersonId],
    ([trackId, canonicalPersonId]) => {
      if (!trackId && !canonicalPersonId) return
      manualBall.clearSelection()
      if (trackId) {
        const track = session.scene.value?.payload.tracks.find((item) => item.id === trackId)
        viewport.selectedCanonicalPersonId.value = track?.canonicalPersonId ?? canonicalPersonId
      }
    },
  )
  watch(() => session.scene.value?.id, () => {
    manualBall.reset()
    playerActions.reset()
    segmentLayout.reset()
  })

  return {
    manualBall,
    selection,
    playerActions,
    segmentLayout,
    composition,
    selectedTeam,
    showPlayerActionTimeline,
    busy,
    videoPathUsesReferenceCamera,
    videoPathProjectionContext,
    videoPathUnavailableReason,
    videoPathSurfaceNote,
    moveSelected,
    updateSelectedTrackMetadata,
    updateTrackPosition,
    trackQualityFor,
    canonicalHasActiveDedicatedUnbind,
  }
}

export type EditorCompositionContext = ReturnType<typeof useEditorCompositionContext>
