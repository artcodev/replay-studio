import { ref, type Ref } from 'vue'
import { calibrationClient } from '../lib/api/calibration'
import { clientPointToContainedMedia } from '../lib/videoReviewTransform'
import { usePitchCalibrationPresentation } from '../features/calibration/usePitchCalibrationPresentation'
import { preserveSceneActorSelection } from '../features/editor/selection/preserveSceneActorSelection'
import {
  PITCH_CALIBRATION_PRESETS,
  projectPitchCalibrationPresetAnchors,
} from '../lib/pitchCalibrationPresets'
import type { CalibrationFrameEvidence, PitchCalibrationPreset } from '../types/calibration'
import type { SceneDocument } from '../types/scene'

type PitchCalibrationEditorOptions = {
  scene: Ref<SceneDocument | null>
  currentTime: Ref<number>
  activeTab: Ref<'binding' | 'qa' | 'events'>
  playing: Ref<boolean>
  sourceVideo: Ref<HTMLVideoElement | null>
  viewMode: Ref<'video' | 'split' | '3d'>
  reconstructing: Ref<boolean>
  selectedTrackId: Ref<string | null>
  selectedCanonicalPersonId: Ref<string | null>
  calibrationFrames: Readonly<Ref<CalibrationFrameEvidence[]>>
  saveState: Ref<string>
  error: Ref<string | null>
  projectId: () => string
  seekTo: (time: number) => void
  clearFrameAnalysis: () => void
}

/** Owns current-frame pitch calibration and attack-direction edits. */
export function usePitchCalibrationEditor(options: PitchCalibrationEditorOptions) {
  const draft = ref<Awaited<ReturnType<typeof calibrationClient.auto>> | null>(null)
  const preset = ref<PitchCalibrationPreset>('penalty-area-right')
  const loading = ref(false)
  const applying = ref(false)
  const pitchSideSaving = ref(false)
  const overlay = ref<SVGSVGElement | null>(null)
  const draggedAnchor = ref<string | null>(null)

  const frames = options.calibrationFrames
  const presentation = usePitchCalibrationPresentation({
    scene: options.scene,
    currentTime: options.currentTime,
    activeTab: options.activeTab,
    sourceVideo: options.sourceVideo,
    frames,
    draft,
  })

  async function open(nextPreset?: PitchCalibrationPreset) {
    const scene = options.scene.value
    if (!scene?.payload.videoAsset?.selectedSegmentId || loading.value) return
    options.playing.value = false
    options.sourceVideo.value?.pause()
    options.viewMode.value = 'split'
    options.clearFrameAnalysis()
    loading.value = true
    if (nextPreset) preset.value = nextPreset
    options.saveState.value = `Preparing pitch overlay at ${options.currentTime.value.toFixed(2)}s…`
    try {
      const prepared = await calibrationClient.auto(
        options.projectId(),
        scene.id,
        options.currentTime.value,
        nextPreset,
      )
      draft.value = prepared
      preset.value = prepared.preset
      options.seekTo(prepared.sceneTime)
      options.saveState.value = prepared.alignmentError === null
        ? 'Pitch anchors ready for review'
        : `Pitch overlay · ${prepared.alignmentError.toFixed(1)} px error`
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not prepare pitch calibration'
    } finally {
      loading.value = false
    }
  }

  async function changePreset(event: Event) {
    const value = draft.value
    if (!value || loading.value) return
    preset.value = (event.target as HTMLSelectElement).value as PitchCalibrationPreset
    value.preset = preset.value
    value.anchors = projectPitchCalibrationPresetAnchors(
      value.imageToPitch,
      preset.value,
      value.frameWidth,
      value.frameHeight,
    )
    await refresh()
  }

  function imagePoint(event: PointerEvent) {
    if (!overlay.value || !draft.value) return null
    return clientPointToContainedMedia(
      event.clientX,
      event.clientY,
      overlay.value.getBoundingClientRect(),
      draft.value.frameWidth,
      draft.value.frameHeight,
    )
  }

  function updateDraggedAnchor(event: PointerEvent) {
    const point = imagePoint(event)
    const anchor = draft.value?.anchors.find((item) => item.id === draggedAnchor.value)
    if (!anchor || !point) return
    anchor.image.x = point.x
    anchor.image.y = point.y
  }

  function startAnchorDrag(event: PointerEvent, anchorId: string) {
    draggedAnchor.value = anchorId
    try {
      overlay.value?.setPointerCapture(event.pointerId)
    } catch {
      // Synthetic accessibility input may not own a native pointer capture.
    }
    updateDraggedAnchor(event)
  }

  async function refresh() {
    const scene = options.scene.value
    if (!scene || !draft.value || loading.value) return
    loading.value = true
    try {
      draft.value = await calibrationClient.preview(
        options.projectId(),
        scene.id,
        draft.value.sceneTime,
        preset.value,
        draft.value.anchors,
      )
      options.saveState.value = draft.value.alignmentError === null
        ? 'Pitch overlay updated'
        : `Pitch overlay · ${draft.value.alignmentError.toFixed(1)} px error`
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not update pitch overlay'
    } finally {
      loading.value = false
    }
  }

  function finishAnchorDrag(event: PointerEvent) {
    if (!draggedAnchor.value) return
    updateDraggedAnchor(event)
    draggedAnchor.value = null
    try {
      if (overlay.value?.hasPointerCapture(event.pointerId)) overlay.value.releasePointerCapture(event.pointerId)
    } catch {
      // The drag is already complete when the browser releases capture first.
    }
    void refresh()
  }

  function nudgeAnchor(event: KeyboardEvent, anchorId: string) {
    const value = draft.value
    const anchor = value?.anchors.find((item) => item.id === anchorId)
    if (!value || !anchor) return
    const amount = event.shiftKey ? 16 : 4
    const movement = {
      ArrowLeft: [-amount, 0],
      ArrowRight: [amount, 0],
      ArrowUp: [0, -amount],
      ArrowDown: [0, amount],
    }[event.key]
    if (!movement) return
    anchor.image.x = Math.max(0, Math.min(value.frameWidth, anchor.image.x + movement[0]))
    anchor.image.y = Math.max(0, Math.min(value.frameHeight, anchor.image.y + movement[1]))
    void refresh()
  }

  function close() {
    draft.value = null
    draggedAnchor.value = null
    options.saveState.value = 'Pitch calibration cancelled'
  }

  async function apply(acceptQualityWarning = false) {
    const scene = options.scene.value
    const value = draft.value
    if (!scene || !value || applying.value) return
    applying.value = true
    options.saveState.value = 'Saving frame correction as a draft…'
    try {
      const updated = await calibrationClient.saveDraft(
        options.projectId(),
        scene.id,
        value.sceneTime,
        preset.value,
        value.anchors,
        value.source,
        acceptQualityWarning,
      )
      if (options.scene.value?.id !== scene.id) return
      const selection = preserveSceneActorSelection(updated, {
        trackId: options.selectedTrackId.value,
        canonicalPersonId: options.selectedCanonicalPersonId.value,
      })
      options.scene.value = updated
      draft.value = null
      options.selectedTrackId.value = selection.trackId
      options.selectedCanonicalPersonId.value = selection.canonicalPersonId
      options.saveState.value = 'Frame correction staged · finalize it in the calibration workspace'
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not apply pitch calibration'
    } finally {
      applying.value = false
    }
  }

  async function calibrateQaFrame(sceneTime: number) {
    options.seekTo(sceneTime)
    await open()
  }

  async function changeAttackingGoal(event: Event | 'left' | 'right') {
    const side = typeof event === 'string'
      ? event
      : (event.target as HTMLSelectElement).value as 'left' | 'right'
    const scene = options.scene.value
    if (!scene || !['left', 'right'].includes(side) || pitchSideSaving.value) return
    pitchSideSaving.value = true
    options.playing.value = false
    options.sourceVideo.value?.pause()
    try {
      const updated = await calibrationClient.setAttackingGoal(options.projectId(), scene.id, side)
      const selection = preserveSceneActorSelection(updated, {
        trackId: options.selectedTrackId.value,
        canonicalPersonId: options.selectedCanonicalPersonId.value,
      })
      options.scene.value = updated
      options.clearFrameAnalysis()
      draft.value = null
      options.selectedTrackId.value = selection.trackId
      options.selectedCanonicalPersonId.value = selection.canonicalPersonId
      options.saveState.value = `Attack direction set to ${side} · calibration unchanged`
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not change pitch side'
    } finally {
      pitchSideSaving.value = false
    }
  }

  function reset() {
    draft.value = null
    draggedAnchor.value = null
  }

  return {
    draft,
    activeDraft: presentation.activeDraft,
    preset,
    loading,
    applying,
    pitchSideSaving,
    overlay,
    draggedAnchor,
    diagnostics: presentation.diagnostics,
    warnings: presentation.warnings,
    pitchCalibration: presentation.pitchCalibration,
    visiblePitchSide: presentation.visiblePitchSide,
    visiblePitchSideSource: presentation.visiblePitchSideSource,
    attackingGoalSide: presentation.attackingGoalSide,
    activeQaFrame: presentation.activeQaFrame,
    qaFrameSize: presentation.qaFrameSize,
    qaMarkings: presentation.qaMarkings,
    presets: PITCH_CALIBRATION_PRESETS,
    open,
    changePreset,
    updateDraggedAnchor,
    startAnchorDrag,
    finishAnchorDrag,
    nudgeAnchor,
    close,
    apply,
    calibrateQaFrame,
    changeAttackingGoal,
    reset,
  }
}
