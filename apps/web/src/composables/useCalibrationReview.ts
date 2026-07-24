import { ref, type Ref } from 'vue'
import { calibrationClient } from '../lib/api/calibration'
import { clientPointToContainedMedia } from '../lib/videoReviewTransform'
import {
  PITCH_CALIBRATION_PRESETS,
  projectPitchCalibrationPresetAnchors,
} from '../lib/pitchCalibrationPresets'
import type {
  CalibrationBorrowSource,
  CalibrationDraftSource,
  PitchCalibrationAnchor,
  PitchCalibrationDraft,
  PitchCalibrationPreset,
} from '../types/calibration'
import type { CalibrationReviewSample } from '../types/reconstruction'

type CalibrationReviewOptions = {
  projectId: () => string
  sceneId: () => string | null
  error: Ref<string | null>
  /** Persist this frame's anchors as a draft. Finalization is a separate command. */
  saveFrame: (
    sceneTime: number,
    preset: PitchCalibrationPreset,
    anchors: PitchCalibrationAnchor[],
    source: CalibrationDraftSource,
    acceptQualityWarning?: boolean,
  ) => Promise<void>
}

/**
 * Self-contained pitch calibration for a single frame, scoped to the review
 * modal. It owns its own draft and anchor editing and never touches the main
 * viewport — no shared draft, no view-mode change, no cross-window seeking.
 */
export function useCalibrationReview(options: CalibrationReviewOptions) {
  const draft = ref<PitchCalibrationDraft | null>(null)
  const preset = ref<PitchCalibrationPreset>('penalty-area-right')
  const overlay = ref<SVGSVGElement | null>(null)
  const draggedAnchor = ref<string | null>(null)
  const busy = ref(false) // an auto/preview scoring request is in flight
  const saving = ref(false) // a recalibration was requested for this frame
  const activeSceneTime = ref<number | null>(null)

  async function editFrame(sceneTime: number, hintPreset?: PitchCalibrationPreset) {
    const sceneId = options.sceneId()
    if (!sceneId || busy.value) return
    busy.value = true
    activeSceneTime.value = sceneTime
    if (hintPreset) preset.value = hintPreset
    try {
      draft.value = await calibrationClient.auto(
        options.projectId(),
        sceneId,
        sceneTime,
        preset.value,
      )
      preset.value = draft.value.preset
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not prepare the pitch overlay'
    } finally {
      busy.value = false
    }
  }

  async function prepareStoredFrame(
    frame: CalibrationReviewSample,
    hintPreset?: PitchCalibrationPreset,
  ) {
    if (frame.sceneTime == null || busy.value) return
    const selectedPreset = hintPreset ?? preset.value
    preset.value = selectedPreset
    activeSceneTime.value = frame.sceneTime
    const width = frame.frameWidth ?? 960
    const height = frame.frameHeight ?? 540
    const anchors = projectPitchCalibrationPresetAnchors(
      frame.imageToPitch,
      selectedPreset,
      width,
      height,
    )
    busy.value = true
    try {
      const sceneId = options.sceneId()
      if (!sceneId) return
      draft.value = await calibrationClient.preview(
        options.projectId(),
        sceneId,
        frame.sceneTime,
        selectedPreset,
        anchors,
      )
      preset.value = draft.value.preset
    } catch (cause) {
      options.error.value = cause instanceof Error
        ? cause.message
        : 'Could not prepare the stored frame calibration'
    } finally {
      busy.value = false
    }
  }

  async function changePreset(event: Event) {
    const value = draft.value
    if (!value || activeSceneTime.value === null || busy.value) return
    const selectedPreset = (event.target as HTMLSelectElement).value as PitchCalibrationPreset
    preset.value = selectedPreset
    const anchors = projectPitchCalibrationPresetAnchors(
      value.imageToPitch,
      selectedPreset,
      value.frameWidth,
      value.frameHeight,
    )
    draft.value = {
      ...value,
      preset: selectedPreset,
      anchors,
    }
    await previewAnchors()
  }

  async function borrowFrame(
    sceneTime: number,
    source: CalibrationBorrowSource,
  ) {
    const sceneId = options.sceneId()
    if (!sceneId || busy.value) return
    busy.value = true
    activeSceneTime.value = sceneTime
    try {
      draft.value = await calibrationClient.borrow(
        options.projectId(),
        sceneId,
        sceneTime,
        source,
        preset.value,
      )
      preset.value = draft.value.preset
    } catch (cause) {
      options.error.value = cause instanceof Error
        ? cause.message
        : 'Could not borrow a neighboring frame calibration'
    } finally {
      busy.value = false
    }
  }

  function bindOverlay(element: SVGSVGElement | null) {
    overlay.value = element
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

  async function previewAnchors() {
    const sceneId = options.sceneId()
    const value = draft.value
    if (!sceneId || !value || busy.value) return
    busy.value = true
    try {
      draft.value = await calibrationClient.preview(
        options.projectId(),
        sceneId,
        value.sceneTime,
        preset.value,
        value.anchors,
      )
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not update the pitch overlay'
    } finally {
      busy.value = false
    }
  }

  function finishAnchorDrag(event: PointerEvent) {
    if (!draggedAnchor.value) return
    updateDraggedAnchor(event)
    draggedAnchor.value = null
    try {
      if (overlay.value?.hasPointerCapture(event.pointerId)) {
        overlay.value.releasePointerCapture(event.pointerId)
      }
    } catch {
      // The drag is already complete when the browser releases capture first.
    }
    void previewAnchors()
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
    void previewAnchors()
  }

  async function saveFrame(acceptQualityWarning = false) {
    const value = draft.value
    if (!value || saving.value) return
    saving.value = true
    try {
      await options.saveFrame(
        value.sceneTime,
        preset.value,
        value.anchors,
        value.source,
        acceptQualityWarning,
      )
      cancel()
    } catch {
      // The saveFrame callback already surfaced the failure.
    } finally {
      saving.value = false
    }
  }

  function cancel() {
    draft.value = null
    draggedAnchor.value = null
    activeSceneTime.value = null
  }

  return {
    draft,
    preset,
    overlay,
    draggedAnchor,
    busy,
    saving,
    activeSceneTime,
    presets: PITCH_CALIBRATION_PRESETS,
    editFrame,
    prepareStoredFrame,
    changePreset,
    borrowFrame,
    bindOverlay,
    startAnchorDrag,
    updateDraggedAnchor,
    finishAnchorDrag,
    nudgeAnchor,
    previewAnchors,
    saveFrame,
    cancel,
  }
}
