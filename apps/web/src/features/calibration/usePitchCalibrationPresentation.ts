import { computed, type Ref } from 'vue'
import {
  calibrationFrameDiagnostics,
  calibrationPreviewWarnings,
} from '../../lib/calibrationDiagnostics'
import { projectPitchMarkings } from '../../lib/pitchProjection'
import type { CalibrationFrameEvidence, PitchCalibrationDraft } from '../../types/calibration'
import type { SceneDocument } from '../../types/scene'

type PitchCalibrationPresentationOptions = {
  scene: Ref<SceneDocument | null>
  currentTime: Ref<number>
  activeTab: Ref<'binding' | 'qa' | 'events'>
  sourceVideo: Ref<HTMLVideoElement | null>
  frames: Readonly<Ref<CalibrationFrameEvidence[]>>
  draft: Ref<PitchCalibrationDraft | null>
}

/** Derives read-only overlay and QA state without owning editor commands. */
export function usePitchCalibrationPresentation(options: PitchCalibrationPresentationOptions) {
  const pitchCalibration = computed(() => options.scene.value?.payload.videoAsset?.reconstruction?.pitchCalibration)
  const activeDraft = computed(() => {
    const value = options.draft.value
    return value && Math.abs(options.currentTime.value - value.sceneTime) <= 0.11 ? value : null
  })
  const diagnostics = computed(() => (
    activeDraft.value ? calibrationFrameDiagnostics(activeDraft.value, options.frames.value) : null
  ))
  const warnings = computed(() => options.draft.value ? calibrationPreviewWarnings(options.draft.value) : [])
  const visiblePitchSide = computed<'left' | 'right' | 'unknown'>(() => {
    const explicit = options.scene.value?.payload.videoAsset?.reconstruction?.pitchOrientation?.visiblePitchSide
    if (explicit) return explicit
    const calibrated = pitchCalibration.value?.pitchSide
    if (calibrated) return calibrated
    const landmark = pitchCalibration.value?.preset ?? pitchCalibration.value?.rectangle
    if (landmark?.endsWith('-left')) return 'left'
    if (landmark?.endsWith('-right')) return 'right'
    return 'unknown'
  })
  const visiblePitchSideSource = computed(() => (
    options.scene.value?.payload.videoAsset?.reconstruction?.pitchOrientation?.visiblePitchSideSource
    ?? (visiblePitchSide.value === 'unknown' ? 'unknown' : 'calibration')
  ))
  const attackingGoalSide = computed<'left' | 'right' | 'unknown'>(() => (
    options.scene.value?.payload.videoAsset?.reconstruction?.pitchOrientation?.attackingGoal ?? 'unknown'
  ))
  const activeQaFrame = computed<CalibrationFrameEvidence | null>(() => {
    if (options.activeTab.value !== 'qa' || !options.frames.value.length) return null
    return options.frames.value.reduce((nearest, frame) => (
      Math.abs(frame.sceneTime - options.currentTime.value) < Math.abs(nearest.sceneTime - options.currentTime.value)
        ? frame
        : nearest
    ))
  })
  const qaFrameSize = computed(() => ({
    width: activeQaFrame.value?.frameWidth ?? options.sourceVideo.value?.videoWidth ?? 960,
    height: activeQaFrame.value?.frameHeight ?? options.sourceVideo.value?.videoHeight ?? 540,
  }))
  const qaMarkings = computed(() => {
    const frame = activeQaFrame.value
    if (!frame) return []
    if (frame.markings?.length) return frame.markings
    return projectPitchMarkings(frame.imageToPitch, qaFrameSize.value.width, qaFrameSize.value.height)
  })

  return {
    activeDraft,
    diagnostics,
    warnings,
    pitchCalibration,
    visiblePitchSide,
    visiblePitchSideSource,
    attackingGoalSide,
    activeQaFrame,
    qaFrameSize,
    qaMarkings,
  }
}
