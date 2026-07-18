import { computed, ref, type Ref } from 'vue'
import { frameAnalysisClient } from '../lib/api/frameAnalysis'
import { mergeFrameReconstructionMetadata } from '../lib/reconstructionUi'
import { renderTrackForFramePerson } from '../lib/videoTrackSelection'
import {
  frameAnnotationDraftFromAnnotation,
  frameAnnotationDraftFromPerson,
  frameAnnotationWrite,
  normalizeFrameAnnotationAction,
  type FrameAnnotationDraft,
} from '../features/frame-annotations/frameAnnotationDraft'
import {
  buildFrameIdentityMergeTargets,
  buildFrameIdentitySplitPreview,
  frameIdentitySaveIsDisabled,
} from '../features/frame-annotations/frameAnnotationRules'
import { useFrameAnnotationPointer } from '../features/frame-annotations/useFrameAnnotationPointer'
import type { CanonicalPerson } from '../types/identity'
import type { FrameAnalysis, FrameAnnotation, FrameAnnotationKind, FrameIdentityAction } from '../types/analysis'
import type { SceneDocument } from '../types/scene'

type FrameAnnotationsOptions = {
  scene: Ref<SceneDocument | null>
  analysis: Ref<FrameAnalysis | null>
  activeAnalysis: Readonly<Ref<FrameAnalysis | null>>
  selectedTrackId: Ref<string | null>
  selectedCanonicalPersonId: Ref<string | null>
  selectedFramePersonId: Ref<string | null>
  activeTab: Ref<'binding' | 'qa' | 'events'>
  playing: Ref<boolean>
  sourceVideo: Ref<HTMLVideoElement | null>
  viewMode: Ref<'video' | 'split' | '3d'>
  mutationLocked: Readonly<Ref<boolean>>
  saveState: Ref<string>
  error: Ref<string | null>
  projectId: () => string
  analyzeFrame: () => Promise<FrameAnalysis | null>
  clearFrameAnalysis: () => void
  seekTo: (time: number) => void
  queueRebuild: (sceneId: string, sceneTime: number, label: string) => void
  canonicalPersonById: (id: string | null | undefined) => CanonicalPerson | null
  framePersonCanonicalId: (person: FrameAnalysis['people'][number]) => string | null
  framePersonLabel: (person: FrameAnalysis['people'][number]) => string
}

export const FRAME_ANNOTATION_KINDS: Array<{ value: FrameAnnotationKind; label: string }> = [
  { value: 'home-player', label: 'Home player' },
  { value: 'away-player', label: 'Away player' },
  { value: 'home-goalkeeper', label: 'Home goalkeeper' },
  { value: 'away-goalkeeper', label: 'Away goalkeeper' },
  { value: 'referee', label: 'Referee' },
  { value: 'other', label: 'Other person' },
]

export const FRAME_IDENTITY_ACTIONS: Array<{ value: FrameIdentityAction; label: string }> = [
  { value: 'confirm', label: 'Confirm in tracking' },
  { value: 'exclude', label: 'Exclude detection' },
  { value: 'merge', label: 'Merge with identity' },
  { value: 'split', label: 'Split identity here / range' },
]

/** Manual corrections over one analyzed source frame. */
export function useFrameAnnotations(options: FrameAnnotationsOptions) {
  const mode = ref(false)
  const draft = ref<FrameAnnotationDraft | null>(null)
  const saving = ref(false)

  const mergeTargets = computed(() => {
    const scene = options.scene.value
    const currentDraft = draft.value
    return scene && currentDraft
      ? buildFrameIdentityMergeTargets(scene, currentDraft, options.canonicalPersonById)
      : []
  })
  const saveDisabled = computed(() => frameIdentitySaveIsDisabled(
    options.scene.value,
    draft.value,
    options.activeAnalysis.value?.sceneTime ?? null,
    new Set(mergeTargets.value.map((target) => target.id)),
  ))
  const splitPreview = computed(() => buildFrameIdentitySplitPreview(
    draft.value,
    options.activeAnalysis.value?.sceneTime ?? null,
    options.canonicalPersonById,
  ))

  async function toggleMode() {
    if (mode.value) {
      mode.value = false
      draft.value = null
      pointer.clear()
      options.saveState.value = 'Frame labeling closed'
      return
    }
    if (!options.activeAnalysis.value) await options.analyzeFrame()
    if (!options.analysis.value) return
    mode.value = true
    options.viewMode.value = 'split'
    options.playing.value = false
    options.sourceVideo.value?.pause()
    options.activeTab.value = 'binding'
    options.saveState.value = 'Select a box or drag around any person'
  }

  function selectDetectedPerson(person: FrameAnalysis['people'][number]) {
    options.selectedFramePersonId.value = person.id
    const canonicalPersonId = options.framePersonCanonicalId(person)
    const linkedTrackId = renderTrackForFramePerson(
      person,
      options.scene.value?.payload.tracks ?? [],
    )?.id ?? null
    options.selectedCanonicalPersonId.value = canonicalPersonId
    options.selectedTrackId.value = linkedTrackId
    if (!mode.value) {
      options.saveState.value = linkedTrackId
        ? `${options.framePersonLabel(person)} selected in video and 3D`
        : canonicalPersonId
          ? `${options.framePersonLabel(person)} selected · identity exists, not projected in 3D`
          : `${options.framePersonLabel(person)} selected · identity is not resolved yet`
      return
    }
    const annotations = options.scene.value?.payload.videoAsset?.reconstruction?.frameAnnotations ?? []
    const selection = frameAnnotationDraftFromPerson(person, {
      annotations,
      linkedTrackId,
      canonicalPersonId,
      sceneTime: options.activeAnalysis.value?.sceneTime ?? 0,
      duration: options.scene.value?.duration ?? 0,
    })
    draft.value = selection.draft
    options.saveState.value = selection.source === 'saved'
      ? 'Editing saved frame label'
      : selection.source === 'dedicated-roster'
        ? 'New semantic correction · roster Bind / Unbind remains separate'
        : 'Detection selected for labeling'
  }

  function selectAnnotation(annotation: FrameAnnotation) {
    if (!mode.value) return
    const dedicatedRosterCorrection = annotation.correctionKind === 'canonical-roster-binding-v1'
    draft.value = frameAnnotationDraftFromAnnotation(
      annotation,
      options.scene.value?.payload.tracks ?? [],
      options.scene.value?.duration ?? 0,
    )
    options.saveState.value = dedicatedRosterCorrection
      ? 'New semantic correction · roster Bind / Unbind remains separate'
      : 'Editing saved frame label'
  }

  function onActionChange() {
    const currentDraft = draft.value
    if (!currentDraft) return
    draft.value = normalizeFrameAnnotationAction(
      currentDraft,
      options.activeAnalysis.value?.sceneTime ?? null,
      options.scene.value?.duration ?? null,
    )
  }

  const pointer = useFrameAnnotationPointer({
    analysis: options.activeAnalysis,
    mode,
    draft,
    saveState: options.saveState,
    selectPerson: selectDetectedPerson,
  })

  function syncAnnotations(result: FrameAnalysis) {
    const reconstruction = options.scene.value?.payload.videoAsset?.reconstruction
    if (!reconstruction) return
    reconstruction.frameAnnotations = [
      ...(reconstruction.frameAnnotations ?? []).filter((item) => item.frameIndex !== result.frameIndex),
      ...result.annotations,
    ].sort((left, right) => left.frameIndex - right.frameIndex || left.id.localeCompare(right.id))
  }

  async function save() {
    const currentDraft = draft.value
    const source = options.activeAnalysis.value
    const scene = options.scene.value
    if (!scene || !currentDraft || !source || saving.value || options.mutationLocked.value) return
    saving.value = true
    try {
      const result = await frameAnalysisClient.saveAnnotation(
        options.projectId(),
        scene.id,
        frameAnnotationWrite(currentDraft, source.sceneTime),
      )
      if (options.scene.value?.id !== scene.id) return
      options.scene.value = mergeFrameReconstructionMetadata(options.scene.value, result)
      syncAnnotations(result)
      options.clearFrameAnalysis()
      options.seekTo(result.sceneTime)
      const correctionLabel = currentDraft.action === 'merge'
        ? 'Identity merge'
        : currentDraft.action === 'split'
          ? 'Identity split'
          : currentDraft.action === 'exclude'
            ? 'Exclusion'
            : 'Tracking confirmation'
      if (currentDraft.action === 'merge' && currentDraft.mergeTargetId) {
        const targetTrack = options.scene.value.payload.tracks.find((track) => (
          track.id === currentDraft.mergeTargetId
          || track.canonicalPersonId === currentDraft.mergeTargetId
        )) ?? null
        options.selectedTrackId.value = targetTrack?.id ?? null
        options.selectedCanonicalPersonId.value = options.canonicalPersonById(
          currentDraft.mergeTargetId,
        )?.canonicalPersonId ?? targetTrack?.canonicalPersonId ?? null
      }
      if (currentDraft.action === 'exclude' && currentDraft.scope === 'identity') {
        options.selectedTrackId.value = null
        options.selectedCanonicalPersonId.value = null
      }
      options.queueRebuild(scene.id, result.sceneTime, correctionLabel)
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not save frame label'
    } finally {
      saving.value = false
    }
  }

  async function remove() {
    const annotationId = draft.value?.annotationId
    const scene = options.scene.value
    if (!scene || !annotationId || saving.value || options.mutationLocked.value) return
    saving.value = true
    try {
      const result = await frameAnalysisClient.deleteAnnotation(options.projectId(), scene.id, annotationId)
      if (options.scene.value?.id !== scene.id) return
      options.scene.value = mergeFrameReconstructionMetadata(options.scene.value, result)
      syncAnnotations(result)
      options.clearFrameAnalysis()
      options.seekTo(result.sceneTime)
      options.queueRebuild(scene.id, result.sceneTime, 'Correction removal')
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not remove frame label'
    } finally {
      saving.value = false
    }
  }

  function clear() {
    mode.value = false
    draft.value = null
    pointer.clear()
  }

  return {
    overlay: pointer.overlay,
    mode,
    draft,
    drag: pointer.drag,
    saving,
    mergeTargets,
    saveDisabled,
    splitPreview,
    kinds: FRAME_ANNOTATION_KINDS,
    actions: FRAME_IDENTITY_ACTIONS,
    toggleMode,
    selectDetectedPerson,
    selectAnnotation,
    onActionChange,
    selectAtPoint: pointer.selectAtPoint,
    startDrag: pointer.startDrag,
    updateDrag: pointer.updateDrag,
    finishDrag: pointer.finishDrag,
    save,
    remove,
    clear,
  }
}
