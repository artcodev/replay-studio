import { nextTick, ref, type Ref } from 'vue'
import { mediaClient } from '../lib/api/media'
import { sceneClient } from '../lib/api/scenes'
import type { SceneDocument, SceneVideoAsset } from '../types/scene'
import type { VideoSegment } from '../types/media'
import type { ProjectSegment } from '../types/project'

type CompositionEditorOptions = {
  projectId: () => string
  scene: Ref<SceneDocument | null>
  sceneVideo: Readonly<Ref<SceneVideoAsset | null>>
  projectSegments: Readonly<Ref<ProjectSegment[]>>
  multiPassSelection: Ref<string[]>
  selectedTrackId: Ref<string | null>
  selectedCanonicalPersonId: Ref<string | null>
  currentTime: Ref<number>
  reconstructing: Ref<boolean>
  saveState: Ref<string>
  error: Ref<string | null>
  navigateToScene: (sceneId: string) => Promise<void>
  startReconstructionPolling: (sceneId: string) => Promise<void>
  seekTo: (time: number) => void
}

/** Creates single-segment scenes and multi-angle compositions from timeline choices. */
export function useCompositionEditor(options: CompositionEditorOptions) {
  const segmentCreating = ref<string | null>(null)
  const multiPassStarting = ref(false)

  async function createSceneFromSegment(segment: VideoSegment) {
    const video = options.sceneVideo.value
    if (!video) return
    segmentCreating.value = segment.id
    try {
      const created = segment.sceneId
        ? await sceneClient.get(options.projectId(), segment.sceneId)
        : await mediaClient.createSegmentScene(options.projectId(), video.id, segment.id)
      options.scene.value = created
      options.selectedTrackId.value = null
      options.selectedCanonicalPersonId.value = null
      options.currentTime.value = 0
      options.saveState.value = 'Shot scene created'
      await options.navigateToScene(created.id)
      await nextTick()
      options.seekTo(0)
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not create shot scene'
    } finally {
      segmentCreating.value = null
    }
  }

  async function startMultiPass() {
    const video = options.sceneVideo.value
    const selected = options.multiPassSelection.value
    if (!video || selected.length < 2 || multiPassStarting.value) return
    const selectedCount = selected.length
    multiPassStarting.value = true
    options.reconstructing.value = true
    options.saveState.value = `Analyzing ${selectedCount} camera angles…`
    try {
      const selectedIds = new Set(selected)
      const projectSegmentIds = options.projectSegments.value
        .filter((segment) => segment.assetId === video.id && selectedIds.has(segment.sourceSegmentId))
        .map((segment) => segment.id)
      if (projectSegmentIds.length !== selectedCount) {
        throw new Error('Project segment index is stale; reload the timeline and try again.')
      }
      const created = await mediaClient.createComposition(options.projectId(), projectSegmentIds)
      options.scene.value = created
      options.selectedTrackId.value = null
      options.selectedCanonicalPersonId.value = null
      options.currentTime.value = 0
      options.multiPassSelection.value = []
      await options.navigateToScene(created.id)
      await options.startReconstructionPolling(created.id)
    } catch (cause) {
      options.reconstructing.value = false
      options.error.value = cause instanceof Error
        ? cause.message
        : 'Could not start multi-angle analysis'
    } finally {
      multiPassStarting.value = false
    }
  }

  return { segmentCreating, multiPassStarting, createSceneFromSegment, startMultiPass }
}
