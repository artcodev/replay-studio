import { computed, ref, type Ref } from 'vue'
import { mediaClient } from '../lib/api/media'
import {
  canSplitSegmentTail,
  normalizeSegmentLayout,
  segmentGroupColor,
  splitSegmentTail,
  type SceneVideo,
} from '../features/timeline/segmentLayout'
import type { SceneDocument } from '../types/scene'
import type { VideoSegment } from '../types/media'

type SegmentLayoutEditorOptions = {
  scene: Ref<SceneDocument | null>
  sceneVideo: Ref<SceneVideo | null>
  selectedTrackId: Ref<string | null>
  selectedCanonicalPersonId: Ref<string | null>
  currentTime: Ref<number>
  saveState: Ref<string>
  error: Ref<string | null>
  projectId: () => string
  saveScene: () => Promise<void>
  seekTo: (time: number) => void
  writeRouteTime: (time: number) => void
  notifySceneMutation: () => void
}

/** Owns user edits to the proposed event/shot grouping. */
export function useSegmentLayoutEditor(options: SegmentLayoutEditorOptions) {
  const selection = ref<string[]>([])
  const rebuilding = ref(false)
  const groupEditing = ref(false)
  const layout = computed(() => options.sceneVideo.value?.segmentLayout ?? null)
  const groupOptions = computed(() => {
    const maximum = Math.max(
      1,
      ...(options.sceneVideo.value?.segments ?? []).map((segment) => segment.layout?.group ?? 1),
    )
    return Array.from({ length: maximum + 1 }, (_, index) => index + 1)
  })
  const canSplitSelection = computed(() => (
    canSplitSegmentTail(options.sceneVideo.value, selection.value)
  ))

  function markEdited(message: string) {
    if (layout.value) layout.value.status = 'edited'
    const video = options.sceneVideo.value
    if (video) normalizeSegmentLayout(video)
    options.notifySceneMutation()
    options.saveState.value = message
  }

  function assignGroup(segment: VideoSegment, value: string) {
    const group = Number(value)
    if (!Number.isFinite(group) || group < 1) return
    segment.layout = {
      group,
      variant: segment.layout?.variant ?? 'A',
      label: segment.layout?.label ?? `${group}-A`,
      role: segment.layout?.role ?? 'continuation',
      confidence: 1,
      motionCost: segment.layout?.motionCost,
    }
    markEdited('Timeline layout has unsaved changes')
  }

  function assignRole(segment: VideoSegment, value: string) {
    if (!segment.layout || !['original', 'replay', 'continuation'].includes(value)) return
    segment.layout.role = value as 'original' | 'replay' | 'continuation'
    segment.layout.confidence = 1
    markEdited('Timeline layout has unsaved changes')
  }

  function splitSelection() {
    const video = options.sceneVideo.value
    if (!video) return
    const newEvent = splitSegmentTail(video, selection.value)
    if (!newEvent) {
      options.error.value = 'Select a continuous group tail, leaving its first segment in the original event.'
      return
    }
    selection.value = []
    options.notifySceneMutation()
    options.saveState.value = `Created Event ${newEvent}; later events shifted`
  }

  function toggleGroupEditing() {
    selection.value = []
    groupEditing.value = !groupEditing.value
  }

  function handleSegment(segment: VideoSegment) {
    if (!groupEditing.value) {
      options.seekTo(segment.start)
      options.writeRouteTime(Number(segment.start.toFixed(3)))
      return
    }
    if (selection.value.includes(segment.id)) {
      selection.value = selection.value.filter((id) => id !== segment.id)
    } else if (selection.value.length < 6) {
      selection.value = [...selection.value, segment.id]
    }
  }

  async function confirm() {
    const video = options.sceneVideo.value
    if (!video?.segmentLayout) return
    normalizeSegmentLayout(video, true)
    video.segmentLayout.status = 'confirmed'
    options.notifySceneMutation()
    await options.saveScene()
  }

  async function saveGroupMap() {
    await confirm()
    groupEditing.value = false
    selection.value = []
  }

  async function rebuild() {
    const video = options.sceneVideo.value
    if (!video || rebuilding.value) return
    rebuilding.value = true
    options.saveState.value = 'Rebuilding event map…'
    try {
      options.scene.value = await mediaClient.proposeSegmentLayout(options.projectId(), video.id)
      options.selectedTrackId.value = null
      options.selectedCanonicalPersonId.value = null
      options.currentTime.value = 0
      options.saveState.value = 'New event map proposed'
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not rebuild the event map'
    } finally {
      rebuilding.value = false
    }
  }

  function reset() {
    selection.value = []
    groupEditing.value = false
  }

  return {
    selection,
    rebuilding,
    groupEditing,
    layout,
    groupOptions,
    canSplitSelection,
    segmentGroupColor,
    assignGroup,
    assignRole,
    splitSelection,
    toggleGroupEditing,
    handleSegment,
    confirm,
    saveGroupMap,
    rebuild,
    reset,
  }
}
