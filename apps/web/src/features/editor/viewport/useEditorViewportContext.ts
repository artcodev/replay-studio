import { computed, onMounted, ref, watch } from 'vue'
import { useMultiPassPlayback } from '../../../composables/useMultiPassPlayback'
import { usePlaybackClock } from '../../../composables/usePlaybackClock'
import type { SceneDocumentState } from '../../../composables/useSceneDocumentState'
import {
  DEFAULT_THREE_VIEW_OPTIONS,
  type ThreeRenderQuality,
  type ThreeViewOptions,
} from '../../../lib/threeViewOptions'
import {
  loadThreeViewPreferences,
  saveThreeViewPreferences,
} from '../../../lib/threeViewPreferences'

export type EditorCameraName = 'broadcast' | 'orbit' | 'tactical' | 'goal'
export type EditorViewMode = 'video' | 'split' | '3d'
export type EditorInspectorTab = 'binding' | 'qa' | 'events'
export type EditorViewportApi = { cameraPreset: (name: EditorCameraName) => void }

/** Playback, camera and transient selection state for the editor viewport. */
export function useEditorViewportContext(document: SceneDocumentState) {
  const { scene } = document
  const selectedTrackId = ref<string | null>(null)
  const selectedCanonicalPersonId = ref<string | null>(null)
  const selectedFramePersonId = ref<string | null>(null)
  const trackQuery = ref('')
  const editMode = ref(false)
  const viewOptions = ref<ThreeViewOptions>({ ...DEFAULT_THREE_VIEW_OPTIONS })
  const renderQuality = ref<ThreeRenderQuality>('basic')
  const viewMode = ref<EditorViewMode>('split')
  const activeCamera = ref<EditorCameraName>('broadcast')
  const viewport = ref<EditorViewportApi | null>(null)
  const activeTab = ref<EditorInspectorTab>('binding')
  const activePassSceneId = ref<string | null>(null)
  const sceneVideo = computed(() => scene.value?.payload.videoAsset ?? null)

  const multiPassPlayback = useMultiPassPlayback({ scene, sceneVideo, activePassSceneId })
  const playbackClock = usePlaybackClock({
    duration: () => scene.value?.duration ?? 0,
    hasSourceVideo: () => Boolean(sceneVideo.value),
    sourceStart: () => multiPassPlayback.sourceStart.value,
    sourceEnd: () => multiPassPlayback.sourceEnd.value,
    canonicalToSourceTime: multiPassPlayback.canonicalToPassTime,
    sourceToCanonicalTime: multiPassPlayback.passToCanonicalTime,
  })
  const {
    currentTime,
    playing,
    playbackRate,
    sourceVideo,
    seek: seekTo,
    pauseAtPlayhead: onTimelineInput,
    toggle: togglePlay,
  } = playbackClock

  const timeLabel = computed(() => {
    const minutes = Math.floor(currentTime.value / 60)
    const remaining = currentTime.value % 60
    return `${String(minutes).padStart(2, '0')}:${remaining.toFixed(2).padStart(5, '0')}`
  })

  function chooseSourcePass(event: Event) {
    activePassSceneId.value = (event.target as HTMLSelectElement).value
    seekTo(currentTime.value)
  }

  function setCamera(name: EditorCameraName) {
    activeCamera.value = name
    viewport.value?.cameraPreset(name)
  }

  function onCameraPresetChange(event: Event) {
    setCamera((event.target as HTMLSelectElement).value as EditorCameraName)
  }

  onMounted(() => {
    const saved = loadThreeViewPreferences(window.localStorage)
    if (!saved) return
    viewOptions.value = saved.options
    renderQuality.value = saved.renderQuality
  })

  watch([viewOptions, renderQuality], ([options, quality]) => {
    saveThreeViewPreferences(window.localStorage, {
      options: { ...options },
      renderQuality: quality,
    })
  }, { deep: true })

  watch(() => scene.value?.id, () => {
    playing.value = false
    sourceVideo.value?.pause()
    viewMode.value = sceneVideo.value ? 'split' : '3d'
    trackQuery.value = ''
    activeTab.value = 'binding'
    activePassSceneId.value = multiPassPlayback.analysis.value?.referenceSceneId ?? null
    // A route may replace one segment with another from the same media URL
    // while the local playhead is already zero. Calling seek explicitly keeps
    // the reused video element aligned to the new sourceStart; assigning the
    // unchanged currentTime ref would not trigger the playback watcher.
    seekTo(0)
  })

  watch(() => multiPassPlayback.analysis.value?.referenceSceneId, (referenceSceneId) => {
    if (!activePassSceneId.value && referenceSceneId) activePassSceneId.value = referenceSceneId
  })

  return {
    sceneVideo,
    selectedTrackId,
    selectedCanonicalPersonId,
    selectedFramePersonId,
    trackQuery,
    editMode,
    viewOptions,
    renderQuality,
    viewMode,
    activeCamera,
    viewport,
    activeTab,
    activePassSceneId,
    multiPassPlayback,
    playbackClock,
    currentTime,
    playing,
    playbackRate,
    sourceVideo,
    seekTo,
    onTimelineInput,
    togglePlay,
    timeLabel,
    chooseSourcePass,
    onCameraPresetChange,
  }
}

export type EditorViewportContext = ReturnType<typeof useEditorViewportContext>
