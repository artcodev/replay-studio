import { computed, ref, shallowRef } from 'vue'
import type { RouteLocationNormalizedLoaded, Router } from 'vue-router'
import {
  appRouteLocation,
  projectSegmentRoute,
  projectWorkspaceRoute,
  videoTimelineRoute,
} from '../../../lib/appRoutes'
import { useEditorRouteSession } from '../../../composables/useEditorRouteSession'
import { injectProjectWorkspace } from '../../../composables/useProjectWorkspace'
import type { SceneDocumentState } from '../../../composables/useSceneDocumentState'
import { useSceneSession } from '../../../composables/useSceneSession'
import type { VideoAsset, VideoSegment } from '../../../types/media'
import type { SceneDocument, SceneSummary } from '../../../types/scene'
import type { EditorViewportContext } from '../viewport/useEditorViewportContext'

type EditorSessionContextOptions = {
  route: RouteLocationNormalizedLoaded
  router: Router
  document: SceneDocumentState
  viewport: EditorViewportContext
  exitEditor: () => void | Promise<void>
}

/** Project-scoped scene persistence, route synchronization and navigation. */
export function useEditorSessionContext(options: EditorSessionContextOptions) {
  const { scene, mutate: mutateScene, notifyMutation: notifySceneMutation } = options.document
  const workspace = injectProjectWorkspace()
  const scenes = ref<SceneSummary[]>([])
  const saveState = ref('Saved locally')
  const error = ref<string | null>(null)
  const videoIngestOpen = ref(false)
  const timelineSceneState = shallowRef<SceneDocument | null>(null)
  const timelineScene = computed(() => (
    timelineSceneState.value?.id === scene.value?.id
      ? scene.value
      : timelineSceneState.value
  ))

  function editorProjectId(): string {
    const projectId = workspace.catalog.project.value?.id
    if (!projectId) throw new Error('Open a project before editing its scenes.')
    return projectId
  }

  const sceneSession = useSceneSession({
    projectId: editorProjectId,
    scene,
    scenes,
    saveState,
  })

  async function returnToProjects() {
    options.viewport.playing.value = false
    options.viewport.sourceVideo.value?.pause()
    workspace.catalog.error.value = null
    await options.exitEditor()
  }

  const routeSession = useEditorRouteSession({
    route: options.route,
    router: options.router,
    workspace,
    scene,
    timelineScene: timelineSceneState,
    sceneSession,
    playing: options.viewport.playing,
    sourceVideo: options.viewport.sourceVideo,
    error,
    selectedTrackId: options.viewport.selectedTrackId,
    selectedCanonicalPersonId: options.viewport.selectedCanonicalPersonId,
    activeTab: options.viewport.activeTab,
    seekTo: options.viewport.seekTo,
    exitEditor: () => { void returnToProjects() },
  })

  const projects = computed(() => scenes.value.filter((item) => item.kind === 'video'))
  const activeProjectId = computed(() => {
    if (!scene.value) return null
    return options.viewport.sceneVideo.value?.multiPass?.parentSceneId
      ?? options.viewport.sceneVideo.value?.parentSceneId
      ?? (projects.value.some((item) => item.id === scene.value?.id) ? scene.value.id : null)
  })
  const internalSceneLabel = computed(() => {
    if (!scene.value || scene.value.id === activeProjectId.value) return null
    if (options.viewport.multiPassPlayback.analysis.value) return 'Multi-angle reconstruction'
    if (options.viewport.sceneVideo.value?.selectedSegmentId) return 'Segment reconstruction'
    return 'Internal scene'
  })

  function updateSceneTitle(title: string) {
    if (!mutateScene((document) => { document.title = title })) return
    saveState.value = 'Unsaved changes'
  }

  async function openTimelineSegment(segment: VideoSegment) {
    const activeScene = scene.value
    const rootScene = timelineScene.value
    const rootVideo = rootScene?.payload.videoAsset
    if (!activeScene || !rootScene || !rootVideo) return

    const asset = workspace.media.assets.value.find(
      (item) => item.timelineSceneId === rootScene.id,
    )
    if (!asset) {
      error.value = 'The source timeline is no longer linked to this project.'
      return
    }
    const projectSegment = workspace.media.segments.value.find((item) => (
      item.assetId === asset.id && item.sourceSegmentId === segment.id
    ))
    if (projectSegment?.sceneId === activeScene.id) {
      const sourceStart = activeScene.payload.videoAsset?.sourceStart ?? segment.start
      const localTime = Math.max(0, segment.start - sourceStart)
      options.viewport.seekTo(localTime)
      routeSession.replaceView({ time: Number(localTime.toFixed(3)) })
      return
    }
    if (projectSegment?.sceneId) {
      await options.router.push(appRouteLocation(
        projectSegmentRoute(editorProjectId(), projectSegment.id),
      ))
      return
    }
    await options.router.push(appRouteLocation(videoTimelineRoute(
      editorProjectId(),
      asset.id,
      { time: Number(segment.start.toFixed(3)) },
    )))
  }

  async function openProcessedVideo(asset: VideoAsset) {
    videoIngestOpen.value = false
    const projectId = workspace.catalog.project.value?.id
    if (projectId) await workspace.load(projectId)
    if (!asset.scene_id) {
      workspace.catalog.error.value = 'The video is ready, but its editor scene is not available yet.'
      if (projectId) {
        await options.router.push(appRouteLocation(projectWorkspaceRoute(projectId, 'analysis')))
      }
      return
    }
    await workspace.media.openTimeline(asset.id)
    options.viewport.currentTime.value = 0
    options.viewport.selectedTrackId.value = null
    options.viewport.selectedCanonicalPersonId.value = null
    options.viewport.viewMode.value = 'split'
    saveState.value = 'Video timeline ready'
  }

  return {
    router: options.router,
    workspace,
    workspaceProjects: workspace.catalog.projects,
    selectedProject: workspace.catalog.project,
    workspaceMatch: workspace.match.snapshot,
    workspaceAssets: workspace.media.assets,
    workspaceSegments: workspace.media.segments,
    projectLoading: workspace.catalog.loading,
    projectError: workspace.catalog.error,
    analysisJobs: workspace.jobs,
    projectAnalysisJobs: workspace.jobs.jobs,
    projectCancelingJobIds: workspace.jobs.cancelingJobIds,
    loadProjectsWorkspace: workspace.load,
    scene,
    timelineScene,
    scenes,
    mutateScene,
    notifySceneMutation,
    saveState,
    error,
    videoIngestOpen,
    sceneSession,
    saving: sceneSession.saving,
    saveScene: sceneSession.save,
    routeSession,
    replaceEditorRouteView: routeSession.replaceView,
    navigateEditorScene: routeSession.navigateScene,
    projects,
    activeProjectId,
    internalSceneLabel,
    updateSceneTitle,
    openTimelineSegment,
    openProcessedVideo,
    returnToProjects,
    editorProjectId,
  }
}

export type EditorSessionContext = ReturnType<typeof useEditorSessionContext>
