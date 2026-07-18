import { nextTick, onScopeDispose, watch, type Ref, type ShallowRef } from 'vue'
import type { RouteLocationNormalizedLoaded, RouteLocationRaw, Router } from 'vue-router'
import type { ProjectWorkspace } from './useProjectWorkspace'
import type { useSceneSession } from './useSceneSession'
import {
  APP_ROUTE_NAMES,
  appRouteLocation,
  parseAppRoute,
  projectSceneRoute,
  projectSegmentRoute,
  projectWorkspaceRoute,
  videoTimelineRoute,
  type AppRouteIntent,
  type EditorRouteView,
} from '../lib/appRoutes'
import type { SceneDocument } from '../types/scene'

type SceneSession = ReturnType<typeof useSceneSession>

type EditorRouteSessionOptions = {
  route: RouteLocationNormalizedLoaded
  router: Router
  workspace: ProjectWorkspace
  scene: ShallowRef<SceneDocument | null>
  timelineScene: ShallowRef<SceneDocument | null>
  sceneSession: SceneSession
  playing: Ref<boolean>
  sourceVideo: Ref<HTMLVideoElement | null>
  error: Ref<string | null>
  selectedTrackId: Ref<string | null>
  selectedCanonicalPersonId: Ref<string | null>
  activeTab: Ref<'binding' | 'qa' | 'events'>
  seekTo: (time: number) => void
  exitEditor: () => void
}

function routeLocation(intent: AppRouteIntent): RouteLocationRaw {
  return appRouteLocation(intent) as RouteLocationRaw
}

/** Owns URL ↔ editor-document synchronization and fences stale route loads. */
export function useEditorRouteSession(options: EditorRouteSessionOptions) {
  let requestId = 0

  function replaceView(patch: Partial<EditorRouteView>) {
    const intent = parseAppRoute(options.route.fullPath)
    if (
      !intent
      || intent.name === APP_ROUTE_NAMES.projects
      || intent.name === APP_ROUTE_NAMES.projectWorkspace
    ) return
    void options.router.replace(routeLocation({
      ...intent,
      view: { ...intent.view, ...patch },
    }))
  }

  async function navigateScene(sceneId: string) {
    const projectId = options.workspace.catalog.project.value?.id
    if (!projectId || !sceneId) return
    const timelineAsset = options.workspace.media.assets.value.find(
      (asset) => asset.timelineSceneId === sceneId,
    )
    if (timelineAsset) {
      await options.router.push(routeLocation(videoTimelineRoute(projectId, timelineAsset.id)))
      return
    }
    const segment = options.workspace.media.segments.value.find((item) => item.sceneId === sceneId)
    if (segment) {
      await options.router.push(routeLocation(projectSegmentRoute(projectId, segment.id)))
      return
    }
    await options.router.push(routeLocation(projectSceneRoute(projectId, sceneId)))
  }

  async function sync() {
    const activeRequest = ++requestId
    const intent = parseAppRoute(options.route.fullPath)
    if (!intent || intent.name === APP_ROUTE_NAMES.projects) {
      options.exitEditor()
      return
    }

    const workspace = options.workspace
    const needsWorkspaceLoad = !workspace.catalog.projects.value.length
      || workspace.catalog.project.value?.id !== intent.projectId
    const outcome = needsWorkspaceLoad ? await workspace.load(intent.projectId) : 'loaded'
    if (activeRequest !== requestId) return
    if (outcome === 'not-found') {
      options.exitEditor()
      return
    }
    if (outcome === 'failed' || outcome === 'stale') return

    const project = workspace.catalog.project.value
    if (!project || project.id !== intent.projectId) {
      workspace.catalog.error.value = `Project ${intent.projectId} was not found.`
      options.exitEditor()
      return
    }
    if (intent.name === APP_ROUTE_NAMES.projectWorkspace) {
      options.playing.value = false
      options.sourceVideo.value?.pause()
      options.error.value = null
      return
    }

    let targetSceneId: string
    let timelineSceneId: string
    if (intent.name === APP_ROUTE_NAMES.videoTimeline) {
      const asset = workspace.media.assets.value.find((item) => item.id === intent.assetId)
      if (!asset) {
        workspace.catalog.error.value = `Video ${intent.assetId} does not belong to this project.`
        await options.router.replace(routeLocation(projectWorkspaceRoute(project.id, 'overview')))
        return
      }
      if (!asset.timelineSceneId) {
        workspace.catalog.error.value = `${asset.filename} is still being prepared. Its timeline is not available yet.`
        await options.router.replace(routeLocation(projectWorkspaceRoute(project.id, 'analysis')))
        return
      }
      targetSceneId = asset.timelineSceneId
      timelineSceneId = asset.timelineSceneId
    } else if (intent.name === APP_ROUTE_NAMES.projectSegment) {
      const segment = workspace.media.segments.value.find((item) => item.id === intent.segmentId)
      if (!segment) {
        workspace.catalog.error.value = `Moment ${intent.segmentId} does not belong to this project.`
        await options.router.replace(routeLocation(projectWorkspaceRoute(project.id, 'overview')))
        return
      }
      if (!segment.sceneId) {
        workspace.catalog.error.value = `${segment.label} is not ready for editing yet.`
        await options.router.replace(routeLocation(projectWorkspaceRoute(project.id, 'analysis')))
        return
      }
      targetSceneId = segment.sceneId
      const asset = workspace.media.assets.value.find((item) => item.id === segment.assetId)
      if (!asset?.timelineSceneId) {
        workspace.catalog.error.value = `${segment.label} has no available source timeline.`
        await options.router.replace(routeLocation(projectWorkspaceRoute(project.id, 'analysis')))
        return
      }
      timelineSceneId = asset.timelineSceneId
    } else {
      targetSceneId = intent.sceneId
      timelineSceneId = targetSceneId
    }

    workspace.catalog.loading.value = true
    try {
      const projectScenes = await options.sceneSession.list()
      if (activeRequest !== requestId) return
      if (!projectScenes.some((item) => item.id === targetSceneId)) {
        throw new Error(`Scene ${targetSceneId} does not belong to this project.`)
      }
      if (!projectScenes.some((item) => item.id === timelineSceneId)) {
        throw new Error(`Timeline Scene ${timelineSceneId} does not belong to this project.`)
      }

      if (intent.name === APP_ROUTE_NAMES.projectScene) {
        const timelineAsset = workspace.media.assets.value.find(
          (asset) => asset.timelineSceneId === targetSceneId,
        )
        if (timelineAsset) {
          await options.router.replace(routeLocation(videoTimelineRoute(project.id, timelineAsset.id, intent.view)))
          return
        }
        const segment = workspace.media.segments.value.find((item) => item.sceneId === targetSceneId)
        if (segment) {
          await options.router.replace(routeLocation(projectSegmentRoute(project.id, segment.id, intent.view)))
          return
        }
      }

      const sceneChanged = options.scene.value?.id !== targetSceneId
      if (sceneChanged) {
        options.playing.value = false
        options.sourceVideo.value?.pause()
        await options.sceneSession.load(targetSceneId)
      }
      if (activeRequest !== requestId || options.scene.value?.id !== targetSceneId) return
      if (timelineSceneId === targetSceneId) {
        options.timelineScene.value = options.scene.value
      } else if (
        needsWorkspaceLoad
        || options.timelineScene.value?.id !== timelineSceneId
      ) {
        const timelineScene = await options.sceneSession.read(timelineSceneId)
        if (activeRequest !== requestId) return
        options.timelineScene.value = timelineScene
      }
      if (sceneChanged) {
        options.selectedTrackId.value = null
        options.selectedCanonicalPersonId.value = null
      }
      const view = intent.view
      if (view?.time !== undefined) {
        await nextTick()
        options.seekTo(view.time)
      }
      if (view?.panel === 'quality') options.activeTab.value = 'qa'
      else if (view?.panel === 'events') options.activeTab.value = 'events'
      else if (view?.panel === 'inspector') options.activeTab.value = 'binding'
      if (view?.subject) {
        const person = options.scene.value.payload.canonicalPeople?.find(
          (candidate) => candidate.canonicalPersonId === view.subject,
        )
        if (person) {
          options.selectedCanonicalPersonId.value = person.canonicalPersonId
          options.selectedTrackId.value = options.scene.value.payload.tracks.find(
            (track) => track.canonicalPersonId === person.canonicalPersonId,
          )?.id ?? null
        }
      }
      options.error.value = null
      workspace.catalog.error.value = null
    } catch (cause) {
      if (activeRequest !== requestId) return
      options.scene.value = null
      options.timelineScene.value = null
      options.error.value = cause instanceof Error
        ? cause.message
        : 'Could not open this project route.'
    } finally {
      if (activeRequest === requestId) workspace.catalog.loading.value = false
    }
  }

  watch(() => options.route.fullPath, () => { void sync() }, { immediate: true })
  onScopeDispose(() => {
    requestId += 1
    options.sceneSession.cancelPendingLoad()
  })

  return { replaceView, navigateScene, sync }
}
