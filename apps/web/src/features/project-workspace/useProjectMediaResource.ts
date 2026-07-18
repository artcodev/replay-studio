import { ref } from 'vue'
import type { Router, RouteLocationRaw } from 'vue-router'
import { projectClient } from '../../lib/api/projects'
import {
  appRouteLocation,
  projectSegmentRoute,
  projectWorkspaceRoute,
  videoTimelineRoute,
} from '../../lib/appRoutes'
import type { ProjectAsset, ProjectSegment } from '../../types/project'
import type { ProjectCatalog } from './useProjectCatalog'

/** Source videos, moments and navigation into their editor routes. */
export function useProjectMediaResource(router: Router, catalog: ProjectCatalog) {
  const assets = ref<ProjectAsset[]>([])
  const segments = ref<ProjectSegment[]>([])

  async function load(projectId: string, signal: AbortSignal) {
    const [loadedAssets, loadedSegments] = await Promise.all([
      projectClient.assets(projectId, signal),
      projectClient.segments(projectId, signal),
    ])
    signal.throwIfAborted()
    assets.value = loadedAssets
    segments.value = loadedSegments
  }

  async function openTimeline(assetId?: string | null) {
    const active = catalog.project.value
    if (!active) return
    catalog.error.value = null
    const ready = assets.value.filter(
      (asset) => asset.status === 'ready' && Boolean(asset.timelineSceneId),
    )
    const target = assetId
      ? ready.find((asset) => asset.id === assetId)
      : ready.length === 1 ? ready[0] : null
    if (!target) {
      if (ready.length > 1) {
        catalog.error.value = 'This project has several source videos. Choose Open timeline on the video you want to review.'
        await router.push(appRouteLocation(projectWorkspaceRoute(active.id, 'overview')) as RouteLocationRaw)
        return
      }
      catalog.error.value = 'This project has no ready video timeline yet. Import a clip or check its Analysis job.'
      await router.push(appRouteLocation(projectWorkspaceRoute(active.id, 'analysis')) as RouteLocationRaw)
      return
    }
    await router.push(appRouteLocation(videoTimelineRoute(active.id, target.id)) as RouteLocationRaw)
  }

  async function selectSegment(segmentId: string) {
    const segment = segments.value.find((item) => item.id === segmentId)
    if (!segment) return
    if (!segment.sceneId) {
      catalog.error.value = `${segment.label} is not ready for editing yet.`
      await router.push(appRouteLocation(projectWorkspaceRoute(segment.projectId, 'analysis')) as RouteLocationRaw)
      return
    }
    await router.push(appRouteLocation(projectSegmentRoute(segment.projectId, segment.id)) as RouteLocationRaw)
  }

  function clear() {
    assets.value = []
    segments.value = []
  }

  return { assets, segments, load, openTimeline, selectSegment, clear }
}

export type ProjectMediaResource = ReturnType<typeof useProjectMediaResource>
