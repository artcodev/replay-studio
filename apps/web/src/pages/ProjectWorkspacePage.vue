<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter, type RouteLocationRaw } from 'vue-router'
import type { ProjectDashboardTab } from '../components/ProjectDashboard.vue'
import ProjectWorkspaceSurface from '../components/workspace/ProjectWorkspaceSurface.vue'
import StudioTopbar from '../components/shell/StudioTopbar.vue'
import VideoIngestDrawer from '../components/VideoIngestDrawer.vue'
import { injectProjectWorkspace } from '../composables/useProjectWorkspace'
import {
  APP_ROUTE_NAMES,
  appRouteLocation,
  projectWorkspaceRoute,
  projectsRoute,
  type AppRouteIntent,
} from '../lib/appRoutes'
import type { VideoAsset } from '../types/media'

const route = useRoute()
const router = useRouter()
const workspace = injectProjectWorkspace()
const videoIngestOpen = ref(false)
let loadRequestId = 0

const projectId = computed(() => (
  route.name === APP_ROUTE_NAMES.projectWorkspace
    ? String(route.params.projectId)
    : null
))
const dashboardTab = computed<ProjectDashboardTab>({
  get: () => route.name === APP_ROUTE_NAMES.projectWorkspace
    ? route.params.tab as ProjectDashboardTab
    : 'overview',
  set: (tab) => {
    const id = projectId.value ?? workspace.catalog.project.value?.id
    if (id) void router.push(routeLocation(projectWorkspaceRoute(id, tab)))
  },
})

function routeLocation(intent: AppRouteIntent): RouteLocationRaw {
  return appRouteLocation(intent) as RouteLocationRaw
}

async function loadRoute() {
  const requestId = ++loadRequestId
  const requestedProjectId = projectId.value
  const outcome = await workspace.load(requestedProjectId)
  if (requestId !== loadRequestId) return
  if (requestedProjectId && outcome === 'not-found') {
    await router.replace(routeLocation(projectsRoute()))
  }
}

async function openProcessedVideo(asset: VideoAsset) {
  videoIngestOpen.value = false
  const id = workspace.catalog.project.value?.id
  if (!id) return
  await workspace.load(id)
  await workspace.media.openTimeline(asset.id)
}

watch(projectId, () => { void loadRoute() }, { immediate: true })
</script>

<template>
  <StudioTopbar
    surface="projects"
    :scene-title="null"
    :project="workspace.catalog.project.value"
    :project-count="workspace.catalog.projects.value.length"
    :asset-count="workspace.media.assets.value.length"
    :segment-count="workspace.media.segments.value.length"
    save-state=""
    :project-loading="workspace.catalog.loading.value"
    :save-disabled="true"
    :saving="false"
    @open-import="videoIngestOpen = true"
  />
  <ProjectWorkspaceSurface v-model:active-tab="dashboardTab" @retry-route="loadRoute" />
  <VideoIngestDrawer
    :open="videoIngestOpen"
    :project-id="workspace.catalog.project.value?.id"
    :project-title="workspace.catalog.project.value?.title"
    @close="videoIngestOpen = false"
    @ready="openProcessedVideo"
  />
</template>
