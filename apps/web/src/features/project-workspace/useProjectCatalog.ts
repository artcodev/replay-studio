import { ref } from 'vue'
import type { Router, RouteLocationRaw } from 'vue-router'
import { projectClient } from '../../lib/api/projects'
import {
  appRouteLocation,
  projectWorkspaceRoute,
} from '../../lib/appRoutes'
import { resolveExplicitWorkspaceProjectId } from '../../lib/projectSelection'
import type { Project } from '../../types/project'

export type ProjectCatalogLoadOutcome = 'loaded' | 'index' | 'not-found'

/** Project index and selected project header; no match/media/identity ownership. */
export function useProjectCatalog(router: Router) {
  const projects = ref<Project[]>([])
  const project = ref<Project | null>(null)
  const loading = ref(false)
  const mutationBusy = ref(false)
  const error = ref<string | null>(null)

  async function load(
    requestedProjectId: string | null | undefined,
    signal: AbortSignal,
  ): Promise<ProjectCatalogLoadOutcome> {
    const rows = await projectClient.list(signal)
    signal.throwIfAborted()
    const projectId = resolveExplicitWorkspaceProjectId(rows, requestedProjectId)
    if (!projectId) {
      projects.value = rows
      project.value = null
      if (requestedProjectId) {
        error.value = `Project ${requestedProjectId} was not found.`
        return 'not-found'
      }
      return 'index'
    }
    const selected = await projectClient.get(projectId, signal)
    signal.throwIfAborted()
    projects.value = rows
    project.value = selected
    return 'loaded'
  }

  async function refreshActive(signal?: AbortSignal) {
    const active = project.value
    if (!active) return null
    const updated = await projectClient.get(active.id, signal)
    if (project.value?.id === active.id) project.value = updated
    return updated
  }

  async function select(projectId: string) {
    await router.push(appRouteLocation(projectWorkspaceRoute(projectId)) as RouteLocationRaw)
  }

  async function create() {
    if (mutationBusy.value) return
    const title = window.prompt('Project name', 'New football project')?.trim()
    if (!title) return
    mutationBusy.value = true
    error.value = null
    try {
      const created = await projectClient.create(title)
      await select(created.id)
    } catch (cause) {
      error.value = cause instanceof Error ? cause.message : 'Could not create the project'
    } finally {
      mutationBusy.value = false
    }
  }

  function clearSelection() {
    project.value = null
  }

  return {
    projects,
    project,
    loading,
    mutationBusy,
    error,
    load,
    refreshActive,
    select,
    create,
    clearSelection,
  }
}

export type ProjectCatalog = ReturnType<typeof useProjectCatalog>
