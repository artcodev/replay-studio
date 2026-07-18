import { inject, provide, type InjectionKey } from 'vue'
import type { Router } from 'vue-router'
import { useAnalysisJobs } from '../lib/analysisJobs'
import { useProjectCatalog } from '../features/project-workspace/useProjectCatalog'
import { useProjectIdentityResource } from '../features/project-workspace/useProjectIdentityResource'
import { useProjectMatchResource } from '../features/project-workspace/useProjectMatchResource'
import { useProjectMediaResource } from '../features/project-workspace/useProjectMediaResource'

export type ProjectWorkspaceLoadOutcome = 'loaded' | 'index' | 'not-found' | 'failed' | 'stale'

function aborted(cause: unknown) {
  return Boolean(cause && typeof cause === 'object' && 'name' in cause && cause.name === 'AbortError')
}

/**
 * Thin route-level composition of independently owned project resources.
 * Consumers use the named resource that owns each workflow.
 */
export function useProjectWorkspace(router: Router) {
  const catalog = useProjectCatalog(router)
  const jobs = useAnalysisJobs()
  const match = useProjectMatchResource(catalog, jobs.refresh)
  const media = useProjectMediaResource(router, catalog)
  const identities = useProjectIdentityResource(catalog)
  let loadRequestId = 0
  let loadController: AbortController | null = null

  function clearResources() {
    catalog.clearSelection()
    match.clear()
    media.clear()
    identities.clear()
    jobs.stop()
  }

  async function load(requestedProjectId?: string | null): Promise<ProjectWorkspaceLoadOutcome> {
    const requestId = ++loadRequestId
    loadController?.abort()
    if (!requestedProjectId || catalog.project.value?.id !== requestedProjectId) clearResources()
    const controller = new AbortController()
    loadController = controller
    catalog.loading.value = true
    catalog.error.value = null
    try {
      const outcome = await catalog.load(requestedProjectId, controller.signal)
      if (requestId !== loadRequestId) return 'stale'
      if (outcome !== 'loaded') {
        match.clear()
        media.clear()
        identities.clear()
        jobs.stop()
        return outcome
      }
      const project = catalog.project.value
      if (!project) return 'failed'
      await Promise.all([
        match.load(project.id, controller.signal),
        media.load(project.id, controller.signal),
        identities.load(project.id, controller.signal),
      ])
      if (requestId !== loadRequestId) return 'stale'
      void jobs.start(project.id)
      return 'loaded'
    } catch (cause) {
      if (requestId !== loadRequestId || controller.signal.aborted || aborted(cause)) return 'stale'
      catalog.error.value = cause instanceof Error ? cause.message : 'Could not open projects'
      return 'failed'
    } finally {
      if (requestId === loadRequestId) {
        catalog.loading.value = false
        if (loadController === controller) loadController = null
      }
    }
  }

  function dispose() {
    loadRequestId += 1
    loadController?.abort()
    match.clear()
    identities.clear()
    jobs.stop()
  }

  return { catalog, match, media, identities, jobs, load, dispose }
}

export type ProjectWorkspace = ReturnType<typeof useProjectWorkspace>

const PROJECT_WORKSPACE_KEY: InjectionKey<ProjectWorkspace> = Symbol('ProjectWorkspace')

export function provideProjectWorkspace(workspace: ProjectWorkspace) {
  provide(PROJECT_WORKSPACE_KEY, workspace)
}

export function injectProjectWorkspace() {
  const workspace = inject(PROJECT_WORKSPACE_KEY)
  if (!workspace) throw new Error('Project workspace context is not available')
  return workspace
}
