import { ref } from 'vue'
import { projectClient } from '../../lib/api/projects'
import type {
  ProjectIdentity,
  ProjectIdentityMembershipAssignment,
} from '../../types/project'
import type { ProjectCatalog } from './useProjectCatalog'

function aborted(cause: unknown) {
  return Boolean(cause && typeof cause === 'object' && 'name' in cause && cause.name === 'AbortError')
}

/** Cross-scene project identities and membership reassignment. */
export function useProjectIdentityResource(catalog: ProjectCatalog) {
  const rows = ref<ProjectIdentity[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const assigningMembershipId = ref<string | null>(null)
  const assignmentError = ref<string | null>(null)
  let requestId = 0
  let controller: AbortController | null = null

  async function load(projectId: string, signal: AbortSignal) {
    const identities = await projectClient.identities(projectId, signal)
    signal.throwIfAborted()
    rows.value = identities
    error.value = null
    assignmentError.value = null
  }

  async function refresh() {
    const active = catalog.project.value
    if (!active || loading.value) return
    const activeRequest = ++requestId
    controller?.abort()
    const nextController = new AbortController()
    controller = nextController
    loading.value = true
    error.value = null
    assignmentError.value = null
    try {
      const identities = await projectClient.identities(active.id, nextController.signal)
      if (activeRequest !== requestId || catalog.project.value?.id !== active.id) return
      rows.value = identities
    } catch (cause) {
      if (activeRequest !== requestId || nextController.signal.aborted || aborted(cause)) return
      error.value = cause instanceof Error ? cause.message : 'Could not load project identities'
    } finally {
      if (activeRequest === requestId) {
        loading.value = false
        if (controller === nextController) controller = null
      }
    }
  }

  async function assign(assignment: ProjectIdentityMembershipAssignment) {
    const active = catalog.project.value
    if (!active || assigningMembershipId.value) return
    const membership = rows.value
      .flatMap((identity) => identity.memberships)
      .find((candidate) => candidate.id === assignment.membershipId)
    const targetExists = rows.value.some(
      (identity) => identity.id === assignment.projectPersonId,
    )
    if (
      !membership
      || membership.projectPersonId !== assignment.currentProjectPersonId
      || membership.projectPersonId === assignment.projectPersonId
      || !targetExists
    ) return
    assigningMembershipId.value = assignment.membershipId
    assignmentError.value = null
    try {
      await projectClient.assignIdentityMembership(
        active.id,
        assignment.projectPersonId,
        assignment.sceneId,
        assignment.scenePersonId,
      )
      if (catalog.project.value?.id === active.id) await refresh()
    } catch (cause) {
      if (catalog.project.value?.id === active.id) {
        assignmentError.value = cause instanceof Error
          ? cause.message
          : 'Could not reassign this scene identity'
      }
    } finally {
      if (assigningMembershipId.value === assignment.membershipId) {
        assigningMembershipId.value = null
      }
    }
  }

  function clear() {
    rows.value = []
    error.value = null
    assignmentError.value = null
    assigningMembershipId.value = null
    requestId += 1
    controller?.abort()
    controller = null
    loading.value = false
  }

  return {
    rows,
    loading,
    error,
    assigningMembershipId,
    assignmentError,
    load,
    refresh,
    assign,
    clear,
  }
}

export type ProjectIdentityResource = ReturnType<typeof useProjectIdentityResource>
