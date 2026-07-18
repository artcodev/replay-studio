import { ref } from 'vue'
import { matchClient } from '../../lib/api/matches'
import type { CanonicalMatch, MatchCandidate } from '../../types/match'
import type { ProjectCatalog } from './useProjectCatalog'

function aborted(cause: unknown) {
  return Boolean(cause && typeof cause === 'object' && 'name' in cause && cause.name === 'AbortError')
}

/** Match snapshot, provider search and selection for the active project. */
export function useProjectMatchResource(
  catalog: ProjectCatalog,
  refreshJobs: () => Promise<void>,
) {
  const snapshot = ref<CanonicalMatch | null>(null)
  const busy = ref(false)
  const searchQuery = ref('Spain vs Belgium')
  const searchDate = ref(new Date().toISOString().slice(0, 10))
  const candidates = ref<MatchCandidate[]>([])
  const searchLoading = ref(false)
  const searchError = ref<string | null>(null)
  const selectingId = ref<string | null>(null)
  let searchRequestId = 0
  let searchController: AbortController | null = null

  async function load(projectId: string, signal: AbortSignal) {
    const loaded = await matchClient.get(projectId, signal)
    signal.throwIfAborted()
    snapshot.value = loaded
    candidates.value = []
    searchError.value = null
  }

  async function search(mode: 'query' | 'date') {
    const active = catalog.project.value
    if (!active || searchLoading.value) return
    if (mode === 'query' && searchQuery.value.trim().length < 3) return
    const requestId = ++searchRequestId
    searchController?.abort()
    const controller = new AbortController()
    searchController = controller
    searchLoading.value = true
    searchError.value = null
    try {
      const rows = await matchClient.search(active.id, {
        ...(mode === 'query' ? { query: searchQuery.value } : { date: searchDate.value }),
        signal: controller.signal,
      })
      if (requestId === searchRequestId) candidates.value = rows
    } catch (cause) {
      if (requestId !== searchRequestId || controller.signal.aborted || aborted(cause)) return
      candidates.value = []
      searchError.value = cause instanceof Error ? cause.message : 'Could not search matches'
    } finally {
      if (requestId === searchRequestId) {
        searchLoading.value = false
        if (searchController === controller) searchController = null
      }
    }
  }

  async function select(candidate: MatchCandidate) {
    const active = catalog.project.value
    if (!active || selectingId.value) return
    selectingId.value = candidate.id
    searchError.value = null
    try {
      const selected = await matchClient.select(active.id, candidate.id)
      if (catalog.project.value?.id !== active.id) return
      await catalog.refreshActive()
      if (catalog.project.value?.id !== active.id) return
      snapshot.value = selected
      candidates.value = []
      void refreshJobs()
    } catch (cause) {
      searchError.value = cause instanceof Error ? cause.message : 'Could not select this match'
    } finally {
      selectingId.value = null
    }
  }

  async function refresh() {
    const active = catalog.project.value
    if (!active || busy.value) return
    busy.value = true
    catalog.error.value = null
    try {
      const refreshed = await matchClient.refresh(active.id)
      if (catalog.project.value?.id !== active.id) return
      await catalog.refreshActive()
      if (catalog.project.value?.id !== active.id) return
      snapshot.value = refreshed
      void refreshJobs()
    } catch (cause) {
      catalog.error.value = cause instanceof Error ? cause.message : 'Could not refresh match data'
    } finally {
      busy.value = false
    }
  }

  function clear() {
    snapshot.value = null
    busy.value = false
    candidates.value = []
    searchError.value = null
    selectingId.value = null
    searchRequestId += 1
    searchController?.abort()
    searchController = null
    searchLoading.value = false
  }

  return {
    snapshot,
    busy,
    searchQuery,
    searchDate,
    candidates,
    searchLoading,
    searchError,
    selectingId,
    load,
    search,
    select,
    refresh,
    clear,
  }
}

export type ProjectMatchResource = ReturnType<typeof useProjectMatchResource>
