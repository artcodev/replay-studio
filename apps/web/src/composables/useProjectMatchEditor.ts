import { ref, type Ref } from 'vue'
import { sceneClient } from '../lib/api/scenes'
import { matchClient } from '../lib/api/matches'
import { reconstructionClient } from '../lib/api/reconstruction'
import { parseManualMatchImport } from '../lib/matchImport'
import type { BallDetectionBackend, ReconstructionModel } from '../types/reconstruction'
import type { SceneDocument } from '../types/scene'
import type { CanonicalMatch } from '../types/match'

type ProjectMatchEditorOptions = {
  projectId: () => string
  scene: Ref<SceneDocument | null>
  match: Ref<CanonicalMatch | null>
  mutationLocked: () => boolean
  selectedModel: Ref<ReconstructionModel>
  selectedBallBackend: Ref<BallDetectionBackend>
  reconstructing: Ref<boolean>
  saveState: Ref<string>
  error: Ref<string | null>
  invalidateIdentityReview: () => void
  clearFrameAnalysis: () => void
  loadIdentityReview: (sceneId: string) => Promise<void>
  startReconstructionPolling: (sceneId: string) => Promise<void>
  refreshWorkspace: () => unknown | Promise<unknown>
}

/** Project match snapshots and manual roster imports share one atomic workflow. */
export function useProjectMatchEditor(options: ProjectMatchEditorOptions) {
  const refreshing = ref(false)
  const importing = ref(false)
  const importError = ref<string | null>(null)

  async function applyToOpenScene(sceneId: string, label: string) {
    if (options.scene.value?.id !== sceneId) return
    const video = options.scene.value.payload.videoAsset
    const singlePass = Boolean(video?.selectedSegmentId && !video.multiPass)
    options.invalidateIdentityReview()
    options.clearFrameAnalysis()
    if (!singlePass) {
      options.scene.value = await sceneClient.get(options.projectId(), sceneId)
      options.saveState.value = video?.multiPass
        ? `${label} · rerun multi-angle analysis when ready`
        : label
      void options.loadIdentityReview(sceneId)
      return
    }
    options.scene.value = await reconstructionClient.reconstruct(
      options.projectId(),
      sceneId,
      options.selectedModel.value,
      options.selectedBallBackend.value,
    )
    options.reconstructing.value = true
    options.saveState.value = `${label} · rebuilding identity…`
    void options.startReconstructionPolling(sceneId)
  }

  async function refresh() {
    const current = options.scene.value
    if (
      !current
      || options.match.value?.sync.state === 'manual'
      || refreshing.value
      || importing.value
      || options.mutationLocked()
    ) return
    const sceneId = current.id
    refreshing.value = true
    options.error.value = null
    options.saveState.value = 'Refreshing project match snapshot…'
    try {
      options.match.value = await matchClient.refresh(options.projectId())
      if (options.scene.value?.id !== sceneId) return
      await applyToOpenScene(sceneId, 'Project match snapshot refreshed')
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not refresh match data'
    } finally {
      refreshing.value = false
    }
  }

  async function importFile(event: Event) {
    const input = event.target as HTMLInputElement
    const file = input.files?.[0]
    if (!file || !options.scene.value) return
    if (options.mutationLocked() || importing.value || refreshing.value) {
      importError.value = 'Wait for reconstruction or project match refresh to finish before importing a roster.'
      input.value = ''
      return
    }
    const sceneId = options.scene.value.id
    importError.value = null
    if (!file.name.toLowerCase().endsWith('.json')) {
      importError.value = 'Choose a .json roster file.'
      input.value = ''
      return
    }
    if (file.size > 2 * 1024 * 1024) {
      importError.value = 'The roster JSON is larger than the 2 MB import limit.'
      input.value = ''
      return
    }
    importing.value = true
    options.saveState.value = `Importing ${file.name}…`
    try {
      const payload = parseManualMatchImport(await file.text())
      if (options.scene.value?.id !== sceneId) return
      options.match.value = await matchClient.import(options.projectId(), payload)
      if (options.scene.value?.id !== sceneId) return
      await applyToOpenScene(sceneId, 'Project roster imported')
      void options.refreshWorkspace()
    } catch (cause) {
      importError.value = cause instanceof Error
        ? cause.message
        : 'Could not import the roster JSON.'
      options.saveState.value = 'Manual roster was not imported'
    } finally {
      importing.value = false
      input.value = ''
    }
  }

  return {
    refreshing,
    importing,
    importError,
    refresh,
    importFile,
  }
}
