import { ref, type Ref } from 'vue'
import { sceneClient } from '../lib/api/scenes'
import type { SceneDocument, SceneSummary } from '../types/scene'

type SceneSessionOptions = {
  projectId: () => string
  scene: Ref<SceneDocument | null>
  scenes: Ref<SceneSummary[]>
  saveState: Ref<string>
}

/** Owns the persisted scene document lifecycle and fences stale scene loads. */
export function useSceneSession(options: SceneSessionOptions) {
  const saving = ref(false)
  let loadRequestId = 0
  let listRequestId = 0

  async function list(): Promise<SceneSummary[]> {
    const requestId = ++listRequestId
    const projectScenes = await sceneClient.list(options.projectId())
    if (requestId === listRequestId) options.scenes.value = projectScenes
    return projectScenes
  }

  async function load(id: string): Promise<SceneDocument | null> {
    const requestId = ++loadRequestId
    const loaded = await sceneClient.get(options.projectId(), id)
    if (requestId !== loadRequestId) return null
    options.scene.value = loaded
    return loaded
  }

  /** Read a secondary Scene projection without replacing the writable document. */
  async function read(id: string): Promise<SceneDocument> {
    return sceneClient.get(options.projectId(), id)
  }

  async function refresh(id: string): Promise<SceneDocument | null> {
    const loaded = await sceneClient.get(options.projectId(), id)
    if (options.scene.value?.id !== id) return null
    options.scene.value = loaded
    return loaded
  }

  /**
   * Run one dedicated scene command. Commands carry only their own domain,
   * so the server applies them onto the currently stored scene: an editor
   * that has not seen the newest revision still saves, and a reconstruction
   * write can no longer invalidate an unrelated UI edit.
   */
  async function runCommand(
    write: (projectId: string, sceneId: string) => Promise<SceneDocument>,
    successState: string,
    failureState: string,
  ): Promise<void> {
    const current = options.scene.value
    if (!current || saving.value) return
    saving.value = true
    options.saveState.value = 'Saving…'
    try {
      const saved = await write(options.projectId(), current.id)
      if (options.scene.value?.id !== current.id) return
      options.scene.value = saved
      options.saveState.value = successState
    } catch (cause) {
      options.saveState.value = cause instanceof Error ? cause.message : failureState
    } finally {
      saving.value = false
    }
  }

  async function saveTitle(title: string): Promise<void> {
    await runCommand(
      (projectId, sceneId) => sceneClient.saveTitle(projectId, sceneId, title),
      'Title saved',
      'Could not save the title',
    )
  }

  async function saveEventBindings(): Promise<void> {
    const bindings = options.scene.value?.payload.eventBindings ?? []
    await runCommand(
      (projectId, sceneId) => sceneClient.saveEventBindings(projectId, sceneId, bindings),
      'Event markers saved',
      'Could not save the event markers',
    )
  }

  async function saveTrackMetadata(
    trackId: string,
    metadata: { label?: string; number?: number },
  ): Promise<void> {
    await runCommand(
      (projectId, sceneId) => sceneClient.saveTrackMetadata(projectId, sceneId, trackId, metadata),
      'Track updated',
      'Could not update the track',
    )
  }

  async function saveTrackTrajectory(
    trackId: string,
    keyframes: Array<{ t: number; x: number; z: number }>,
  ): Promise<void> {
    await runCommand(
      (projectId, sceneId) => sceneClient.saveTrackTrajectory(projectId, sceneId, trackId, keyframes),
      'Trajectory correction saved',
      'Could not save the trajectory correction',
    )
  }

  async function saveSegmentLayout(): Promise<void> {
    const current = options.scene.value
    if (!current) return
    await runCommand(
      (projectId) => sceneClient.saveSegmentLayout(projectId, current),
      'Timeline layout saved',
      'Could not save the timeline layout',
    )
  }

  function cancelPendingLoad() {
    loadRequestId += 1
    listRequestId += 1
  }

  return {
    saving,
    list,
    load,
    read,
    refresh,
    saveTitle,
    saveEventBindings,
    saveTrackMetadata,
    saveTrackTrajectory,
    saveSegmentLayout,
    cancelPendingLoad,
  }
}
