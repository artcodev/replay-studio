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

  async function save(): Promise<void> {
    const current = options.scene.value
    if (!current || saving.value) return
    saving.value = true
    options.saveState.value = 'Saving…'
    try {
      const saved = await sceneClient.save(options.projectId(), current)
      if (options.scene.value?.id !== current.id) return
      options.scene.value = saved
      options.saveState.value = 'All changes saved'
    } catch (cause) {
      options.saveState.value = cause instanceof Error ? cause.message : 'Save failed'
    } finally {
      saving.value = false
    }
  }

  function cancelPendingLoad() {
    loadRequestId += 1
    listRequestId += 1
  }

  return { saving, list, load, read, refresh, save, cancelPendingLoad }
}
