import { shallowRef, triggerRef } from 'vue'
import type { SceneDocument } from '../types/scene'

/** Large scene documents are replaced atomically; explicit edits trigger one shallow update. */
export function useSceneDocumentState(initial: SceneDocument | null = null) {
  const scene = shallowRef<SceneDocument | null>(initial)

  function notifyMutation() {
    triggerRef(scene)
  }

  function mutate(mutator: (document: SceneDocument) => void): boolean {
    const current = scene.value
    if (!current) return false
    mutator(current)
    notifyMutation()
    return true
  }

  return { scene, mutate, notifyMutation }
}

export type SceneDocumentState = ReturnType<typeof useSceneDocumentState>
