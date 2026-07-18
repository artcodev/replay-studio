import { computed } from 'vue'
import { describe, expect, it } from 'vitest'
import type { SceneDocument } from '../types/scene'
import { useSceneDocumentState } from './useSceneDocumentState'

function document(): SceneDocument {
  return {
    id: 'scene-1',
    title: 'Before',
    version: 1,
    revision: 1,
    duration: 1,
    payload: {
      pitch: { length: 105, width: 68 },
      teams: [],
      tracks: [],
      ball: { keyframes: [] },
      eventBindings: [],
      cameraCuts: [],
    },
  }
}

describe('shallow scene document state', () => {
  it('publishes explicit nested edits while preserving the document used for saving', () => {
    const state = useSceneDocumentState(document())
    const title = computed(() => state.scene.value?.title)
    expect(title.value).toBe('Before')

    const original = state.scene.value
    state.mutate((scene) => { scene.title = 'After' })

    expect(title.value).toBe('After')
    expect(state.scene.value).toBe(original)
    expect(state.scene.value?.title).toBe('After')
  })

  it('reacts to immutable server replacements', () => {
    const state = useSceneDocumentState(document())
    const revision = computed(() => state.scene.value?.revision)
    expect(revision.value).toBe(1)
    state.scene.value = { ...document(), revision: 2 }
    expect(revision.value).toBe(2)
  })
})
