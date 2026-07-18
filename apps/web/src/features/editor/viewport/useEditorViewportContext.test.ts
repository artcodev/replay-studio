import { createRenderer, defineComponent, h, nextTick } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import { useSceneDocumentState, type SceneDocumentState } from '../../../composables/useSceneDocumentState'
import type { SceneDocument } from '../../../types/scene'
import { useEditorViewportContext, type EditorViewportContext } from './useEditorViewportContext'

type HostNode = {
  type: string
  parent: HostNode | null
  children: HostNode[]
  text: string
}

function hostNode(type: string, text = ''): HostNode {
  return { type, parent: null, children: [], text }
}

function hostRenderer() {
  return createRenderer<HostNode, HostNode>({
    patchProp: () => undefined,
    insert(element, parent, anchor) {
      element.parent = parent
      const anchorIndex = anchor ? parent.children.indexOf(anchor) : -1
      if (anchorIndex < 0) parent.children.push(element)
      else parent.children.splice(anchorIndex, 0, element)
    },
    remove(element) {
      if (!element.parent) return
      const index = element.parent.children.indexOf(element)
      if (index >= 0) element.parent.children.splice(index, 1)
      element.parent = null
    },
    createElement: (type) => hostNode(type),
    createText: (text) => hostNode('#text', text),
    createComment: (text) => hostNode('#comment', text),
    setText(node, text) {
      node.text = text
    },
    setElementText(node, text) {
      node.text = text
      node.children = []
    },
    parentNode: (node) => node.parent,
    nextSibling(node) {
      if (!node.parent) return null
      const index = node.parent.children.indexOf(node)
      return node.parent.children[index + 1] ?? null
    },
  })
}

function scene(id: string, sourceStart?: number): SceneDocument {
  return {
    id,
    title: id,
    version: 1,
    revision: 1,
    duration: 8,
    payload: {
      pitch: { length: 105, width: 68 },
      ...(sourceStart === undefined
        ? {}
        : {
            videoAsset: {
              id: 'shared-video',
              filename: 'shared.mp4',
              mediaUrl: '/shared.mp4',
              posterUrl: '/shared.jpg',
              fps: 25,
              frameCount: 200,
              processingState: 'ready',
              sourceStart,
              sourceEnd: sourceStart + 8,
            },
          }),
      teams: [],
      tracks: [],
      ball: { keyframes: [] },
      eventBindings: [],
      cameraCuts: [],
    },
  }
}

describe('editor viewport scene transitions', () => {
  it('resets layout and seeks a reused video element to the new source range', async () => {
    const originalWindow = Object.getOwnPropertyDescriptor(globalThis, 'window')
    const originalRequestAnimationFrame = Object.getOwnPropertyDescriptor(
      globalThis,
      'requestAnimationFrame',
    )
    const originalCancelAnimationFrame = Object.getOwnPropertyDescriptor(
      globalThis,
      'cancelAnimationFrame',
    )
    const storage = new Map<string, string>()
    Object.defineProperty(globalThis, 'window', {
      configurable: true,
      value: {
        localStorage: {
          getItem: (key: string) => storage.get(key) ?? null,
          setItem: (key: string, value: string) => { storage.set(key, value) },
        },
      },
    })
    Object.defineProperty(globalThis, 'requestAnimationFrame', {
      configurable: true,
      value: vi.fn(() => 1),
    })
    Object.defineProperty(globalThis, 'cancelAnimationFrame', {
      configurable: true,
      value: vi.fn(),
    })

    let documentState!: SceneDocumentState
    let viewport!: EditorViewportContext
    const pause = vi.fn()
    const video = {
      currentTime: 5,
      playbackRate: 1,
      ended: false,
      pause,
      play: vi.fn(async () => undefined),
    } as unknown as HTMLVideoElement
    const renderer = hostRenderer()
    const app = renderer.createApp(defineComponent({
      setup() {
        documentState = useSceneDocumentState(scene('segment-a', 5))
        viewport = useEditorViewportContext(documentState)
        return () => h('div')
      },
    }))

    try {
      app.mount(hostNode('root'))
      viewport.sourceVideo.value = video
      viewport.currentTime.value = 0
      viewport.viewMode.value = 'video'

      // Both segments reuse the same media URL. Since the local time remains
      // zero, only an explicit seek can move the element to the new sourceStart.
      documentState.scene.value = scene('segment-b', 37)
      await nextTick()

      expect(viewport.viewMode.value).toBe('split')
      expect(viewport.currentTime.value).toBe(0)
      expect(video.currentTime).toBe(37)
      expect(pause).toHaveBeenCalled()

      documentState.scene.value = scene('synthetic-3d')
      await nextTick()
      expect(viewport.viewMode.value).toBe('3d')
    } finally {
      app.unmount()
      if (originalWindow) Object.defineProperty(globalThis, 'window', originalWindow)
      else Reflect.deleteProperty(globalThis, 'window')
      if (originalRequestAnimationFrame) {
        Object.defineProperty(
          globalThis,
          'requestAnimationFrame',
          originalRequestAnimationFrame,
        )
      } else Reflect.deleteProperty(globalThis, 'requestAnimationFrame')
      if (originalCancelAnimationFrame) {
        Object.defineProperty(
          globalThis,
          'cancelAnimationFrame',
          originalCancelAnimationFrame,
        )
      } else Reflect.deleteProperty(globalThis, 'cancelAnimationFrame')
    }
  })
})
