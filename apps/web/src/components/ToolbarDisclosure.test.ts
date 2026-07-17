import { createRenderer, createSSRApp, defineComponent, h, nextTick, ref } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import ToolbarDisclosure, { useToolbarDisclosureDismissal } from './ToolbarDisclosure.vue'

type HostNode = {
  type: string
  text: string
  parent: HostNode | null
  children: HostNode[]
  props: Record<string, unknown>
  contains: (target: unknown) => boolean
  focus: () => void
}

function hostNode(type: string, text = ''): HostNode {
  const node: HostNode = {
    type,
    text,
    parent: null,
    children: [],
    props: {},
    contains: (target) => target === node || node.children.some((child) => child.contains(target)),
    focus: () => undefined,
  }
  return node
}

function findHostNode(node: HostNode, type: string): HostNode | null {
  if (node.type === type) return node
  for (const child of node.children) {
    const match = findHostNode(child, type)
    if (match) return match
  }
  return null
}

function createHostRenderer() {
  return createRenderer<HostNode, HostNode>({
    patchProp(element, key, _previous, next) {
      element.props[key] = next
    },
    insert(element, parent, anchor) {
      element.parent = parent
      const anchorIndex = anchor ? parent.children.indexOf(anchor) : -1
      if (anchorIndex === -1) parent.children.push(element)
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

function fakeDocument() {
  const listeners = new Map<string, Set<EventListenerOrEventListenerObject>>()
  return {
    addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
      const registered = listeners.get(type) ?? new Set<EventListenerOrEventListenerObject>()
      registered.add(listener)
      listeners.set(type, registered)
    },
    removeEventListener(type: string, listener: EventListenerOrEventListenerObject) {
      listeners.get(type)?.delete(listener)
    },
    dispatch(type: string, event: Event) {
      listeners.get(type)?.forEach((listener) => {
        if (typeof listener === 'function') listener(event)
        else listener.handleEvent(event)
      })
    },
    listenerCount(type: string) {
      return listeners.get(type)?.size ?? 0
    },
  }
}

describe('ToolbarDisclosure', () => {
  it('renders an accessible collapsed disclosure trigger', async () => {
    const html = await renderToString(createSSRApp({
      render: () => h(ToolbarDisclosure, { label: 'Reconstruction' }, {
        default: () => h('span', 'Advanced controls'),
      }),
    }))

    expect(html).toContain('aria-label="Reconstruction"')
    expect(html).toContain('aria-expanded="false"')
    expect(html).toContain('aria-haspopup="dialog"')
    expect(html).not.toContain('Advanced controls')
  })

  it('closes when focus leaves and removes the focus listener on unmount', async () => {
    const documentStub = fakeDocument()
    const originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document')
    Object.defineProperty(globalThis, 'document', {
      configurable: true,
      value: documentStub,
    })

    try {
      const renderer = createHostRenderer()
      const container = hostNode('root')
      const Harness = defineComponent({
        setup() {
          const open = ref(true)
          const disclosureRoot = ref<HTMLElement | null>(null)
          useToolbarDisclosureDismissal({
            open,
            root: disclosureRoot,
            closeMenu: () => { open.value = false },
          })
          return () => h('div', {
            ref: disclosureRoot,
            'data-open': open.value,
          }, open.value ? [h('section')] : [])
        },
      })
      const app = renderer.createApp(Harness)

      app.mount(container)
      expect(documentStub.listenerCount('focusin')).toBe(1)

      const disclosure = findHostNode(container, 'div')
      expect(disclosure?.props['data-open']).toBe(true)
      const panel = findHostNode(container, 'section')
      expect(panel).not.toBeNull()

      documentStub.dispatch('focusin', { target: panel } as unknown as Event)
      await nextTick()
      expect(disclosure?.props['data-open']).toBe(true)

      documentStub.dispatch('focusin', { target: hostNode('outside') } as unknown as Event)
      await nextTick()
      expect(disclosure?.props['data-open']).toBe(false)
      expect(findHostNode(container, 'section')).toBeNull()

      app.unmount()
      expect(documentStub.listenerCount('focusin')).toBe(0)
    } finally {
      if (originalDocument) Object.defineProperty(globalThis, 'document', originalDocument)
      else Reflect.deleteProperty(globalThis, 'document')
    }
  })
})
