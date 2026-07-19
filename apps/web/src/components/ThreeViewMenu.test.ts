import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import ThreeViewMenu from './ThreeViewMenu.vue'
import { DEFAULT_THREE_VIEW_OPTIONS } from '../lib/threeViewOptions'

describe('ThreeViewMenu', () => {
  it('renders an accessible collapsed trigger and reports hidden layers', async () => {
    const html = await renderToString(createSSRApp(ThreeViewMenu, {
      modelValue: { ...DEFAULT_THREE_VIEW_OPTIONS, labels: false, trajectory: false },
      renderQuality: 'basic',
    }))

    expect(html).toContain('aria-label="View settings, 4 items hidden"')
    expect(html).toContain('aria-expanded="false"')
    expect(html).toContain('aria-haspopup="dialog"')
    expect(html).toContain('>View<')
    expect(html).toContain('hidden-count')
  })

  it('keeps path tracking available as a distinct opt-in layer', async () => {
    const app = createSSRApp(ThreeViewMenu, {
      modelValue: DEFAULT_THREE_VIEW_OPTIONS,
      renderQuality: 'basic',
    })
    const html = await renderToString(app)

    // The panel is intentionally collapsed in SSR; its accessible summary
    // still reports the default-off path layer as hidden.
    expect(html).toContain('aria-label="View settings, 2 items hidden"')
  })

  it('disables the settings trigger while its parent is busy', async () => {
    const html = await renderToString(createSSRApp(ThreeViewMenu, {
      modelValue: DEFAULT_THREE_VIEW_OPTIONS,
      renderQuality: 'enhanced',
      disabled: true,
    }))

    expect(html).toContain('disabled')
    expect(html).toContain('aria-label="View settings, 2 items hidden"')
  })
})
