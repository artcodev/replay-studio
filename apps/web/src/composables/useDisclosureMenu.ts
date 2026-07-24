import { nextTick, onBeforeUnmount, onMounted, ref, watch, type Ref } from 'vue'

/**
 * Shared behaviour for the viewport "View" popovers: outside-pointer and
 * focus dismissal, Escape with focus restoration, and focusing the first
 * control on open. Each menu keeps its own markup; only this interaction
 * contract is shared.
 */
export function useDisclosureMenu(disabled: Ref<boolean>) {
  const root = ref<HTMLElement | null>(null)
  const trigger = ref<HTMLButtonElement | null>(null)
  const panel = ref<HTMLElement | null>(null)
  const open = ref(false)

  function closeMenu(restoreFocus = false) {
    if (!open.value) return
    open.value = false
    if (restoreFocus) nextTick(() => trigger.value?.focus())
  }

  function toggleMenu() {
    if (disabled.value) return
    open.value = !open.value
    if (open.value) {
      nextTick(() => panel.value?.querySelector<HTMLInputElement>('input')?.focus())
    }
  }

  function onDocumentPointerDown(event: PointerEvent) {
    if (open.value && !root.value?.contains(event.target as Node)) closeMenu()
  }

  function onDocumentKeyDown(event: KeyboardEvent) {
    if (open.value && event.key === 'Escape') {
      event.preventDefault()
      closeMenu(true)
    }
  }

  function onDocumentFocusIn(event: FocusEvent) {
    if (open.value && !root.value?.contains(event.target as Node)) closeMenu()
  }

  watch(disabled, (value) => {
    if (value) closeMenu()
  })

  onMounted(() => {
    document.addEventListener('pointerdown', onDocumentPointerDown)
    document.addEventListener('keydown', onDocumentKeyDown)
    document.addEventListener('focusin', onDocumentFocusIn)
  })

  onBeforeUnmount(() => {
    document.removeEventListener('pointerdown', onDocumentPointerDown)
    document.removeEventListener('keydown', onDocumentKeyDown)
    document.removeEventListener('focusin', onDocumentFocusIn)
  })

  return { root, trigger, panel, open, closeMenu, toggleMenu }
}
