<script lang="ts">
import { onBeforeUnmount, onMounted, type Ref } from 'vue'

type ToolbarDisclosureDismissalOptions = {
  open: Readonly<Ref<boolean>>
  root: Readonly<Ref<HTMLElement | null>>
  closeMenu: (restoreFocus?: boolean) => void
}

export function useToolbarDisclosureDismissal({
  open,
  root,
  closeMenu,
}: ToolbarDisclosureDismissalOptions) {
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
}
</script>

<script setup lang="ts">
import { nextTick, ref, useId, watch } from 'vue'

const props = withDefaults(defineProps<{
  label: string
  active?: boolean
  disabled?: boolean
}>(), {
  active: false,
  disabled: false,
})

defineSlots<{
  icon?: () => unknown
  default: (props: { closeMenu: () => void }) => unknown
}>()

const root = ref<HTMLElement | null>(null)
const trigger = ref<HTMLButtonElement | null>(null)
const open = ref(false)
const panelId = `${useId()}-toolbar-panel`

function closeMenu(restoreFocus = false) {
  if (!open.value) return
  open.value = false
  if (restoreFocus) nextTick(() => trigger.value?.focus())
}

function toggleMenu() {
  if (!props.disabled) open.value = !open.value
}

watch(() => props.disabled, (disabled) => {
  if (disabled) closeMenu()
})

useToolbarDisclosureDismissal({ open, root, closeMenu })
</script>

<template>
  <div ref="root" class="toolbar-disclosure">
    <button
      ref="trigger"
      type="button"
      class="toolbar-disclosure-trigger"
      :class="{ active: active || open }"
      :disabled="disabled"
      :aria-label="label"
      :title="label"
      :aria-expanded="open"
      :aria-controls="panelId"
      aria-haspopup="dialog"
      @click="toggleMenu"
    >
      <slot name="icon" />
      <span>{{ label }}</span>
      <svg viewBox="0 0 12 8" aria-hidden="true"><path d="m1 1 5 5 5-5" /></svg>
    </button>
    <section
      v-if="open"
      :id="panelId"
      class="toolbar-disclosure-panel"
      role="dialog"
      :aria-label="label"
    >
      <slot :close-menu="closeMenu" />
    </section>
  </div>
</template>

<style scoped>
.toolbar-disclosure { position: relative; flex: 0 0 auto; }
.toolbar-disclosure-trigger {
  min-height: 34px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 0 9px;
  border: 1px solid rgba(238, 243, 235, .11);
  border-radius: 3px;
  background: transparent;
  color: #9ca5a0;
  font-size: 11px;
  cursor: pointer;
}
.toolbar-disclosure-trigger:hover,
.toolbar-disclosure-trigger.active {
  border-color: rgba(255, 211, 106, .42);
  background: rgba(255, 211, 106, .055);
  color: #f4f6f1;
}
.toolbar-disclosure-trigger:focus-visible { outline: 2px solid #ffd36a; outline-offset: 2px; }
.toolbar-disclosure-trigger:disabled { opacity: .5; cursor: wait; }
.toolbar-disclosure-trigger > :deep(svg:first-child) {
  width: 16px;
  height: 16px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.6;
}
.toolbar-disclosure-trigger > svg:last-child {
  width: 9px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.5;
  transition: transform .16s ease;
}
.toolbar-disclosure-trigger[aria-expanded="true"] > svg:last-child { transform: rotate(180deg); }
.toolbar-disclosure-panel {
  position: absolute;
  z-index: 24;
  top: calc(100% + 7px);
  right: 0;
  width: min(330px, calc(100vw - 24px));
  border: 1px solid rgba(238, 243, 235, .18);
  border-radius: 5px;
  background: rgba(14, 18, 18, .98);
  box-shadow: 0 18px 48px rgba(0, 0, 0, .54);
  backdrop-filter: blur(16px);
}
@media (prefers-reduced-motion: reduce) {
  .toolbar-disclosure-trigger > svg:last-child { transition: none; }
}
</style>
