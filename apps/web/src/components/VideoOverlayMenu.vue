<script setup lang="ts">
import { computed, toRef, useId } from 'vue'
import { useDisclosureMenu } from '../composables/useDisclosureMenu'
import {
  VIDEO_OVERLAY_LAYER_ITEMS,
  videoOverlayGroups,
  withVideoOverlayOption,
  type VideoOverlayOptionKey,
  type VideoOverlayOptions,
} from '../lib/videoOverlayOptions'

const props = withDefaults(defineProps<{
  modelValue: VideoOverlayOptions
  disabled?: boolean
}>(), {
  disabled: false,
})

const emit = defineEmits<{
  'update:modelValue': [options: VideoOverlayOptions]
}>()

const groups = videoOverlayGroups()
const { root, trigger, panel, open, toggleMenu } = useDisclosureMenu(
  toRef(props, 'disabled'),
)
const panelId = `${useId()}-video-overlay-panel`

const hiddenCount = computed(() => (
  VIDEO_OVERLAY_LAYER_ITEMS.reduce(
    (total, item) => total + (props.modelValue[item.key] ? 0 : 1),
    0,
  )
))

const triggerLabel = computed(() => (
  hiddenCount.value
    ? `Video overlay, ${hiddenCount.value} ${hiddenCount.value === 1 ? 'layer' : 'layers'} hidden`
    : 'Video overlay settings'
))

function updateOption(key: VideoOverlayOptionKey, event: Event) {
  const input = event.target as HTMLInputElement
  emit('update:modelValue', withVideoOverlayOption(props.modelValue, key, input.checked))
}
</script>

<template>
  <div ref="root" class="video-view-menu">
    <button
      ref="trigger"
      class="video-view-trigger"
      type="button"
      :class="{ active: open }"
      :disabled="disabled"
      :aria-label="triggerLabel"
      :title="triggerLabel"
      :aria-expanded="open"
      :aria-controls="panelId"
      aria-haspopup="dialog"
      @click="toggleMenu"
    >
      <svg class="eye-icon" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M2.3 12s3.3-6 9.7-6 9.7 6 9.7 6-3.3 6-9.7 6-9.7-6-9.7-6Z" />
        <circle cx="12" cy="12" r="2.75" />
      </svg>
      <span>View</span>
      <i v-if="hiddenCount" class="hidden-count" aria-hidden="true">{{ hiddenCount }}</i>
      <svg class="chevron" viewBox="0 0 12 8" aria-hidden="true">
        <path d="m1 1 5 5 5-5" />
      </svg>
    </button>

    <section
      v-if="open"
      :id="panelId"
      ref="panel"
      class="video-view-panel"
      role="dialog"
      aria-label="Video overlay settings"
    >
      <fieldset v-for="group in groups" :key="group.group">
        <legend>{{ group.label }}</legend>
        <label v-for="item in group.items" :key="item.key" class="view-option">
          <span class="checkbox-shell">
            <input
              type="checkbox"
              :checked="modelValue[item.key]"
              @change="updateOption(item.key, $event)"
            />
            <i aria-hidden="true">
              <svg viewBox="0 0 12 9"><path d="m1 4 3 3 7-6" /></svg>
            </i>
          </span>
          <span class="option-copy">
            <strong>{{ item.label }}</strong>
            <small>{{ item.detail }}</small>
          </span>
        </label>
      </fieldset>
    </section>
  </div>
</template>

<style scoped>
.video-view-menu {
  position: relative;
  flex: 0 0 auto;
  color: #ecf0ea;
}

.video-view-trigger {
  min-height: 30px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 0 9px;
  border: 1px solid rgba(238, 243, 235, .11);
  border-radius: 3px;
  background: rgba(10, 13, 12, .72);
  color: #9ca5a0;
  font-size: 11px;
  cursor: pointer;
  transition: border-color .16s ease, background .16s ease, color .16s ease;
}

.video-view-trigger:hover,
.video-view-trigger.active {
  border-color: rgba(255, 211, 106, .42);
  background: rgba(255, 211, 106, .1);
  color: #f4f6f1;
}

.video-view-trigger:focus-visible {
  outline: 2px solid #ffd36a;
  outline-offset: 2px;
}

.video-view-trigger:disabled {
  opacity: .5;
  cursor: not-allowed;
}

.eye-icon {
  width: 16px;
  height: 16px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.6;
}

.chevron {
  width: 9px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.7;
}

.hidden-count {
  min-width: 16px;
  height: 16px;
  display: grid;
  place-items: center;
  border-radius: 9px;
  background: rgba(255, 211, 106, .14);
  color: #ffd36a;
  font: 600 9px 'DM Mono', monospace;
  font-style: normal;
}

.video-view-panel {
  position: absolute;
  z-index: 30;
  top: calc(100% + 6px);
  right: 0;
  width: 268px;
  max-height: 60vh;
  overflow-y: auto;
  display: grid;
  gap: 10px;
  padding: 12px;
  border: 1px solid rgba(238, 243, 235, .12);
  border-radius: 6px;
  background: rgba(12, 15, 14, .97);
  box-shadow: 0 18px 40px rgba(0, 0, 0, .5);
}

fieldset {
  margin: 0;
  padding: 0;
  border: 0;
  display: grid;
  gap: 6px;
}

legend {
  padding: 0;
  color: #7f8a84;
  font: 600 9px 'DM Mono', monospace;
  letter-spacing: .09em;
  text-transform: uppercase;
}

.view-option {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  padding: 5px 6px;
  border-radius: 4px;
  cursor: pointer;
}

.view-option:hover {
  background: rgba(255, 255, 255, .04);
}

.checkbox-shell {
  position: relative;
  flex: 0 0 auto;
  width: 15px;
  height: 15px;
  margin-top: 2px;
}

.checkbox-shell input {
  position: absolute;
  inset: 0;
  margin: 0;
  opacity: 0;
  cursor: pointer;
}

.checkbox-shell i {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  border: 1px solid rgba(238, 243, 235, .26);
  border-radius: 3px;
  pointer-events: none;
}

.checkbox-shell svg {
  width: 9px;
  fill: none;
  stroke: #0b0e0d;
  stroke-width: 2;
  opacity: 0;
}

.checkbox-shell input:checked + i {
  border-color: #ffd36a;
  background: #ffd36a;
}

.checkbox-shell input:checked + i svg {
  opacity: 1;
}

.checkbox-shell input:focus-visible + i {
  outline: 2px solid #ffd36a;
  outline-offset: 2px;
}

.option-copy {
  display: grid;
  gap: 1px;
}

.option-copy strong {
  color: #eef3eb;
  font-size: 11.5px;
  font-weight: 600;
}

.option-copy small {
  color: #7f8a84;
  font-size: 10px;
  line-height: 1.32;
}
</style>
