<script setup lang="ts">
import { computed, toRef, useId } from 'vue'
import { useDisclosureMenu } from '../composables/useDisclosureMenu'
import {
  INFERRED_POSITION_RENDER_MODE_ITEMS,
  THREE_VIEW_LAYER_ITEMS,
  withInferredPositionRenderMode,
  withThreeViewOption,
  type InferredPositionRenderMode,
  type ThreeRenderQuality,
  type ThreeViewOptionKey,
  type ThreeViewOptions,
} from '../lib/threeViewOptions'

const props = withDefaults(defineProps<{
  modelValue: ThreeViewOptions
  renderQuality: ThreeRenderQuality
  disabled?: boolean
}>(), {
  disabled: false,
})

const emit = defineEmits<{
  'update:modelValue': [options: ThreeViewOptions]
  'update:renderQuality': [quality: ThreeRenderQuality]
}>()

const displayItems = THREE_VIEW_LAYER_ITEMS
const inferredPositionItems = INFERRED_POSITION_RENDER_MODE_ITEMS

const qualityItems: ReadonlyArray<{
  value: ThreeRenderQuality
  label: string
  detail: string
}> = [
  { value: 'basic', label: 'Basic', detail: 'Lower GPU load, no shadows' },
  { value: 'enhanced', label: 'Enhanced', detail: 'Higher detail, lighting and shadows' },
]

const { root, trigger, panel, open, toggleMenu } = useDisclosureMenu(
  toRef(props, 'disabled'),
)
const componentId = useId()
const panelId = `${componentId}-three-view-panel`
const qualityGroupId = `${componentId}-render-quality`
const inferredPositionGroupId = `${componentId}-inferred-positions`

const hiddenCount = computed(() => (
  displayItems.reduce((total, item) => total + (props.modelValue[item.key] ? 0 : 1), 0)
))

const triggerLabel = computed(() => (
  hiddenCount.value
    ? `View settings, ${hiddenCount.value} ${hiddenCount.value === 1 ? 'item' : 'items'} hidden`
    : 'View settings'
))

function updateOption(key: ThreeViewOptionKey, event: Event) {
  const input = event.target as HTMLInputElement
  emit('update:modelValue', withThreeViewOption(props.modelValue, key, input.checked))
}

function updateQuality(quality: ThreeRenderQuality, event: Event) {
  if ((event.target as HTMLInputElement).checked) emit('update:renderQuality', quality)
}

function updateInferredPositionMode(mode: InferredPositionRenderMode, event: Event) {
  if ((event.target as HTMLInputElement).checked) {
    emit('update:modelValue', withInferredPositionRenderMode(props.modelValue, mode))
  }
}

</script>

<template>
  <div ref="root" class="three-view-menu">
    <button
      ref="trigger"
      class="three-view-trigger"
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
      class="three-view-panel"
      role="dialog"
      aria-label="View settings"
    >
      <fieldset>
        <legend>Display layers</legend>
        <label v-for="item in displayItems" :key="item.key" class="view-option">
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

      <fieldset>
        <legend>Inferred positions</legend>
        <div class="quality-options inferred-position-options">
          <label v-for="item in inferredPositionItems" :key="item.value">
            <input
              type="radio"
              :name="inferredPositionGroupId"
              :value="item.value"
              :checked="modelValue.inferredPositions === item.value"
              @change="updateInferredPositionMode(item.value, $event)"
            />
            <span>
              <strong>{{ item.label }}</strong>
              <small>{{ item.detail }}</small>
            </span>
          </label>
        </div>
      </fieldset>

      <fieldset :aria-describedby="`${qualityGroupId}-hint`">
        <legend>Render quality</legend>
        <div class="quality-options">
          <label v-for="item in qualityItems" :key="item.value">
            <input
              type="radio"
              :name="qualityGroupId"
              :value="item.value"
              :checked="renderQuality === item.value"
              @change="updateQuality(item.value, $event)"
            />
            <span>
              <strong>{{ item.label }}</strong>
              <small>{{ item.detail }}</small>
            </span>
          </label>
        </div>
        <p :id="`${qualityGroupId}-hint`">Enhanced mode uses more GPU resources.</p>
      </fieldset>
    </section>
  </div>
</template>

<style scoped>
.three-view-menu {
  position: relative;
  flex: 0 0 auto;
  color: #ecf0ea;
}

.three-view-trigger {
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
  transition: border-color .16s ease, background .16s ease, color .16s ease;
}

.three-view-trigger:hover,
.three-view-trigger.active {
  border-color: rgba(255, 211, 106, .42);
  background: rgba(255, 211, 106, .055);
  color: #f4f6f1;
}

.three-view-trigger:focus-visible {
  outline: 2px solid #ffd36a;
  outline-offset: 2px;
}

.three-view-trigger:disabled {
  opacity: .5;
  cursor: wait;
}

.eye-icon {
  width: 17px;
  height: 17px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.6;
}

.hidden-count {
  min-width: 17px;
  height: 17px;
  display: grid;
  place-items: center;
  border-radius: 9px;
  background: rgba(255, 211, 106, .14);
  color: #ffd36a;
  font: 600 9px 'DM Mono', monospace;
  font-style: normal;
}

.chevron {
  width: 9px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.5;
  transition: transform .16s ease;
}

.active .chevron { transform: rotate(180deg); }

.three-view-panel {
  position: absolute;
  z-index: 20;
  top: calc(100% + 7px);
  right: 0;
  width: min(310px, calc(100vw - 24px));
  overflow: hidden;
  border: 1px solid rgba(238, 243, 235, .18);
  border-radius: 5px;
  background: rgba(14, 18, 18, .98);
  box-shadow: 0 18px 48px rgba(0, 0, 0, .54);
  backdrop-filter: blur(16px);
}

fieldset {
  min-width: 0;
  margin: 0;
  padding: 12px;
  border: 0;
}

fieldset + fieldset { border-top: 1px solid rgba(238, 243, 235, .1); }

legend {
  width: 100%;
  margin: 0 0 7px;
  padding: 0;
  color: #828c86;
  font: 500 11px/1.3 'DM Mono', monospace;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.view-option {
  min-height: 44px;
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 -5px;
  padding: 5px;
  border-radius: 3px;
  cursor: pointer;
}

.view-option:hover { background: rgba(255, 255, 255, .035); }

.checkbox-shell {
  position: relative;
  flex: 0 0 18px;
  width: 18px;
  height: 18px;
}

.checkbox-shell input {
  position: absolute;
  inset: 0;
  z-index: 1;
  width: 100%;
  height: 100%;
  margin: 0;
  opacity: 0;
  cursor: pointer;
}

.checkbox-shell i {
  width: 100%;
  height: 100%;
  display: grid;
  place-items: center;
  border: 1px solid rgba(238, 243, 235, .25);
  border-radius: 3px;
  background: #090c0c;
  transition: border-color .14s ease, background .14s ease;
}

.checkbox-shell svg {
  width: 11px;
  fill: none;
  stroke: #15160f;
  stroke-width: 2;
  opacity: 0;
}

.checkbox-shell input:checked + i {
  border-color: #ffd36a;
  background: #ffd36a;
}

.checkbox-shell input:checked + i svg { opacity: 1; }
.checkbox-shell input:focus-visible + i { outline: 2px solid #ffd36a; outline-offset: 2px; }

.option-copy,
.quality-options span {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.option-copy strong,
.quality-options strong {
  color: #dce2dd;
  font-size: 12px;
  font-weight: 600;
}

.option-copy small,
.quality-options small {
  color: #747e78;
  font-size: 11px;
  line-height: 1.35;
}

.quality-options {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
}

.inferred-position-options { grid-template-columns: 1fr; }

.inferred-position-options label > span {
  min-height: 0;
  align-items: flex-start;
  text-align: left;
}

.quality-options label { position: relative; cursor: pointer; }

.quality-options input {
  position: absolute;
  width: 1px;
  height: 1px;
  opacity: 0;
}

.quality-options label > span {
  min-height: 57px;
  justify-content: center;
  padding: 8px 9px;
  border: 1px solid rgba(238, 243, 235, .12);
  border-radius: 3px;
  background: #0a0d0d;
}

.quality-options input:checked + span {
  border-color: rgba(255, 211, 106, .58);
  background: rgba(255, 211, 106, .065);
}

.quality-options input:checked + span strong { color: #ffd36a; }
.quality-options input:focus-visible + span { outline: 2px solid #ffd36a; outline-offset: 2px; }

fieldset p {
  margin: 8px 0 0;
  color: #626c66;
  font-size: 11px;
  line-height: 1.4;
}

@media (prefers-reduced-motion: reduce) {
  .three-view-trigger,
  .chevron,
  .checkbox-shell i { transition: none; }
}
</style>
