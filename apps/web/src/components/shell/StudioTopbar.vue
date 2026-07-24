<script setup lang="ts">
import type { Project } from '../../types/project'

defineProps<{
  surface: 'projects' | 'editor'
  sceneTitle: string | null
  project: Project | null
  projectCount: number
  assetCount: number
  segmentCount: number
  saveState: string
  projectLoading: boolean
}>()

const emit = defineEmits<{
  'update:sceneTitle': [value: string]
  'open-import': []
  'return-projects': []
}>()

function updateTitle(event: Event) {
  emit('update:sceneTitle', (event.target as HTMLInputElement).value)
}
</script>

<template>
  <header class="topbar">
    <div class="brand-block">
      <div class="brand-mark"><span>R</span></div>
      <div>
        <p class="eyebrow">Interactive football lab</p>
        <h1>Replay Studio <span>α</span></h1>
      </div>
    </div>

    <div v-if="surface === 'editor' && sceneTitle !== null" class="moment-title">
      <input :value="sceneTitle" aria-label="Moment title" @input="updateTitle" />
    </div>
    <div v-else class="moment-title project-title">
      <strong>{{ project?.title || 'Projects' }}</strong>
    </div>

    <div class="top-actions">
      <template v-if="surface === 'projects'">
        <span class="save-state">{{ project ? `${assetCount} videos · ${segmentCount} moments` : projectCount ? 'Choose a project to open' : 'Create a project to begin' }}</span>
        <button class="button import-button" :disabled="!project || projectLoading" @click="emit('open-import')">＋ Import clip</button>
      </template>
      <template v-else>
        <span class="save-state">{{ saveState }}</span>
        <button class="button ghost" @click="emit('return-projects')">← Projects</button>
        <button class="button import-button" :disabled="!project" @click="emit('open-import')">＋ Import clip</button>
      </template>
    </div>
  </header>
</template>
