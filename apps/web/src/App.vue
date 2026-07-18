<script setup lang="ts">
import { onBeforeUnmount } from 'vue'
import { RouterView, useRouter } from 'vue-router'
import { provideProjectWorkspace, useProjectWorkspace } from './composables/useProjectWorkspace'

const workspace = useProjectWorkspace(useRouter())
provideProjectWorkspace(workspace)
onBeforeUnmount(workspace.dispose)
</script>

<template>
  <main class="app-shell">
    <RouterView />
  </main>
</template>
