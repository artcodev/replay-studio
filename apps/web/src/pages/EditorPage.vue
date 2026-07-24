<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue'
import { useRoute, useRouter, type RouteLocationRaw } from 'vue-router'
import EditorWorkspaceSurface from '../components/editor/EditorWorkspaceSurface.vue'
import { useSceneDocumentState } from '../composables/useSceneDocumentState'
import { useEditorAnalysisContext } from '../features/editor/analysis/useEditorAnalysisContext'
import { useEditorCompositionContext } from '../features/editor/composition/useEditorCompositionContext'
import { provideEditorContexts } from '../features/editor/editorContexts'
import { useEditorIdentityContext } from '../features/editor/identity/useEditorIdentityContext'
import { useEditorSessionContext } from '../features/editor/session/useEditorSessionContext'
import { useEditorViewportContext } from '../features/editor/viewport/useEditorViewportContext'
import { appRouteLocation, projectsRoute } from '../lib/appRoutes'

const router = useRouter()
const route = useRoute()

async function navigateToProjects() {
  await router.push(appRouteLocation(projectsRoute()) as RouteLocationRaw)
}

const document = useSceneDocumentState()
const viewport = useEditorViewportContext(document)
const session = useEditorSessionContext({
  route,
  router,
  document,
  viewport,
  exitEditor: navigateToProjects,
})
const analysis = useEditorAnalysisContext(session, viewport)
const composition = useEditorCompositionContext(session, viewport, analysis)
const identity = useEditorIdentityContext(session, viewport, analysis, composition)
provideEditorContexts({ session, viewport, analysis, composition, identity })

function onKeydown(event: KeyboardEvent) {
  if ((event.target as HTMLElement)?.matches('input, select, textarea, button, [role="button"]')) return
  if (event.code === 'Space') {
    event.preventDefault()
    viewport.togglePlay()
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
    // Every edit persists immediately through its own dedicated scene command,
    // so there is no whole-scene save to trigger — just suppress the browser's
    // native save-page dialog.
    event.preventDefault()
  }
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onBeforeUnmount(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <EditorWorkspaceSurface />
</template>
