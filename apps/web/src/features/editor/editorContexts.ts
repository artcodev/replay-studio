import { inject, provide, type InjectionKey } from 'vue'
import type { EditorAnalysisContext } from './analysis/useEditorAnalysisContext'
import type { EditorCompositionContext } from './composition/useEditorCompositionContext'
import type { EditorIdentityContext } from './identity/useEditorIdentityContext'
import type { EditorSessionContext } from './session/useEditorSessionContext'
import type { EditorViewportContext } from './viewport/useEditorViewportContext'

export type EditorContexts = {
  session: EditorSessionContext
  viewport: EditorViewportContext
  analysis: EditorAnalysisContext
  composition: EditorCompositionContext
  identity: EditorIdentityContext
}

const editorContextsKey: InjectionKey<EditorContexts> = Symbol('editor-contexts')

export function provideEditorContexts(contexts: EditorContexts) {
  provide(editorContextsKey, contexts)
}

export function injectEditorContexts(): EditorContexts {
  const contexts = inject(editorContextsKey)
  if (!contexts) throw new Error('Editor contexts must be provided by EditorPage.')
  return contexts
}
