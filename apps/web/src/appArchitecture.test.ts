import { describe, expect, it } from 'vitest'
import appSource from './App.vue?raw'
import editorPageSource from './pages/EditorPage.vue?raw'
import editorSurfaceSource from './components/editor/EditorWorkspaceSurface.vue?raw'
import editorSessionSource from './features/editor/session/useEditorSessionContext.ts?raw'
import editorViewportSource from './features/editor/viewport/useEditorViewportContext.ts?raw'
import editorAnalysisSource from './features/editor/analysis/useEditorAnalysisContext.ts?raw'
import editorCompositionSource from './features/editor/composition/useEditorCompositionContext.ts?raw'
import editorIdentitySource from './features/editor/identity/useEditorIdentityContext.ts?raw'
import projectWorkspaceSource from './composables/useProjectWorkspace.ts?raw'
import viewportSource from './components/ThreeViewport.vue?raw'
import editorViewportSurfaceSource from './components/editor/EditorViewportSurface.vue?raw'
import videoReviewPaneSource from './components/editor/VideoReviewPane.vue?raw'
import calibrationOverlaySource from './components/editor/PitchCalibrationOverlay.vue?raw'
import calibrationPanelSource from './components/editor/PitchCalibrationPanel.vue?raw'
import frameDetectionOverlaySource from './components/editor/FrameDetectionOverlay.vue?raw'
import threeScenePaneSource from './components/editor/ThreeScenePane.vue?raw'
import actionTimelineSource from './components/PlayerActionTimeline.vue?raw'
import manualBallTimelineSource from './components/ManualBallTimeline.vue?raw'
import manualBallTimelineDomainSource from './features/manual-ball/manualBallTimelineDomain.ts?raw'
import manualBallEditorSource from './composables/useManualBallEditor.ts?raw'
import manualBallTrajectorySource from './features/manual-ball/manualBallTrajectory.ts?raw'
import frameAnnotationsSource from './composables/useFrameAnnotations.ts?raw'
import frameAnnotationDraftSource from './features/frame-annotations/frameAnnotationDraft.ts?raw'
import frameAnnotationPointerSource from './features/frame-annotations/useFrameAnnotationPointer.ts?raw'
import frameAnalysisSource from './composables/useFrameAnalysis.ts?raw'
import frameAnalysisSelectionSource from './features/frame-analysis/frameAnalysisSelection.ts?raw'
import pitchCalibrationEditorSource from './composables/usePitchCalibrationEditor.ts?raw'
import pitchCalibrationPresentationSource from './features/calibration/usePitchCalibrationPresentation.ts?raw'
import calibrationQaSource from './components/CalibrationQaPanel.vue?raw'
import calibrationQaPresentationSource from './features/calibration/calibrationQaPresentation.ts?raw'
import identityReviewPanelSource from './components/IdentityReviewPanel.vue?raw'
import identityRosterSelectionSource from './features/identity-review/useIdentityRosterSelection.ts?raw'
import selectionLayerSource from './features/three-viewport/selectionLayer.ts?raw'
import pathLayerSource from './features/three-viewport/selectedPathLayer.ts?raw'
import lightingSource from './features/three-viewport/viewportLighting.ts?raw'
import pitchLayerSource from './features/three-viewport/pitchLayer.ts?raw'
import playerLayerSource from './features/three-viewport/playerLayer.ts?raw'
import ballLayerSource from './features/three-viewport/ballLayer.ts?raw'
import analysisMarkerLayerSource from './features/three-viewport/analysisMarkerLayer.ts?raw'
import renderSurfaceSource from './features/three-viewport/viewportRenderSurface.ts?raw'
import pointerSelectionSource from './features/three-viewport/viewportPointerSelection.ts?raw'

const styleSource = (globalThis as typeof globalThis & {
  process: {
    getBuiltinModule(name: 'node:fs'): {
      readFileSync(path: URL, encoding: 'utf8'): string
    }
  }
}).process.getBuiltinModule('node:fs').readFileSync(
  new URL('./style.css', import.meta.url),
  'utf8',
)

const sourceModules = import.meta.glob('./**/*.{ts,vue}', {
  eager: true,
  query: '?raw',
  import: 'default',
}) as Record<string, string>

describe('application shell architecture', () => {
  it('keeps HTTP ownership outside App.vue', () => {
    expect(appSource).not.toMatch(/from ['"]\.\/lib\/api['"]/)
    expect(appSource).not.toMatch(/\bapi\./)
    expect(appSource).not.toMatch(/\bfetch\s*\(/)
  })

  it('uses the router as the root composition boundary', () => {
    expect(appSource).toContain('<RouterView />')
    expect(appSource).not.toMatch(/useRoute\s*\(/)
    expect(appSource).not.toMatch(/parseAppRoute|RouteState|requestAnimationFrame\s*\(/)
    expect(editorPageSource).toContain('useEditorSessionContext')
    expect(editorPageSource).toContain('useEditorViewportContext')
    expect(editorPageSource).toContain('useEditorAnalysisContext')
    expect(editorPageSource).toContain('useEditorCompositionContext')
    expect(editorPageSource).toContain('useEditorIdentityContext')
  })

  it('preserves the editor route flex-height contract across its surface boundary', () => {
    const rule = styleSource.match(/\.editor-workspace-surface\s*\{([^}]*)\}/)?.[1] ?? ''
    expect(rule).toMatch(/display:\s*flex/)
    expect(rule).toMatch(/flex-direction:\s*column/)
    expect(rule).toMatch(/flex:\s*1\s+1\s+auto/)
    expect(rule).toMatch(/min-height:\s*0/)
    expect(rule).toMatch(/overflow:\s*hidden/)
  })

  it('gives every viewport layout an explicit bounded surface', () => {
    expect(editorViewportSurfaceSource).toContain("'split-view':")
    expect(editorViewportSurfaceSource).toContain("'video-only':")
    expect(editorViewportSurfaceSource).toContain("'three-only':")
    expect(styleSource).toMatch(/\.viewport-wrap\.three-only\s*\{[^}]*display:\s*block/)
    expect(styleSource).toMatch(/\.viewport-wrap\.three-only \.three-pane\s*\{[^}]*height:\s*100%/)
  })

  it('keeps the editor surface presentation-only', () => {
    expect(editorSurfaceSource).toContain('injectEditorContexts')
    expect(editorSurfaceSource).not.toMatch(/from ['"][^'"]*composables\//)
    expect(editorSurfaceSource).not.toMatch(/\buse(?:Reconstruction|Frame|Manual|Player|Pitch|Model|Segment|Composition|Identity)/)
    expect(editorSurfaceSource.match(/injectEditorContexts\s*\(/g)).toHaveLength(1)
  })

  it('enforces explicit editor context ownership', () => {
    for (const source of [
      editorSessionSource,
      editorViewportSource,
      editorAnalysisSource,
      editorCompositionSource,
      editorIdentitySource,
    ]) {
      expect(source).not.toMatch(/bindIntegration|registerSaveGuard|register(?:Before|After)Load|serviceLocator/)
    }
    expect(editorSessionSource).not.toMatch(/use(?:Reconstruction|FrameAnalysis|ManualBall|PlayerAction|IdentityReview)/)
    expect(editorViewportSource).not.toMatch(/api\/|useProjectWorkspace|useSceneSession/)
    expect(editorAnalysisSource).not.toMatch(/use(?:ManualBall|PlayerAction|CompositionEditor|IdentityReview)/)
    expect(editorCompositionSource).not.toMatch(/use(?:IdentityReview|ProjectMatchEditor|ReconstructionController)/)
    expect(editorIdentitySource).not.toMatch(/use(?:PlaybackClock|SceneSession|ManualBallEditor)/)
  })

  it('composes project resources without a broad writable facade', () => {
    expect(projectWorkspaceSource).toContain('useProjectCatalog')
    expect(projectWorkspaceSource).toContain('useProjectMatchResource')
    expect(projectWorkspaceSource).toContain('useProjectMediaResource')
    expect(projectWorkspaceSource).toContain('useProjectIdentityResource')
    expect(projectWorkspaceSource).not.toMatch(/const (projects|project|match|assets|segments|identities) = ref/)
    expect(projectWorkspaceSource).toContain('return { catalog, match, media, identities, jobs, load, dispose }')
  })

  it('has no compatibility type or API barrels', () => {
    expect(sourceModules).not.toHaveProperty('./types.ts')
    expect(sourceModules).not.toHaveProperty('./lib/api.ts')
    for (const [path, source] of Object.entries(sourceModules)) {
      if (path.endsWith('.test.ts')) continue
      expect(source, path).not.toMatch(/from ['"][^'"]*(?:\/types|\/lib\/api)['"]/)
    }
  })

  it('keeps the playback loop and large scene reactivity out of the root shell', () => {
    expect(appSource).not.toMatch(/requestAnimationFrame\s*\(/)
    expect(appSource).not.toMatch(/ref<SceneDocument/)
  })

  it('keeps the editor viewport surface as layout-only composition', () => {
    expect(editorViewportSurfaceSource).toContain("./VideoReviewPane.vue")
    expect(editorViewportSurfaceSource).toContain("./PitchCalibrationOverlay.vue")
    expect(editorViewportSurfaceSource).toContain("./PitchCalibrationPanel.vue")
    expect(editorViewportSurfaceSource).toContain("./FrameDetectionOverlay.vue")
    expect(editorViewportSurfaceSource).toContain("./ThreeScenePane.vue")
    expect(editorViewportSurfaceSource).not.toMatch(/<(?:video|svg|canvas)\b/)
    expect(editorViewportSurfaceSource).not.toMatch(/calibrationPointResidual|rawLinePoints|annotationIdentityAction|frameMetricBadge/)
    expect(editorViewportSurfaceSource).not.toMatch(/\bcontrollers\s*:/)
    for (const source of [
      videoReviewPaneSource,
      calibrationOverlaySource,
      calibrationPanelSource,
      frameDetectionOverlaySource,
      threeScenePaneSource,
    ]) {
      expect(source).not.toMatch(/from ['"][^'"]*composables\//)
    }
  })

  it('keeps Three runtime and scene layers framework-independent', () => {
    expect(viewportSource).toContain("../features/three-viewport/selectionLayer")
    expect(viewportSource).toContain("../features/three-viewport/selectedPathLayer")
    expect(viewportSource).toContain("../features/three-viewport/viewportRenderSurface")
    expect(viewportSource).toContain("../features/three-viewport/viewportPointerSelection")
    expect(viewportSource).toContain("../features/three-viewport/pitchLayer")
    expect(viewportSource).toContain("../features/three-viewport/playerLayer")
    expect(viewportSource).toContain("../features/three-viewport/ballLayer")
    expect(viewportSource).toContain("../features/three-viewport/analysisMarkerLayer")
    expect(viewportSource).not.toMatch(/new THREE\.|new OrbitControls|new ResizeObserver|new ViewportLighting/)
    for (const source of [
      selectionLayerSource,
      pathLayerSource,
      lightingSource,
      pitchLayerSource,
      playerLayerSource,
      ballLayerSource,
      analysisMarkerLayerSource,
      renderSurfaceSource,
      pointerSelectionSource,
    ]) {
      expect(source).not.toMatch(/from ['"]vue['"]/) 
    }
    expect(renderSurfaceSource).toContain('renderer.setAnimationLoop(null)')
    expect(renderSurfaceSource).toContain('THREE.PCFShadowMap')
    expect(renderSurfaceSource).not.toContain('PCFSoftShadowMap')
    expect(pointerSelectionSource).toContain('raycaster.setFromCamera')
    expect(pointerSelectionSource).toContain('intersectObjects(targets.players, true)')
  })

  it('keeps player-action domain reducers out of the Vue component', () => {
    expect(actionTimelineSource.match(/<script\b/g)).toHaveLength(1)
    expect(actionTimelineSource).toContain('playerActionTimelineDomain')
    expect(actionTimelineSource).not.toMatch(/export function (normalize|reduce|layout|clamp)PlayerAction/)
  })

  it('keeps manual-ball domain reducers out of the Vue component', () => {
    expect(manualBallTimelineSource.match(/<script\b/g)).toHaveLength(1)
    expect(manualBallTimelineSource).toContain('manualBallTimelineDomain')
    expect(manualBallTimelineSource).not.toMatch(/export function (?:clamp|normalize|manualBallTimelineEvents)/)
    expect(manualBallTimelineDomainSource).not.toMatch(/from ['"]vue['"]|document\.|window\./)
  })

  it('separates editor gestures, domain drafts and request orchestration', () => {
    expect(frameAnnotationsSource).toContain('frameAnnotationDraft')
    expect(frameAnnotationsSource).toContain('useFrameAnnotationPointer')
    expect(frameAnnotationsSource).not.toMatch(/selectFrameDetectionHit|clientPointToContainedMedia/)
    expect(frameAnnotationDraftSource).not.toMatch(/from ['"]vue['"]|document\.|window\./)
    expect(frameAnnotationPointerSource).not.toMatch(/frameAnalysisClient|saveAnnotation|deleteAnnotation/)

    expect(frameAnalysisSource).toContain('frameAnalysisSelection')
    expect(frameAnalysisSource).not.toMatch(/linkedFrameMetricSelectionStatus|videoTrackSelectionStatus/)
    expect(frameAnalysisSelectionSource).not.toMatch(/from ['"]vue['"]|frameAnalysisClient/)
  })

  it('keeps calibration and manual-ball presentation/domain logic outside command composables', () => {
    expect(pitchCalibrationEditorSource).toContain('usePitchCalibrationPresentation')
    expect(pitchCalibrationEditorSource).not.toMatch(/calibrationFrameDiagnostics|projectPitchMarkings/)
    expect(pitchCalibrationPresentationSource).not.toMatch(/calibrationClient|fetch\s*\(/)
    expect(calibrationQaSource).toContain('calibrationQaPresentation')
    expect(calibrationQaPresentationSource).not.toMatch(/from ['"]vue['"]|document\.|window\./)

    expect(manualBallEditorSource).toContain('manualBallTrajectory')
    expect(manualBallEditorSource).not.toMatch(/interpolateKeyframes/)
    expect(manualBallTrajectorySource).not.toMatch(/from ['"]vue['"]|reconstructionClient/)
  })

  it('keeps identity roster-picker state outside the review surface', () => {
    expect(identityReviewPanelSource).toContain('useIdentityRosterSelection')
    expect(identityReviewPanelSource).not.toMatch(/manualRosterBindingDecision|rosterNumberSortValue/)
    expect(identityRosterSelectionSource).not.toMatch(/identityReviewClient|fetch\s*\(/)
  })
})
