import { computed, nextTick, watch } from 'vue'
import { appRouteLocation, projectWorkspaceRoute } from '../../../lib/appRoutes'
import type { IdentityReviewInspectFrame } from '../../../lib/identityReview'
import { useIdentityReviewEditor } from '../../../composables/useIdentityReviewEditor'
import { useIdentityReviewPresentation } from '../../../composables/useIdentityReviewPresentation'
import { useProjectMatchEditor } from '../../../composables/useProjectMatchEditor'
import type { CanonicalMatchEvent } from '../../../types/match'
import type { EditorAnalysisContext } from '../analysis/useEditorAnalysisContext'
import type { EditorCompositionContext } from '../composition/useEditorCompositionContext'
import type { EditorSessionContext } from '../session/useEditorSessionContext'
import type { EditorViewportContext } from '../viewport/useEditorViewportContext'

/** Canonical identity decisions and the project match/roster projection. */
export function useEditorIdentityContext(
  session: EditorSessionContext,
  viewport: EditorViewportContext,
  analysis: EditorAnalysisContext,
  composition: EditorCompositionContext,
) {
  const rosterPlayers = computed(() => (session.workspaceMatch.value?.roster ?? []).map((player) => ({
    id: player.id,
    name: player.name,
    team_id: player.teamId,
    team_name: player.teamId === session.workspaceMatch.value?.homeTeam.id
      ? session.workspaceMatch.value.homeTeam.name
      : player.teamId === session.workspaceMatch.value?.awayTeam.id
        ? session.workspaceMatch.value.awayTeam.name
        : null,
    position: player.position,
    number: player.number,
    lineup_role: player.role === 'squad' ? 'unknown' as const : player.role,
  })))
  const matchSnapshotRefreshAvailable = computed(() => Boolean(
    session.workspaceMatch.value && session.workspaceMatch.value.sync.state !== 'manual',
  ))
  const projectMatchContext = computed(() => ({
    label: session.workspaceMatch.value
      ? `Project match data · ${session.workspaceMatch.value.sync.state}`
      : 'Project has no match data',
  }))
  const projectMatchTeams = computed(() => {
    const fallbackHome = session.scene.value?.payload.teams[0] ?? {
      id: 'home', name: 'Home', color: '#46d7c2', externalTeamId: null,
    }
    const fallbackAway = session.scene.value?.payload.teams[1] ?? {
      id: 'away', name: 'Away', color: '#f2c94c', externalTeamId: null,
    }
    return {
      home: session.workspaceMatch.value ? {
        ...fallbackHome,
        id: session.workspaceMatch.value.homeTeam.id,
        name: session.workspaceMatch.value.homeTeam.name,
        externalTeamId: session.workspaceMatch.value.homeTeam.id,
      } : fallbackHome,
      away: session.workspaceMatch.value ? {
        ...fallbackAway,
        id: session.workspaceMatch.value.awayTeam.id,
        name: session.workspaceMatch.value.awayTeam.name,
        externalTeamId: session.workspaceMatch.value.awayTeam.id,
      } : fallbackAway,
    }
  })

  const identityReview = useIdentityReviewEditor({
    projectId: session.editorProjectId,
    scene: session.scene,
    rosterPlayers: () => rosterPlayers.value,
    mutationLocked: () => analysis.reconstruction.mutationLocked.value,
    reconstructionRunning: () => analysis.reconstruction.running.value,
    reconstructing: analysis.reconstruction.reconstructing,
    selectedCanonicalPersonId: viewport.selectedCanonicalPersonId,
    selectedTrackId: viewport.selectedTrackId,
    selectedFramePersonId: viewport.selectedFramePersonId,
    saveState: session.saveState,
    error: session.error,
    canonicalPersonById: analysis.frameAnalysis.canonicalPersonById,
    renderTrackForCanonicalPerson: analysis.frameAnalysis.renderTrackForCanonicalPerson,
    hasDedicatedUnbind: composition.canonicalHasActiveDedicatedUnbind,
    clearFrameAnalysis: analysis.frameAnalysis.clear,
    startReconstructionPolling: analysis.reconstruction.startPolling,
  })
  const identityPresentation = useIdentityReviewPresentation({
    scene: session.scene,
    selectedPerson: composition.selection.selectedCanonicalPerson,
    snapshot: identityReview.snapshot,
    hasDedicatedUnbind: composition.canonicalHasActiveDedicatedUnbind,
  })
  const projectMatch = useProjectMatchEditor({
    projectId: session.editorProjectId,
    scene: session.scene,
    match: session.workspaceMatch,
    mutationLocked: () => analysis.reconstruction.mutationLocked.value,
    selectedModel: analysis.reconstruction.selectedModel,
    selectedBallBackend: analysis.reconstruction.selectedBallBackend,
    reconstructing: analysis.reconstruction.reconstructing,
    saveState: session.saveState,
    error: session.error,
    invalidateIdentityReview: identityReview.invalidate,
    clearFrameAnalysis: analysis.frameAnalysis.clear,
    loadIdentityReview: identityReview.load,
    startReconstructionPolling: analysis.reconstruction.startPolling,
    refreshWorkspace: () => session.selectedProject.value
      ? session.loadProjectsWorkspace(session.selectedProject.value.id)
      : undefined,
  })
  const busy = computed(() => Boolean(
    identityReview.decisionSaving.value
    || identityReview.rosterBindingSaving.value
    || projectMatch.refreshing.value
    || projectMatch.importing.value,
  ))

  async function openProjectMatchWorkspace() {
    const projectId = session.selectedProject.value?.id
    if (projectId) {
      await session.router.push(appRouteLocation(projectWorkspaceRoute(projectId, 'match')))
    }
  }

  function inspectIdentityFrame(payload: IdentityReviewInspectFrame) {
    if (!session.scene.value?.payload.canonicalPeople?.some(
      (person) => person.canonicalPersonId === payload.canonicalPersonId,
    )) return
    viewport.playing.value = false
    viewport.sourceVideo.value?.pause()
    viewport.selectedCanonicalPersonId.value = payload.canonicalPersonId
    viewport.selectedTrackId.value = analysis.frameAnalysis
      .renderTrackForCanonicalPerson(payload.canonicalPersonId)?.id ?? null
    viewport.viewMode.value = viewport.sceneVideo.value ? 'split' : viewport.viewMode.value
    viewport.activeTab.value = 'binding'
    viewport.seekTo(payload.sceneTime)
    session.saveState.value = `Identity observation · frame ${payload.frameIndex}`
    void nextTick().then(() => analysis.frameAnalysis.analyze())
  }

  function addEventBinding(item: CanonicalMatchEvent) {
    if (!session.mutateScene((document) => {
      document.payload.eventBindings.push({
        sceneTime: Number(viewport.currentTime.value.toFixed(2)),
        externalEventId: item.id,
        label: item.label,
        type: item.kind,
      })
    })) return
    session.saveState.value = 'Unsaved event marker'
  }

  function removeEventBinding(index: number) {
    if (!session.mutateScene((document) => {
      document.payload.eventBindings.splice(index, 1)
    })) return
    session.saveState.value = 'Unsaved changes'
  }

  watch(() => session.scene.value?.id, (sceneId) => {
    if (!sceneId) {
      identityReview.invalidate()
      return
    }
    projectMatch.importError.value = null
    void identityReview.load(sceneId)
  })
  watch(analysis.reconstruction.terminalSync, (result) => {
    if (
      result
      && result.status !== 'cancelled'
      && session.scene.value?.id === result.sceneId
    ) void identityReview.load(result.sceneId)
  })
  watch(
    [analysis.reconstruction.reconstructing, analysis.reconstruction.running],
    ([reconstructing, running]) => {
      if (reconstructing || running) identityReview.invalidate()
    },
  )

  return {
    rosterPlayers,
    matchSnapshotRefreshAvailable,
    projectMatchContext,
    projectMatchTeams,
    identityReview,
    identityPresentation,
    projectMatch,
    busy,
    openProjectMatchWorkspace,
    inspectIdentityFrame,
    addEventBinding,
    removeEventBinding,
  }
}

export type EditorIdentityContext = ReturnType<typeof useEditorIdentityContext>
