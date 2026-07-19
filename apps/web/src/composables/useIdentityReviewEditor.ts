import { ref, type Ref } from 'vue'
import { identityClient } from '../lib/api/identities'
import type { IdentityReviewCandidateDecision } from '../lib/identityReview'
import type { CanonicalPerson } from '../types/identity'
import type { SceneDocument } from '../types/scene'
import type { Track } from '../types/tracking'
import type {
  IdentityReviewArtifactCapability,
  IdentityReviewResponse,
} from '../types/identityReview'
import type { ExternalPlayer } from '../types/match'

type IdentityReviewEditorOptions = {
  projectId: () => string
  scene: Ref<SceneDocument | null>
  rosterPlayers: () => ExternalPlayer[]
  mutationLocked: () => boolean
  reconstructionRunning: () => boolean
  reconstructing: Ref<boolean>
  selectedCanonicalPersonId: Ref<string | null>
  selectedTrackId: Ref<string | null>
  selectedFramePersonId: Ref<string | null>
  activeTab: Ref<'binding' | 'qa' | 'events'>
  saveState: Ref<string>
  error: Ref<string | null>
  canonicalPersonById: (id: string) => CanonicalPerson | null
  renderTrackForCanonicalPerson: (id: string) => Track | null
  hasDedicatedUnbind: (id: string) => boolean
  clearFrameAnalysis: () => void
  startReconstructionPolling: (sceneId: string) => Promise<void>
}

/** Identity evidence and roster decisions are one server-backed capability. */
export function useIdentityReviewEditor(options: IdentityReviewEditorOptions) {
  const snapshot = ref<IdentityReviewResponse | null>(null)
  const loading = ref(false)
  const reviewError = ref<string | null>(null)
  const decisionSaving = ref(false)
  const rosterBindingSaving = ref(false)
  let requestId = 0
  let loadedKey: string | null = null
  let inFlight: { key: string; promise: Promise<void> } | null = null

  type LoadTarget = IdentityReviewArtifactCapability & {
    projectId: string
    key: string
  }

  function loadTarget(sceneId?: string): LoadTarget | null {
    const scene = options.scene.value
    const selectedPersonId = options.selectedCanonicalPersonId.value
    if (
      !scene
      || (sceneId !== undefined && scene.id !== sceneId)
      || !selectedPersonId
      || options.activeTab.value !== 'binding'
      || options.reconstructing.value
      || options.reconstructionRunning()
      || !scene.payload.canonicalPeople?.some(
        (person) => person.canonicalPersonId === selectedPersonId,
      )
    ) return null

    const video = scene.payload.videoAsset
    const reconstruction = video?.reconstruction
    const diagnostics = reconstruction?.artifactManifest?.artifacts.identityDiagnostics
    const timeline = reconstruction?.artifactManifest?.artifacts.identityTimeline
    if (
      !video?.selectedSegmentId
      || reconstruction?.status !== 'ready'
      || !diagnostics
      || !timeline
    ) return null

    const projectId = options.projectId()
    return {
      projectId,
      sceneId: scene.id,
      revision: scene.revision,
      identityDiagnostics: diagnostics,
      identityTimeline: timeline,
      key: JSON.stringify([
        projectId,
        scene.id,
        scene.revision,
        diagnostics.id,
        diagnostics.sha256,
        timeline.id,
        timeline.sha256,
      ]),
    }
  }

  function invalidate() {
    requestId += 1
    loadedKey = null
    inFlight = null
    snapshot.value = null
    loading.value = false
    reviewError.value = null
  }

  function requestReview(sceneId: string | undefined, force: boolean): Promise<void> {
    const target = loadTarget(sceneId)
    if (!target) return Promise.resolve()
    if (!force && loadedKey === target.key && snapshot.value) return Promise.resolve()
    if (!force && inFlight?.key === target.key) return inFlight.promise

    const currentRequest = ++requestId
    loading.value = true
    reviewError.value = null
    let promise!: Promise<void>
    promise = (async () => {
      try {
        const review = await identityClient.review(target.projectId, target.sceneId)
        if (currentRequest !== requestId || loadTarget(target.sceneId)?.key !== target.key) return
        if (review.sceneId !== target.sceneId || review.revision !== target.revision) {
          loadedKey = null
          snapshot.value = null
          reviewError.value = 'Identity review changed with the scene; reload the review.'
          return
        }
        loadedKey = target.key
        snapshot.value = review
      } catch (cause) {
        if (currentRequest !== requestId || loadTarget(target.sceneId)?.key !== target.key) return
        loadedKey = null
        snapshot.value = null
        reviewError.value = cause instanceof Error
          ? cause.message
          : 'Could not load identity review evidence'
      } finally {
        if (inFlight?.promise === promise) inFlight = null
        if (currentRequest === requestId) loading.value = false
      }
    })()
    inFlight = { key: target.key, promise }
    return promise
  }

  function ensureLoaded(sceneId?: string) {
    return requestReview(sceneId, false)
  }

  function reload(sceneId?: string) {
    return requestReview(sceneId, true)
  }

  function selectCanonicalPerson(canonicalPersonId: string) {
    options.selectedCanonicalPersonId.value = canonicalPersonId
    options.selectedTrackId.value = options.renderTrackForCanonicalPerson(canonicalPersonId)?.id ?? null
    options.selectedFramePersonId.value = null
  }

  async function applyQueuedDecision(
    queued: SceneDocument,
    canonicalPersonId: string,
    pendingLabel: string,
    readyLabel: string,
  ) {
    const sceneId = queued.id
    if (options.scene.value?.id !== sceneId) return
    invalidate()
    options.scene.value = queued
    options.clearFrameAnalysis()
    selectCanonicalPerson(canonicalPersonId)
    const status = queued.payload.videoAsset?.reconstruction?.status
    if (status === 'queued' || status === 'processing') {
      options.reconstructing.value = true
      options.saveState.value = pendingLabel
      void options.startReconstructionPolling(sceneId)
    } else {
      options.saveState.value = readyLabel
      void ensureLoaded(sceneId)
    }
  }

  async function updateBinding(canonicalPersonId: string, externalPlayerId: string | null) {
    if (
      !options.scene.value
      || options.reconstructing.value
      || options.reconstructionRunning()
      || rosterBindingSaving.value
    ) return
    const sceneId = options.scene.value.id
    rosterBindingSaving.value = true
    try {
      const queued = await identityClient.updateRosterBinding(
        options.projectId(),
        sceneId,
        canonicalPersonId,
        externalPlayerId,
      )
      if (options.scene.value?.id !== sceneId) return
      await applyQueuedDecision(
        queued,
        canonicalPersonId,
        externalPlayerId
          ? 'Roster binding saved · rebuilding identity…'
          : 'Roster binding removed · rebuilding identity…',
        externalPlayerId ? 'Roster binding saved' : 'Roster binding removed',
      )
    } catch (cause) {
      options.error.value = cause instanceof Error
        ? cause.message
        : externalPlayerId
          ? 'Could not save roster binding'
          : 'Could not remove roster binding'
    } finally {
      rosterBindingSaving.value = false
    }
  }

  async function confirm(payload: { canonicalPersonId: string; externalPlayerId: string }) {
    const identity = options.canonicalPersonById(payload.canonicalPersonId)
    const rosterPlayer = options.rosterPlayers().find((player) => player.id === payload.externalPlayerId)
    if (!identity || !rosterPlayer) {
      options.error.value = 'The selected roster candidate is no longer available'
      return
    }
    await updateBinding(payload.canonicalPersonId, rosterPlayer.id)
  }

  async function unbind(payload: { canonicalPersonId: string }) {
    if (!options.canonicalPersonById(payload.canonicalPersonId)) {
      options.error.value = 'The selected canonical identity is no longer available'
      return
    }
    await updateBinding(payload.canonicalPersonId, null)
  }

  async function reject(payload: IdentityReviewCandidateDecision) {
    if (payload.kind !== 'roster') {
      options.error.value = 'Identity-link rejection is not available from this review snapshot'
      return
    }
    if (
      !options.scene.value
      || decisionSaving.value
      || rosterBindingSaving.value
      || options.mutationLocked()
    ) return
    if (!options.canonicalPersonById(payload.canonicalPersonId)) {
      options.error.value = 'The selected canonical identity is no longer available'
      return
    }
    if (!options.rosterPlayers().some((player) => player.id === payload.externalPlayerId)) {
      options.error.value = 'The roster candidate is absent from the saved match snapshot'
      return
    }
    const sceneId = options.scene.value.id
    decisionSaving.value = true
    options.error.value = null
    options.saveState.value = 'Rejecting roster hypothesis…'
    try {
      const updated = await identityClient.rejectRosterCandidate(
        options.projectId(),
        sceneId,
        payload.canonicalPersonId,
        payload.externalPlayerId,
      )
      if (options.scene.value?.id !== sceneId) return
      await applyQueuedDecision(
        updated,
        payload.canonicalPersonId,
        'Roster hypothesis rejected · rebuilding identity…',
        'Roster hypothesis rejected',
      )
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not reject roster candidate'
      options.saveState.value = 'Roster rejection was not saved'
    } finally {
      decisionSaving.value = false
    }
  }

  async function clearBinding(payload: { canonicalPersonId: string }) {
    if (
      !options.scene.value
      || options.mutationLocked()
      || rosterBindingSaving.value
    ) return
    if (!options.hasDedicatedUnbind(payload.canonicalPersonId)) {
      options.error.value = 'There is no active manual Unbind decision to clear for this identity'
      return
    }
    const sceneId = options.scene.value.id
    rosterBindingSaving.value = true
    try {
      const queued = await identityClient.clearRosterBinding(
        options.projectId(),
        sceneId,
        payload.canonicalPersonId,
      )
      if (options.scene.value?.id !== sceneId) return
      await applyQueuedDecision(
        queued,
        payload.canonicalPersonId,
        'Manual Unbind cleared · rebuilding identity…',
        'Manual Unbind cleared',
      )
    } catch (cause) {
      options.error.value = cause instanceof Error
        ? cause.message
        : 'Could not clear the manual roster Unbind decision'
    } finally {
      rosterBindingSaving.value = false
    }
  }

  return {
    snapshot,
    loading,
    error: reviewError,
    decisionSaving,
    rosterBindingSaving,
    ensureLoaded,
    reload,
    load: reload,
    invalidate,
    confirm,
    reject,
    unbind,
    clearBinding,
    updateBinding,
  }
}
