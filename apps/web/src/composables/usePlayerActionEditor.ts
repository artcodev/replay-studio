import { computed, ref, watch, type Ref } from 'vue'
import { playerActionClient } from '../lib/api/playerActions'
import {
  activePlayerActionPlaybackState,
  filterPlayerActionsForActor,
} from '../lib/playerActions'
import type { SceneDocument } from '../types/scene'
import type { PlayerAction } from '../types/playerActions'
import { buildManualPlayerAction, roundActionTime } from '../features/player-actions/playerActionDraft'

type PlayerActionEditorOptions = {
  scene: Ref<SceneDocument | null>
  currentTime: Ref<number>
  selectedActorId: Ref<string | null>
  mutationLocked: Ref<boolean>
  playing: Ref<boolean>
  sourceVideo: Ref<HTMLVideoElement | null>
  saveState: Ref<string>
  error: Ref<string | null>
  projectId: () => string
  seekTo: (time: number) => void
  notifySceneMutation: () => void
}

/** Coordinates manual player actions; playback and selection stay owned by the editor. */
export function usePlayerActionEditor(options: PlayerActionEditorOptions) {
  const selectedActionId = ref<string | null>(null)
  const saving = ref(false)
  const actions = computed<PlayerAction[]>(() => options.scene.value?.payload.playerActions ?? [])
  const selectedActorActions = computed(() => filterPlayerActionsForActor(
    actions.value,
    options.selectedActorId.value,
  ))
  const activePlayback = computed(() => activePlayerActionPlaybackState(
    actions.value,
    options.currentTime.value,
    options.selectedActorId.value,
  ))
  const visible = computed(() => Boolean(
    options.scene.value && options.selectedActorId.value,
  ))

  async function persist(action: PlayerAction, successMessage: string) {
    const activeScene = options.scene.value
    if (
      !activeScene
      || saving.value
      || options.mutationLocked.value
      || action.source !== 'manual'
    ) return
    const previousActions = [...(activeScene.payload.playerActions ?? [])]
    const nextAction: PlayerAction = {
      ...action,
      startTime: roundActionTime(action.startTime),
      endTime: roundActionTime(action.endTime),
      keypoints: action.keypoints.map((keypoint) => ({
        ...keypoint,
        time: roundActionTime(keypoint.time),
      })),
    }
    const existingIndex = previousActions.findIndex((item) => item.id === nextAction.id)
    activeScene.payload.playerActions = existingIndex < 0
      ? [...previousActions, nextAction]
      : previousActions.map((item, index) => index === existingIndex ? nextAction : item)
    options.notifySceneMutation()
    selectedActionId.value = nextAction.id
    saving.value = true
    options.saveState.value = 'Saving player action…'
    options.error.value = null
    try {
      const updated = await playerActionClient.upsert(options.projectId(), activeScene.id, nextAction)
      if (options.scene.value?.id !== activeScene.id) return
      options.scene.value = updated
      selectedActionId.value = nextAction.id
      options.saveState.value = successMessage
    } catch (cause) {
      if (options.scene.value?.id === activeScene.id) {
        options.scene.value.payload.playerActions = previousActions
        options.notifySceneMutation()
      }
      options.error.value = cause instanceof Error ? cause.message : 'Could not save the player action'
      options.saveState.value = 'Player action change was not saved'
    } finally {
      saving.value = false
    }
  }

  function addAt(time: number) {
    const canonicalPersonId = options.selectedActorId.value
    if (!canonicalPersonId) {
      options.error.value = 'Select a resolved player before adding an action'
      return
    }
    const action = buildManualPlayerAction(
      canonicalPersonId,
      options.scene.value?.duration ?? 0,
      time,
    )
    if (!action) {
      options.error.value = 'This scene is too short for an action interval'
      return
    }
    options.playing.value = false
    options.sourceVideo.value?.pause()
    selectedActionId.value = action.id
    void persist(action, `Added ${action.type} action at ${time.toFixed(2)}s`)
  }

  function select(actionId: string) {
    if (selectedActorActions.value.some((item) => item.id === actionId)) {
      selectedActionId.value = actionId
    }
  }

  function seek(time: number) {
    options.playing.value = false
    options.sourceVideo.value?.pause()
    options.seekTo(time)
  }

  function update(action: PlayerAction) {
    if (action.canonicalPersonId !== options.selectedActorId.value) {
      options.error.value = 'The action belongs to a different canonical player'
      return
    }
    void persist(action, `Saved ${action.type} action`)
  }

  async function remove(actionId: string) {
    const activeScene = options.scene.value
    const action = selectedActorActions.value.find((item) => item.id === actionId)
    if (
      !activeScene
      || !action
      || action.source !== 'manual'
      || saving.value
      || options.mutationLocked.value
    ) return
    const previousActions = [...(activeScene.payload.playerActions ?? [])]
    activeScene.payload.playerActions = previousActions.filter((item) => item.id !== actionId)
    options.notifySceneMutation()
    selectedActionId.value = null
    saving.value = true
    options.saveState.value = 'Removing player action…'
    options.error.value = null
    try {
      const updated = await playerActionClient.remove(options.projectId(), activeScene.id, actionId)
      if (options.scene.value?.id !== activeScene.id) return
      options.scene.value = updated
      options.saveState.value = `Removed ${action.type} action`
    } catch (cause) {
      if (options.scene.value?.id === activeScene.id) {
        options.scene.value.payload.playerActions = previousActions
        options.notifySceneMutation()
      }
      selectedActionId.value = actionId
      options.error.value = cause instanceof Error ? cause.message : 'Could not remove the player action'
      options.saveState.value = 'Player action removal was not saved'
    } finally {
      saving.value = false
    }
  }

  function reset() {
    selectedActionId.value = null
    saving.value = false
  }

  watch(options.selectedActorId, () => { selectedActionId.value = null })

  return {
    selectedActionId,
    saving,
    actions,
    selectedActorActions,
    activePlayback,
    visible,
    addAt,
    select,
    seek,
    update,
    remove,
    reset,
  }
}
