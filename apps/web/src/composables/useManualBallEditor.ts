import { computed, ref, type Ref } from 'vue'
import { reconstructionClient } from '../lib/api/reconstruction'
import type { ThreeViewOptions } from '../lib/threeViewOptions'
import {
  MANUAL_BALL_KEYFRAME_TOLERANCE,
  manualBallKeyframeAt,
  normalizeManualBallKeyframes,
} from '../features/manual-ball/manualBallTrajectory'
import type { BallTrajectoryMode } from '../types/reconstruction'
import type { Keyframe } from '../types/tracking'
import type { SceneDocument } from '../types/scene'

type ManualBallEditorOptions = {
  scene: Ref<SceneDocument | null>
  currentTime: Ref<number>
  playing: Ref<boolean>
  sourceVideo: Ref<HTMLVideoElement | null>
  selectedTrackId: Ref<string | null>
  selectedCanonicalPersonId: Ref<string | null>
  selectedFramePersonId: Ref<string | null>
  editMode: Ref<boolean>
  activeTab: Ref<'binding' | 'qa' | 'events'>
  viewMode: Ref<'video' | 'split' | '3d'>
  viewOptions: Ref<ThreeViewOptions>
  saveState: Ref<string>
  error: Ref<string | null>
  projectId: () => string
  seekTo: (time: number) => void
  notifySceneMutation: () => void
}

/** Owns the manual ball trajectory and its optimistic persistence. */
export function useManualBallEditor(options: ManualBallEditorOptions) {
  const selected = ref(false)
  const placementMode = ref(false)
  const saving = ref(false)
  const selectedKeyframeTime = ref<number | null>(null)
  const mode = computed<BallTrajectoryMode>(() => options.scene.value?.payload.ball.mode ?? 'automatic')
  const manualKeyframes = computed<Keyframe[]>(() => {
    const ball = options.scene.value?.payload.ball
    return ball?.manualKeyframes ?? (ball?.mode === 'manual' ? ball.keyframes : [])
  })
  const automaticKeyframes = computed<Keyframe[]>(() => {
    const ball = options.scene.value?.payload.ball
    return ball?.automaticKeyframes ?? (ball?.mode !== 'manual' ? ball?.keyframes ?? [] : [])
  })
  const selectedKeyframe = computed<Keyframe | null>(() => {
    const selectedTime = selectedKeyframeTime.value
    if (selectedTime === null) return null
    return manualKeyframes.value.find(
      (frame) => Math.abs(frame.t - selectedTime) < MANUAL_BALL_KEYFRAME_TOLERANCE,
    ) ?? null
  })

  function trajectoryBounds() {
    return {
      duration: options.scene.value?.duration ?? 0,
      pitchLength: options.scene.value?.payload.pitch.length ?? 105,
      pitchWidth: options.scene.value?.payload.pitch.width ?? 68,
    }
  }

  function keyframeAt(time: number, position?: { x: number; z: number }): Keyframe {
    const source = manualKeyframes.value.length ? manualKeyframes.value : automaticKeyframes.value
    return manualBallKeyframeAt(source, time, trajectoryBounds(), position)
  }

  async function persist(keyframes: Keyframe[], message: string, selectionTime: number | null) {
    const activeScene = options.scene.value
    if (!activeScene || saving.value) return
    const previousBall = {
      ...activeScene.payload.ball,
      keyframes: [...activeScene.payload.ball.keyframes],
      automaticKeyframes: activeScene.payload.ball.automaticKeyframes
        ? [...activeScene.payload.ball.automaticKeyframes]
        : undefined,
      manualKeyframes: activeScene.payload.ball.manualKeyframes
        ? [...activeScene.payload.ball.manualKeyframes]
        : undefined,
    }
    const normalized = normalizeManualBallKeyframes(keyframes, trajectoryBounds())
    activeScene.payload.ball = {
      ...activeScene.payload.ball,
      mode: 'manual',
      manualKeyframes: normalized,
      keyframes: normalized,
    }
    options.notifySceneMutation()
    selectedKeyframeTime.value = selectionTime
    saving.value = true
    options.saveState.value = 'Saving manual ball trajectory…'
    try {
      const updated = await reconstructionClient.updateBallTrajectory(options.projectId(), activeScene.id, 'manual', normalized)
      if (options.scene.value?.id !== activeScene.id) return
      options.scene.value = updated
      if (selectionTime !== null) {
        selectedKeyframeTime.value = manualKeyframes.value.find(
          (frame) => Math.abs(frame.t - selectionTime) < 0.0011,
        )?.t ?? null
      }
      options.saveState.value = message
    } catch (cause) {
      if (options.scene.value?.id === activeScene.id) {
        options.scene.value.payload.ball = previousBall
        options.notifySceneMutation()
      }
      options.error.value = cause instanceof Error ? cause.message : 'Could not save the manual ball trajectory'
      options.saveState.value = 'Manual ball change was not saved'
    } finally {
      saving.value = false
    }
  }

  function selectBall() {
    selected.value = true
    options.selectedTrackId.value = null
    options.selectedCanonicalPersonId.value = null
    options.selectedFramePersonId.value = null
    options.editMode.value = false
    options.activeTab.value = 'binding'
    options.viewOptions.value.ball = true
    if (mode.value === 'manual') {
      placementMode.value = true
      if (options.viewMode.value === 'video') {
        options.viewMode.value = options.scene.value?.payload.videoAsset ? 'split' : '3d'
      }
    }
  }

  function togglePlacement() {
    if (mode.value !== 'manual') {
      void setMode('manual')
      return
    }
    const next = !placementMode.value
    selectBall()
    placementMode.value = next
    options.saveState.value = next
      ? 'Click anywhere on the 3D pitch to place the ball'
      : 'Ball placement paused'
  }

  async function setMode(nextMode: BallTrajectoryMode) {
    if (!options.scene.value || saving.value || nextMode === mode.value) {
      if (nextMode === 'manual') selectBall()
      return
    }
    const sceneId = options.scene.value.id
    saving.value = true
    options.saveState.value = nextMode === 'manual'
      ? 'Opening manual ball trajectory…'
      : 'Restoring automatic ball trajectory…'
    try {
      const updated = await reconstructionClient.updateBallTrajectory(options.projectId(), sceneId, nextMode)
      if (options.scene.value?.id !== sceneId) return
      options.scene.value = updated
      selectBall()
      placementMode.value = nextMode === 'manual'
      selectedKeyframeTime.value = null
      options.saveState.value = nextMode === 'manual'
        ? 'Manual ball mode · add a keypoint or click the pitch'
        : 'Automatic ball trajectory restored'
    } catch (cause) {
      options.error.value = cause instanceof Error ? cause.message : 'Could not switch the ball trajectory mode'
    } finally {
      saving.value = false
    }
  }

  function changeMode(event: Event) {
    void setMode((event.target as HTMLSelectElement).value as BallTrajectoryMode)
  }

  function selectKeypoint(time: number) {
    const keyframe = manualKeyframes.value.find((frame) => Math.abs(frame.t - time) < 0.0011)
    if (!keyframe) return
    selectBall()
    selectedKeyframeTime.value = keyframe.t
    placementMode.value = true
    options.playing.value = false
    options.sourceVideo.value?.pause()
    options.seekTo(keyframe.t)
    options.saveState.value = `Ball keypoint selected at ${keyframe.t.toFixed(3)}s`
  }

  function addKeypoint(time = options.currentTime.value) {
    if (saving.value) return
    const timestamp = Number(Math.max(0, Math.min(options.scene.value?.duration ?? time, time)).toFixed(3))
    const existing = manualKeyframes.value.find(
      (frame) => Math.abs(frame.t - timestamp) < MANUAL_BALL_KEYFRAME_TOLERANCE,
    )
    if (existing) {
      selectKeypoint(existing.t)
      return
    }
    selectBall()
    options.playing.value = false
    options.sourceVideo.value?.pause()
    const next = keyframeAt(timestamp)
    void persist(
      [...manualKeyframes.value, next],
      `Ball keypoint added at ${timestamp.toFixed(3)}s`,
      timestamp,
    )
  }

  function move(position: { x: number; z: number }) {
    if (mode.value !== 'manual' || saving.value) return
    const timestamp = selectedKeyframeTime.value ?? Number(options.currentTime.value.toFixed(3))
    const current = manualKeyframes.value.find(
      (frame) => Math.abs(frame.t - timestamp) < MANUAL_BALL_KEYFRAME_TOLERANCE,
    ) ?? keyframeAt(timestamp, position)
    const moved = keyframeAt(timestamp, position)
    const next = manualKeyframes.value.filter(
      (frame) => Math.abs(frame.t - timestamp) >= MANUAL_BALL_KEYFRAME_TOLERANCE,
    )
    next.push({ ...current, ...moved })
    selected.value = true
    placementMode.value = true
    options.playing.value = false
    options.sourceVideo.value?.pause()
    void persist(next, `Ball placed at X ${moved.x.toFixed(2)} · Z ${moved.z.toFixed(2)}`, moved.t)
  }

  function removeKeypoint(time: number) {
    if (saving.value) return
    const next = manualKeyframes.value.filter(
      (frame) => Math.abs(frame.t - time) >= MANUAL_BALL_KEYFRAME_TOLERANCE,
    )
    const nearest = [...next].sort(
      (left, right) => Math.abs(left.t - time) - Math.abs(right.t - time),
    )[0] ?? null
    if (nearest) options.seekTo(nearest.t)
    void persist(next, `Ball keypoint removed from ${time.toFixed(3)}s`, nearest?.t ?? null)
  }

  function updateKeypointTime(payload: { from: number; to: number }) {
    if (saving.value) return
    const current = manualKeyframes.value.find(
      (frame) => Math.abs(frame.t - payload.from) < MANUAL_BALL_KEYFRAME_TOLERANCE,
    )
    if (!current) return
    const next = manualKeyframes.value.filter(
      (frame) => Math.abs(frame.t - payload.from) >= MANUAL_BALL_KEYFRAME_TOLERANCE,
    )
    const moved = keyframeAt(payload.to, { x: current.x, z: current.z })
    next.push({ ...current, ...moved })
    options.seekTo(moved.t)
    void persist(next, `Ball keypoint moved to ${moved.t.toFixed(3)}s`, moved.t)
  }

  function updateCoordinate(axis: 'x' | 'z', value: string) {
    const keyframe = selectedKeyframe.value
    const numeric = Number(value)
    if (!keyframe || !Number.isFinite(numeric)) return
    move({
      x: axis === 'x' ? numeric : keyframe.x,
      z: axis === 'z' ? numeric : keyframe.z,
    })
  }

  function syncPlayhead(time: number) {
    if (
      selectedKeyframeTime.value !== null
      && Math.abs(selectedKeyframeTime.value - time) > 0.0011
    ) selectedKeyframeTime.value = null
  }

  function clearSelection() {
    selected.value = false
    placementMode.value = false
    selectedKeyframeTime.value = null
  }

  function reset() {
    clearSelection()
    saving.value = false
  }

  return {
    selected,
    placementMode,
    saving,
    selectedKeyframeTime,
    selectedKeyframe,
    mode,
    manualKeyframes,
    automaticKeyframes,
    selectBall,
    togglePlacement,
    setMode,
    changeMode,
    selectKeypoint,
    addKeypoint,
    move,
    removeKeypoint,
    updateKeypointTime,
    updateCoordinate,
    syncPlayhead,
    clearSelection,
    reset,
  }
}
