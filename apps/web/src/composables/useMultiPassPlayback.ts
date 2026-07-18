import { computed, type ComputedRef, type Ref, type ShallowRef } from 'vue'
import type { SceneDocument, SceneVideoAsset } from '../types/scene'

type MultiPassPlaybackOptions = {
  scene: ShallowRef<SceneDocument | null>
  sceneVideo: ComputedRef<SceneVideoAsset | null>
  activePassSceneId: Ref<string | null>
}

function interpolateMapping(
  value: number,
  anchors: Array<{ referenceTime: number; passTime: number }>,
  input: 'referenceTime' | 'passTime',
  output: 'referenceTime' | 'passTime',
) {
  const ordered = [...anchors].sort((left, right) => left[input] - right[input])
  if (!ordered.length) return value
  if (value <= ordered[0][input]) return ordered[0][output]
  if (value >= ordered[ordered.length - 1][input]) return ordered[ordered.length - 1][output]
  for (let index = 1; index < ordered.length; index += 1) {
    const left = ordered[index - 1]
    const right = ordered[index]
    if (value > right[input]) continue
    const width = right[input] - left[input]
    if (width <= 0.0001) return right[output]
    const progress = (value - left[input]) / width
    return left[output] + (right[output] - left[output]) * progress
  }
  return value
}

/** Maps the canonical editor clock onto one selected replay-angle clock. */
export function useMultiPassPlayback(options: MultiPassPlaybackOptions) {
  const analysis = computed(() => options.sceneVideo.value?.multiPass ?? null)
  const activePass = computed(() => {
    const current = analysis.value
    if (!current) return null
    return current.passes.find((item) => item.sceneId === options.activePassSceneId.value)
      ?? current.passes.find((item) => item.sceneId === current.referenceSceneId)
      ?? current.passes[0]
      ?? null
  })
  const sourceStart = computed(() => (
    activePass.value?.sourceStart ?? options.sceneVideo.value?.sourceStart ?? 0
  ))
  const sourceEnd = computed(() => (
    activePass.value?.sourceEnd
      ?? options.sceneVideo.value?.sourceEnd
      ?? options.scene.value?.duration
      ?? 0
  ))

  function canonicalToPassTime(time: number) {
    const pass = activePass.value
    const duration = Math.max(0.01, options.scene.value?.duration ?? 0.01)
    const passDuration = Math.max(0.01, sourceEnd.value - sourceStart.value)
    if (!pass || pass.sceneId === analysis.value?.referenceSceneId) {
      return Math.min(time, passDuration)
    }
    const alignment = pass.alignment
    if (alignment?.overlap && alignment.anchors.length > 1) {
      return interpolateMapping(time, alignment.anchors, 'referenceTime', 'passTime')
    }
    return Math.min(passDuration, Math.max(0, time / duration * passDuration))
  }

  function passToCanonicalTime(time: number) {
    const pass = activePass.value
    const duration = Math.max(0.01, options.scene.value?.duration ?? 0.01)
    const passDuration = Math.max(0.01, sourceEnd.value - sourceStart.value)
    if (!pass || pass.sceneId === analysis.value?.referenceSceneId) {
      return Math.min(time, duration)
    }
    const alignment = pass.alignment
    if (alignment?.overlap && alignment.anchors.length > 1) {
      return interpolateMapping(time, alignment.anchors, 'passTime', 'referenceTime')
    }
    return Math.min(duration, Math.max(0, time / passDuration * duration))
  }

  function relationLabel(relation?: string) {
    if (relation === 'reference') return 'reference'
    if (relation === 'replay-overlap') return 'aligned replay'
    if (relation === 'continuation-before') return 'earlier context'
    if (relation === 'continuation-after') return 'later context'
    return 'independent'
  }

  return {
    analysis,
    activePass,
    sourceStart,
    sourceEnd,
    canonicalToPassTime,
    passToCanonicalTime,
    relationLabel,
  }
}
