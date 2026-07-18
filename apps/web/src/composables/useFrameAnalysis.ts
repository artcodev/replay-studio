import { computed, ref, type Ref } from 'vue'
import { frameAnalysisClient } from '../lib/api/frameAnalysis'
import { trackPresenceAtTime } from '../lib/trackPresence'
import {
  canonicalSelectionAfterFrameAnalysis,
  selectedFramePeople,
  selectionAfterFrameAnalysis,
} from '../lib/videoTrackSelection'
import {
  bestFramePersonForCanonicalIdentity,
  canonicalIdForFramePerson,
  canonicalPersonForId,
  frameAnalysisSelectionStatus,
  framePersonDisplayLabel,
  renderTrackForCanonicalPerson as findRenderTrackForCanonicalPerson,
  validFrameMatchedTrackId,
} from '../features/frame-analysis/frameAnalysisSelection'
import type { FrameAnalysis } from '../types/analysis'
import type { SceneDocument } from '../types/scene'
import type { Track } from '../types/tracking'

type FrameAnalysisOptions = {
  scene: Ref<SceneDocument | null>
  currentTime: Ref<number>
  selectedTrackId: Ref<string | null>
  selectedCanonicalPersonId: Ref<string | null>
  selectedFramePersonId: Ref<string | null>
  activeTab: Ref<'binding' | 'qa' | 'events'>
  playing: Ref<boolean>
  sourceVideo: Ref<HTMLVideoElement | null>
  saveState: Ref<string>
  error: Ref<string | null>
  projectId: () => string
  seekTo: (time: number) => void
  clearAnnotations: () => void
}

/** Current-frame recognition and bidirectional video/3D selection. */
export function useFrameAnalysis(options: FrameAnalysisOptions) {
  const analyzing = ref(false)
  const analysis = ref<FrameAnalysis | null>(null)
  let requestId = 0
  let activeRequest: { sceneId: string; sceneTime: number } | null = null

  const activeAnalysis = computed(() => {
    const value = analysis.value
    return value && Math.abs(options.currentTime.value - value.sceneTime) <= 0.11 ? value : null
  })
  const selectedTrack = computed<Track | null>(() => (
    options.scene.value?.payload.tracks.find(
      (track) => track.id === options.selectedTrackId.value,
    ) ?? null
  ))
  const selectedFramePerson = computed(() => (
    activeAnalysis.value?.people.find(
      (person) => person.id === options.selectedFramePersonId.value,
    ) ?? null
  ))
  const selectedTrackPresence = computed(() => (
    selectedTrack.value ? trackPresenceAtTime(selectedTrack.value, options.currentTime.value) : null
  ))

  function canonicalPersonById(canonicalPersonId: string | null | undefined) {
    return canonicalPersonForId(options.scene.value?.payload.canonicalPeople, canonicalPersonId)
  }

  function renderTrackForCanonicalPerson(canonicalPersonId: string | null | undefined) {
    return findRenderTrackForCanonicalPerson(
      options.scene.value?.payload.tracks ?? [],
      canonicalPersonId,
    )
  }

  function validMatchedTrackId(person: FrameAnalysis['people'][number]) {
    return validFrameMatchedTrackId(person, options.scene.value?.payload.tracks ?? [])
  }

  function framePersonCanonicalId(person: FrameAnalysis['people'][number]) {
    return canonicalIdForFramePerson(person, options.scene.value?.payload.tracks ?? [])
  }

  function framePersonLabel(person: FrameAnalysis['people'][number]) {
    return framePersonDisplayLabel(
      person,
      options.scene.value?.payload.tracks ?? [],
      options.scene.value?.payload.canonicalPeople,
    )
  }

  function bestFramePersonForTrack(
    source: FrameAnalysis | null,
    canonicalPersonId: string | null = null,
  ) {
    return bestFramePersonForCanonicalIdentity(source, canonicalPersonId)
  }

  const selectionStatus = computed(() => frameAnalysisSelectionStatus({
    analysis: activeAnalysis.value,
    person: selectedFramePerson.value,
    tracks: options.scene.value?.payload.tracks ?? [],
    canonicalPeople: options.scene.value?.payload.canonicalPeople,
    selectedTrack: selectedTrack.value,
    selectedCanonicalPersonId: options.selectedCanonicalPersonId.value,
    analyzing: analyzing.value,
    selectedTrackObserved: selectedTrackPresence.value?.observed ?? null,
  }))

  function selectTrack(trackId: string) {
    const track = options.scene.value?.payload.tracks.find((item) => item.id === trackId) ?? null
    options.selectedTrackId.value = trackId
    options.selectedCanonicalPersonId.value = track?.canonicalPersonId ?? null
    options.selectedFramePersonId.value = bestFramePersonForTrack(
      activeAnalysis.value,
      track?.canonicalPersonId ?? null,
    )?.id ?? null
    if (track) options.saveState.value = `${track.label} selected in video and 3D`
  }

  function selectCanonicalPerson(canonicalPersonId: string) {
    const renderTrack = renderTrackForCanonicalPerson(canonicalPersonId)
    options.selectedCanonicalPersonId.value = canonicalPersonId
    options.selectedTrackId.value = renderTrack?.id ?? null
    options.selectedFramePersonId.value = bestFramePersonForTrack(
      activeAnalysis.value,
      canonicalPersonId,
    )?.id ?? null
    const identity = canonicalPersonById(canonicalPersonId)
    options.saveState.value = renderTrack
      ? `${identity?.displayName ?? canonicalPersonId} selected in video and 3D`
      : `${identity?.displayName ?? canonicalPersonId} selected · not projected in 3D`
  }

  async function run(preserveTrackId?: string): Promise<FrameAnalysis | null> {
    const scene = options.scene.value
    if (!scene?.payload.videoAsset?.selectedSegmentId) return null
    const requestedTime = options.currentTime.value
    if (
      analyzing.value
      && activeRequest?.sceneId === scene.id
      && Math.abs(activeRequest.sceneTime - requestedTime) <= 0.11
    ) return null
    const currentRequestId = ++requestId
    activeRequest = { sceneId: scene.id, sceneTime: requestedTime }
    const selectionAtStart = options.selectedTrackId.value
    const canonicalSelectionAtStart = options.selectedCanonicalPersonId.value
    const framePersonSelectionAtStart = options.selectedFramePersonId.value
    options.playing.value = false
    options.sourceVideo.value?.pause()
    analyzing.value = true
    options.saveState.value = preserveTrackId
      ? `Matching ${selectedTrack.value?.label ?? preserveTrackId} in source frame…`
      : `Analyzing frame at ${requestedTime.toFixed(2)}s…`
    try {
      const result = await frameAnalysisClient.analyze(options.projectId(), scene.id, requestedTime)
      if (currentRequestId !== requestId) return null
      if (options.scene.value?.id !== scene.id || Math.abs(options.currentTime.value - requestedTime) > 0.11) {
        options.saveState.value = 'Frame changed · discarded stale analysis result'
        return null
      }
      analysis.value = result
      options.seekTo(result.sceneTime)
      const firstPerson = result.people.find((item) => item.canonicalPersonId || item.matchedTrackId) ?? null
      const firstCanonicalId = firstPerson ? framePersonCanonicalId(firstPerson) : null
      const firstMatch = renderTrackForCanonicalPerson(firstCanonicalId)?.id
        ?? (firstPerson ? validMatchedTrackId(firstPerson) : null)
      const canonicalSelectionChanged = options.selectedCanonicalPersonId.value !== canonicalSelectionAtStart
      options.selectedTrackId.value = selectionAfterFrameAnalysis(
        selectionAtStart,
        options.selectedTrackId.value,
        canonicalSelectionAtStart || canonicalSelectionChanged ? null : firstMatch,
        preserveTrackId,
      )
      const selectedRenderTrack = options.scene.value.payload.tracks.find(
        (track) => track.id === options.selectedTrackId.value,
      )
      options.selectedCanonicalPersonId.value = canonicalSelectionAfterFrameAnalysis(
        canonicalSelectionAtStart,
        options.selectedCanonicalPersonId.value,
        options.selectedTrackId.value,
        selectedRenderTrack?.canonicalPersonId,
        firstCanonicalId,
      )
      const matchedPerson = bestFramePersonForTrack(
        result,
        options.selectedCanonicalPersonId.value,
      )
      if (matchedPerson) options.selectedFramePersonId.value = matchedPerson.id
      else if (!options.selectedTrackId.value) {
        const requestedPersonId = options.selectedFramePersonId.value !== framePersonSelectionAtStart
          ? options.selectedFramePersonId.value
          : framePersonSelectionAtStart
        options.selectedFramePersonId.value = result.people.some(
          (person) => person.id === requestedPersonId,
        ) ? requestedPersonId : null
      } else options.selectedFramePersonId.value = null
      options.activeTab.value = 'binding'
      const selectedMatches = options.selectedCanonicalPersonId.value
        ? selectedFramePeople(
          result,
          options.selectedCanonicalPersonId.value,
        ).length
        : 0
      options.saveState.value = preserveTrackId
        ? selectedMatches
          ? `${selectedTrack.value?.label ?? options.selectedTrackId.value} linked across video and 3D`
          : `${selectedTrack.value?.label ?? options.selectedTrackId.value} is not visible in this source frame`
        : `${result.people.length} people · ${result.matchedTracks} matched at ${result.sceneTime.toFixed(2)}s`
      return result
    } catch (cause) {
      if (currentRequestId === requestId) {
        options.error.value = cause instanceof Error ? cause.message : 'Could not analyze this frame'
      }
      return null
    } finally {
      if (currentRequestId === requestId) {
        analyzing.value = false
        activeRequest = null
      }
    }
  }

  async function selectTrackFromThree(trackId: string) {
    selectTrack(trackId)
    if (activeAnalysis.value) {
      const matches = selectedFramePeople(
        activeAnalysis.value,
        options.selectedCanonicalPersonId.value,
      )
      options.saveState.value = matches.length
        ? `${selectedTrack.value?.label ?? trackId} selected in video and 3D`
        : `${selectedTrack.value?.label ?? trackId} selected · no visible source detection`
      return
    }
    await run(trackId)
  }

  function clear() {
    requestId += 1
    activeRequest = null
    analyzing.value = false
    analysis.value = null
    options.selectedFramePersonId.value = null
    options.clearAnnotations()
  }

  return {
    analyzing,
    analysis,
    activeAnalysis,
    selectedFramePerson,
    selectedTrackPresence,
    selectionStatus,
    canonicalPersonById,
    renderTrackForCanonicalPerson,
    validMatchedTrackId,
    framePersonCanonicalId,
    framePersonLabel,
    bestFramePersonForTrack,
    selectTrack,
    selectCanonicalPerson,
    run,
    analyze: () => run(),
    selectTrackFromThree,
    clear,
  }
}
