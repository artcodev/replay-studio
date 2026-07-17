<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import CalibrationQaPanel from './components/CalibrationQaPanel.vue'
import IdentityReviewPanel from './components/IdentityReviewPanel.vue'
import ManualBallTimeline from './components/ManualBallTimeline.vue'
import PathTrackingLegend from './components/PathTrackingLegend.vue'
import PlayerActionTimeline from './components/PlayerActionTimeline.vue'
import TrackPresenceCard from './components/TrackPresenceCard.vue'
import ThreeViewport from './components/ThreeViewport.vue'
import ThreeViewMenu from './components/ThreeViewMenu.vue'
import ToolbarDisclosure from './components/ToolbarDisclosure.vue'
import VideoPathTrackingOverlay from './components/VideoPathTrackingOverlay.vue'
import VideoIngestDrawer from './components/VideoIngestDrawer.vue'
import { api } from './lib/api'
import {
  calibrationFrameDiagnostics as buildCalibrationFrameDiagnostics,
  calibrationLineResidualLabel,
  calibrationPreviewWarnings,
  calibrationRejectionReasonLabel,
} from './lib/calibrationDiagnostics'
import { interpolateKeyframes, upsertKeyframe } from './lib/interpolate'
import { parseManualMatchImport } from './lib/matchImport'
import {
  API_FOOTBALL_PROVIDER_ID,
  LEGACY_MATCH_DATA_PROVIDERS,
  matchDataProviderLabel,
  matchDataProviderStatus,
  resolveMatchDataProvider,
} from './lib/matchDataProviders'
import {
  identityReviewItemObservations,
  identityReviewWorkerStates,
  type IdentityReviewCandidateDecision,
  type IdentityReviewInspectFrame,
} from './lib/identityReview'
import { pathHasVisibleProjection, resolvePathProjectionContext } from './lib/pathProjection'
import {
  buildPathTrackingSegments,
  pathTrackingOptionsForSubject,
  pathTrackingPoints,
} from './lib/pathTracking'
import {
  activePlayerActionPlaybackState,
  defaultPlayerActionDuration,
  defaultPlayerActionKeypointKind,
  filterPlayerActionsForActor,
} from './lib/playerActions'
import {
  annotationIdentityAction,
  buildIdentityMergeTargets,
  confirmedRosterBindingsConflict,
  dedicatedRosterMergeCompatible,
  hasActiveDedicatedUnbindForOwner,
  identitySplitObservationCounts,
  identitySplitRangeIsValid,
  semanticAnnotationForEdit,
} from './lib/identityCorrections'
import { selectFrameDetectionHit } from './lib/frameDetectionHitTest'
import { projectPitchMarkings } from './lib/pitchProjection'
import {
  identityValidationSummary,
  matchBindingNeedsRefresh,
  mergeFrameReconstructionMetadata,
  persistedEventBundle,
  projectMatchBindingContext,
  reconstructionLocksMutations,
  resolvedProjectMatchTeams,
  resolvedRosterPlayers,
} from './lib/reconstructionUi'
import { trackPresenceAtTime } from './lib/trackPresence'
import {
  DEFAULT_THREE_VIEW_OPTIONS,
  type ThreeRenderQuality,
  type ThreeViewOptions,
} from './lib/threeViewOptions'
import { loadThreeViewPreferences, saveThreeViewPreferences } from './lib/threeViewPreferences'
import {
  clampVideoReviewTransform,
  clientPointToContainedMedia,
  panVideoReviewTransform,
  VIDEO_REVIEW_MAX_SCALE,
  VIDEO_REVIEW_MIN_SCALE,
  zoomVideoReviewTransform,
} from './lib/videoReviewTransform'
import type { VideoReviewTransform } from './lib/videoReviewTransform'
import {
  canonicalSelectionAfterFrameAnalysis,
  frameMetricBadge,
  linkedFrameMetricSelectionStatus,
  renderTrackForFramePerson,
  selectedFramePeople,
  selectionAfterFrameAnalysis,
  videoTrackSelectionStatus,
} from './lib/videoTrackSelection'
import type { BallDetectionBackend, BallTrajectoryMode, CalibrationEvidenceLine, CalibrationEvidencePoint, CalibrationFrameEvidence, CanonicalPerson, ExternalEvent, FrameAnalysis, FrameAnnotation, FrameAnnotationKind, FrameIdentityAction, FrameIdentityScope, IdentityReviewResponse, Keyframe, MatchDataProvider, MatchDataProviderId, PitchCalibrationDraft, PitchCalibrationPreset, PlayerAction, PlayerActionType, ProcessingStatus, QualityVerdict, ReconstructionModel, ReconstructionPhase, SceneDocument, SceneSummary, TimelineEvent, Track, VideoAsset, VideoSegment } from './types'

type CameraName = 'broadcast' | 'orbit' | 'tactical' | 'goal'
type ViewMode = 'video' | 'split' | '3d'
type ViewportApi = { cameraPreset: (name: CameraName) => void }
const MANUAL_BALL_TIME_TOLERANCE = 0.0005
type FrameAnnotationDraft = {
  annotationId: string | null
  bbox: { x: number; y: number; width: number; height: number }
  kind: FrameAnnotationKind
  label: string
  externalPlayerId: string | null
  action: FrameIdentityAction
  scope: FrameIdentityScope
  mergeTargetId: string | null
  sourceTrackId: string | null
  canonicalPersonId: string | null
  targetObservationId: string | null
  rangeStart: number | null
  rangeEnd: number | null
  affectedPreview: FrameAnnotation['affectedPreview']
}

const scene = ref<SceneDocument | null>(null)
const scenes = ref<SceneSummary[]>([])
const currentTime = ref(0)
const selectedTrackId = ref<string | null>(null)
const selectedCanonicalPersonId = ref<string | null>(null)
const selectedFramePersonId = ref<string | null>(null)
const trackQuery = ref('')
const playing = ref(false)
const playbackRate = ref(1)
const editMode = ref(false)
const viewOptions = ref<ThreeViewOptions>({ ...DEFAULT_THREE_VIEW_OPTIONS })
const renderQuality = ref<ThreeRenderQuality>('basic')
const viewMode = ref<ViewMode>('split')
const activeCamera = ref<CameraName>('broadcast')
const saving = ref(false)
const saveState = ref('Saved locally')
const error = ref<string | null>(null)
const viewport = ref<ViewportApi | null>(null)
const catalogOpen = ref(false)
const catalogDate = ref(new Date().toISOString().slice(0, 10))
const catalogQuery = ref('Spain vs Belgium')
const catalogEvents = ref<ExternalEvent[]>([])
const catalogLoading = ref(false)
const catalogError = ref<string | null>(null)
const providerCatalogLoading = ref(false)
const catalogProviders = ref<MatchDataProvider[]>([
  ...LEGACY_MATCH_DATA_PROVIDERS.providers,
])
// The actual default comes from provider discovery. Until then use the only
// truthful legacy-server default instead of briefly presenting an unconfigured
// API-Football adapter as active.
const selectedCatalogProvider = ref<MatchDataProviderId>(
  LEGACY_MATCH_DATA_PROVIDERS.defaultProvider,
)
const bundleLoading = ref<string | null>(null)
const matchSnapshotRefreshing = ref(false)
const manualRosterImporting = ref(false)
const manualRosterImportError = ref<string | null>(null)
const manualRosterFileInput = ref<HTMLInputElement | null>(null)
const identityReviewSnapshot = ref<IdentityReviewResponse | null>(null)
const identityReviewLoading = ref(false)
const identityReviewError = ref<string | null>(null)
const identityDecisionSaving = ref(false)
const activeTab = ref<'binding' | 'qa' | 'events'>('binding')
const videoIngestOpen = ref(false)
const sourceVideo = ref<HTMLVideoElement | null>(null)
const videoReviewViewport = ref<HTMLDivElement | null>(null)
const videoReviewTransform = ref<VideoReviewTransform>({ scale: 1, x: 0, y: 0 })
const videoReviewPanDrag = ref<{
  pointerId: number
  clientX: number
  clientY: number
  transform: VideoReviewTransform
} | null>(null)
const segmentCreating = ref<string | null>(null)
const reconstructing = ref(false)
const selectedReconstructionModel = ref<ReconstructionModel>('yolo26m.pt')
const selectedBallBackend = ref<BallDetectionBackend>('dedicated-ultralytics')
const ballSelected = ref(false)
const ballEditMode = ref(false)
const ballTrajectorySaving = ref(false)
const selectedBallKeyframeTime = ref<number | null>(null)
const selectedPlayerActionId = ref<string | null>(null)
const playerActionSaving = ref(false)
const frameAnalyzing = ref(false)
const frameAnalysis = ref<FrameAnalysis | null>(null)
const frameAnalysisOverlay = ref<SVGSVGElement | null>(null)
const frameAnnotationMode = ref(false)
const frameAnnotationDraft = ref<FrameAnnotationDraft | null>(null)
const frameAnnotationDrag = ref<{ x: number; y: number; pointerId: number } | null>(null)
const frameAnnotationSaving = ref(false)
const rosterBindingSaving = ref(false)
const modelComparing = ref(false)
const calibrationDraft = ref<PitchCalibrationDraft | null>(null)
const calibrationPreset = ref<PitchCalibrationPreset>('penalty-area-right')
const calibrationLoading = ref(false)
const calibrationApplying = ref(false)
const pitchSideSaving = ref(false)
const calibrationOverlay = ref<SVGSVGElement | null>(null)
const draggedCalibrationAnchor = ref<string | null>(null)
const multiPassSelection = ref<string[]>([])
const multiPassStarting = ref(false)
const activePassSceneId = ref<string | null>(null)
const layoutRebuilding = ref(false)
const timelineGroupEditing = ref(false)
let animationFrame = 0
let previousTime = 0
let reconstructionTimer = 0
let frameAnalysisRequestId = 0
let identityReviewRequestId = 0
let activeFrameAnalysisRequest: { sceneId: string; sceneTime: number } | null = null
let frameDetectionHitCycle: { frameIndex: number; x: number; y: number; personId: string } | null = null
let suppressNextFrameOverlayClick = false
let videoReviewResizeObserver: ResizeObserver | null = null

const selectedTrack = computed<Track | null>(() => {
  if (!scene.value || !selectedTrackId.value) return null
  return scene.value.payload.tracks.find((track) => track.id === selectedTrackId.value) ?? null
})
const selectedCanonicalPerson = computed<CanonicalPerson | null>(() => {
  const canonicalPersonId = selectedCanonicalPersonId.value ?? selectedTrack.value?.canonicalPersonId
  if (!scene.value || !canonicalPersonId) return null
  return scene.value.payload.canonicalPeople?.find(
    (person) => person.canonicalPersonId === canonicalPersonId,
  ) ?? null
})
const selectedIdentityReviewItem = computed(() => {
  const identity = selectedCanonicalPerson.value
  const review = identityReviewSnapshot.value
  if (!identity || !review || review.sceneId !== scene.value?.id) return null
  if (scene.value?.revision !== undefined && review.revision !== scene.value.revision) return null
  return review.items.find(
    (item) => item.canonicalPersonId === identity.canonicalPersonId,
  ) ?? null
})
const selectedIdentityReviewPerson = computed<CanonicalPerson | null>(() => {
  const identity = selectedCanonicalPerson.value
  if (!identity) return null
  const item = selectedIdentityReviewItem.value
  const rejectedIds = new Set(
    (scene.value?.payload.identityReviewDecisions?.rosterRejections ?? [])
      .filter((decision) => decision.canonicalPersonId === identity.canonicalPersonId)
      .map((decision) => decision.externalPlayerId),
  )
  return {
    ...identity,
    ...(item
      ? {
          displayName: item.displayName,
          identityStatus: item.identityStatus,
          identityConfidence: item.identityConfidence ?? null,
          identitySource: item.identitySource ?? null,
          teamId: item.teamId ?? null,
          role: item.role ?? null,
          jerseyNumber: item.jerseyNumber ?? null,
          externalPlayerId: item.externalPlayerId ?? null,
          observationCount: item.observationCount,
          evidence: item.evidence,
          rosterCandidates: item.rosterCandidates,
          conflicts: item.conflicts,
        }
      : {}),
    rosterCandidates: (item?.rosterCandidates ?? identity.rosterCandidates).filter(
      (candidate) => !rejectedIds.has(candidate.externalPlayerId),
    ),
  }
})
const selectedIdentityReviewObservations = computed(() => (
  identityReviewItemObservations(selectedIdentityReviewItem.value)
))
const identityReviewWorkers = computed(() => identityReviewWorkerStates(identityReviewSnapshot.value))
const selectedActionActorId = computed<string | null>(() => {
  if (ballSelected.value) return null
  return selectedCanonicalPerson.value?.canonicalPersonId
    ?? selectedTrack.value?.canonicalPersonId
    ?? null
})
const selectedActionActorLabel = computed(() => (
  selectedCanonicalPerson.value?.displayName
  ?? selectedTrack.value?.label
  ?? 'Selected player'
))
const playerActions = computed<PlayerAction[]>(() => scene.value?.payload.playerActions ?? [])
const selectedActorActions = computed(() => filterPlayerActionsForActor(
  playerActions.value,
  selectedActionActorId.value,
))
const activePlayerActionPlayback = computed(() => activePlayerActionPlaybackState(
  playerActions.value,
  currentTime.value,
  selectedActionActorId.value,
))
const showPlayerActionTimeline = computed(() => Boolean(
  scene.value && selectedActionActorId.value && !ballSelected.value,
))
const selectedPathSubject = computed(() => {
  if (ballSelected.value) {
    const keyframes = scene.value?.payload.ball.keyframes ?? []
    return {
      kind: 'ball' as const,
      label: 'Match ball',
      color: '#5ee7ff',
      sampleCount: pathTrackingPoints(keyframes).length,
    }
  }
  const track = selectedTrack.value
  if (!track) return null
  return {
    kind: 'player' as const,
    label: track.label,
    color: track.color,
    sampleCount: pathTrackingPoints(track.keyframes).length,
  }
})
const selectedPathKeyframes = computed<Keyframe[]>(() => (
  ballSelected.value
    ? scene.value?.payload.ball.keyframes ?? []
    : selectedTrack.value?.keyframes ?? []
))
const selectedPathSegments = computed(() => {
  const kind = selectedPathSubject.value?.kind
  return kind
    ? buildPathTrackingSegments(
      selectedPathKeyframes.value,
      pathTrackingOptionsForSubject(kind),
    )
    : []
})
const unavailablePathSubjectLabel = computed(() => (
  !ballSelected.value && !selectedTrack.value
    ? selectedCanonicalPerson.value?.displayName ?? null
    : null
))
const selectedCanonicalDedicatedUnbindActive = computed(() => (
  selectedCanonicalPerson.value
    ? canonicalHasActiveDedicatedUnbind(selectedCanonicalPerson.value.canonicalPersonId)
    : false
))
const filteredTracks = computed(() => {
  const tracks = scene.value?.payload.tracks ?? []
  const query = trackQuery.value.trim().toLowerCase()
  if (!query) return tracks
  return tracks.filter((track) => [
    track.label,
    track.id,
    track.number,
    track.teamId,
    track.externalPlayerId,
  ].some((value) => String(value ?? '').toLowerCase().includes(query)))
})
const canonicalPeopleWithoutRender = computed(() => {
  const rendered = new Set(
    (scene.value?.payload.tracks ?? [])
      .map((track) => track.canonicalPersonId)
      .filter((id): id is string => Boolean(id)),
  )
  return (scene.value?.payload.canonicalPeople ?? []).filter(
    (person) => !rendered.has(person.canonicalPersonId) && person.identityStatus !== 'excluded',
  )
})
const filteredCanonicalPeopleWithoutRender = computed(() => {
  const query = trackQuery.value.trim().toLowerCase()
  if (!query) return canonicalPeopleWithoutRender.value
  return canonicalPeopleWithoutRender.value.filter((person) => [
    person.displayName,
    person.canonicalPersonId,
    person.jerseyNumber,
    person.teamId,
    person.externalPlayerId,
  ].some((value) => String(value ?? '').toLowerCase().includes(query)))
})
const ballMatchesTrackQuery = computed(() => {
  const query = trackQuery.value.trim().toLowerCase()
  return !query || 'match ball'.includes(query)
})
const ballTrajectoryMode = computed<BallTrajectoryMode>(() => (
  scene.value?.payload.ball.mode ?? 'automatic'
))
const manualBallKeyframes = computed<Keyframe[]>(() => {
  const ball = scene.value?.payload.ball
  if (!ball) return []
  return ball.manualKeyframes ?? (ball.mode === 'manual' ? ball.keyframes : [])
})
const automaticBallKeyframes = computed<Keyframe[]>(() => {
  const ball = scene.value?.payload.ball
  if (!ball) return []
  return ball.automaticKeyframes ?? (ball.mode !== 'manual' ? ball.keyframes : [])
})
const selectedManualBallKeyframe = computed<Keyframe | null>(() => {
  const selectedTime = selectedBallKeyframeTime.value
  if (selectedTime === null) return null
  return manualBallKeyframes.value.find(
    (frame) => Math.abs(frame.t - selectedTime) < MANUAL_BALL_TIME_TOLERANCE,
  ) ?? null
})
const reconstructionPreviewScene = computed<SceneDocument | null>(() => {
  const current = scene.value
  if (!current) return null
  const hiddenTrackIds = new Set(
    (current.payload.videoAsset?.reconstruction?.frameAnnotations ?? [])
      .filter((annotation) => (
        annotation.scope === 'identity'
        && ['exclude', 'merge'].includes(annotationIdentityAction(annotation))
        && annotation.sourceTrackId
      ))
      .map((annotation) => annotation.sourceTrackId as string),
  )
  if (!hiddenTrackIds.size) return current
  return {
    ...current,
    payload: {
      ...current.payload,
      tracks: current.payload.tracks.filter((track) => !hiddenTrackIds.has(track.id)),
    },
  }
})

const selectedTeam = computed(() => {
  if (!scene.value) return null
  const teamId = selectedTrack.value?.teamId ?? selectedCanonicalPerson.value?.teamId
  return scene.value.payload.teams.find((team) => team.id === teamId) ?? null
})

watch(selectedTrackId, (trackId) => {
  if (!trackId) return
  ballSelected.value = false
  ballEditMode.value = false
  selectedBallKeyframeTime.value = null
  const track = scene.value?.payload.tracks.find((item) => item.id === trackId)
  selectedCanonicalPersonId.value = track?.canonicalPersonId ?? null
})

const rosterPlayers = computed(() => resolvedRosterPlayers(scene.value))
const eventBundle = computed(() => persistedEventBundle(scene.value))
const matchSnapshotRefreshAvailable = computed(() => matchBindingNeedsRefresh(scene.value))
const projectMatchContext = computed(() => projectMatchBindingContext(scene.value))
const projectMatchTeams = computed(() => resolvedProjectMatchTeams(scene.value))
const selectedMatchDataProvider = computed(() => (
  catalogProviders.value.find((provider) => provider.id === selectedCatalogProvider.value) ?? null
))
const selectedMatchDataProviderReady = computed(() => (
  selectedMatchDataProvider.value?.configured === true
  && selectedMatchDataProvider.value.available === true
))
const unavailableCatalogProviders = computed(() => catalogProviders.value.filter(
  (provider) => !provider.configured || !provider.available,
))
function providerUnavailableReason(provider: MatchDataProvider | null): string {
  if (!provider) return 'The selected match-data provider is unknown to the API server.'
  if (provider.reason) return provider.reason
  if (provider.id === API_FOOTBALL_PROVIDER_ID && !provider.configured) {
    return 'Set API_FOOTBALL_API_KEY on the API server and restart it. Credentials never reach the browser.'
  }
  if (!provider.configured) return 'Configure this provider on the API server before using it.'
  if (!provider.available) return 'The provider is temporarily unavailable.'
  return ''
}
const selectedMatchDataProviderReason = computed(() => (
  providerUnavailableReason(selectedMatchDataProvider.value)
))
const boundMatchProviderLabel = computed(() => (
  matchDataProviderLabel(scene.value?.payload.matchBinding?.source)
))

const timeLabel = computed(() => {
  const seconds = currentTime.value
  const minutes = Math.floor(seconds / 60)
  const remaining = seconds % 60
  return `${String(minutes).padStart(2, '0')}:${remaining.toFixed(2).padStart(5, '0')}`
})

function timelineTick(seconds: number) {
  const minutes = Math.floor(seconds / 60)
  const remaining = seconds % 60
  const decimals = seconds < 10 ? 1 : 0
  return `${String(minutes).padStart(2, '0')}:${remaining.toFixed(decimals).padStart(decimals ? 4 : 2, '0')}`
}

const boundEvent = computed(() => eventBundle.value?.event ?? null)
const sceneVideo = computed(() => scene.value?.payload.videoAsset ?? null)
const projects = computed(() => scenes.value.filter((item) => item.kind === 'video'))
const activeProjectId = computed(() => {
  if (!scene.value) return null
  return sceneVideo.value?.multiPass?.parentSceneId
    ?? sceneVideo.value?.parentSceneId
    ?? (projects.value.some((item) => item.id === scene.value?.id) ? scene.value.id : null)
})
const reconstructionStatus = computed(() => sceneVideo.value?.reconstruction?.status)
const reconstructionProcessingStatus = computed<ProcessingStatus>(() => {
  const reconstruction = sceneVideo.value?.reconstruction
  if (reconstruction?.processingStatus) return reconstruction.processingStatus
  if (reconstruction?.status === 'ready') return 'completed'
  return reconstruction?.status ?? 'completed'
})
const reconstructionQualityVerdict = computed<QualityVerdict>(() => (
  sceneVideo.value?.reconstruction?.qualityVerdict
  ?? sceneVideo.value?.reconstruction?.quality?.verdict
  ?? sceneVideo.value?.reconstruction?.qualityReport?.verdict
  ?? 'unknown'
))
const reconstructionProgress = computed(() => sceneVideo.value?.reconstruction?.progress ?? null)
const reconstructionRunning = computed(() => reconstructionStatus.value === 'queued' || reconstructionStatus.value === 'processing')
const reconstructionMutationLocked = computed(() => reconstructionLocksMutations(
  reconstructionStatus.value,
  reconstructing.value,
))
const reconstructionPhases = computed<ReconstructionPhase[]>(() => reconstructionProgress.value?.phases?.length
  ? reconstructionProgress.value.phases
  : [
      { id: 'preparing', label: 'Prepare inputs', status: 'current' },
      { id: 'calibration', label: 'Calibrate pitch', status: 'pending' },
      { id: 'detection', label: 'Detect objects', status: 'pending' },
      { id: 'tracking', label: 'Build tracks', status: 'pending' },
      { id: 'projection', label: 'Reconstruct 3D', status: 'pending' },
      { id: 'finalizing', label: 'Save result', status: 'pending' },
    ])
const multiPassAnalysis = computed(() => sceneVideo.value?.multiPass ?? null)
const internalSceneLabel = computed(() => {
  if (!scene.value || scene.value.id === activeProjectId.value) return null
  if (multiPassAnalysis.value) return 'Multi-angle reconstruction'
  if (sceneVideo.value?.selectedSegmentId) return 'Segment reconstruction'
  return 'Internal scene'
})
const segmentLayout = computed(() => sceneVideo.value?.segmentLayout ?? null)
const layoutGroupOptions = computed(() => {
  const maximum = Math.max(
    1,
    ...(sceneVideo.value?.segments ?? []).map((segment) => segment.layout?.group ?? 1),
  )
  return Array.from({ length: maximum + 1 }, (_, index) => index + 1)
})
const canSplitSelection = computed(() => {
  const ordered = [...(sceneVideo.value?.segments ?? [])].sort((left, right) => left.start - right.start)
  const selected = ordered.filter((segment) => multiPassSelection.value.includes(segment.id))
  if (!selected.length) return false
  const group = selected[0].layout?.group
  if (!group || selected.some((segment) => segment.layout?.group !== group)) return false
  const groupSegments = ordered.filter((segment) => segment.layout?.group === group)
  const firstSelectedIndex = groupSegments.findIndex((segment) => selected[0].id === segment.id)
  if (firstSelectedIndex <= 0) return false
  return groupSegments.slice(firstSelectedIndex).every((segment) => multiPassSelection.value.includes(segment.id))
    && selected.length === groupSegments.length - firstSelectedIndex
})
const activePass = computed(() => {
  const analysis = multiPassAnalysis.value
  if (!analysis) return null
  return analysis.passes.find((item) => item.sceneId === activePassSceneId.value)
    ?? analysis.passes.find((item) => item.sceneId === analysis.referenceSceneId)
    ?? analysis.passes[0]
    ?? null
})
const sourceStart = computed(() => activePass.value?.sourceStart ?? sceneVideo.value?.sourceStart ?? 0)
const sourceEnd = computed(() => activePass.value?.sourceEnd ?? sceneVideo.value?.sourceEnd ?? scene.value?.duration ?? 0)
const pitchCalibration = computed(() => sceneVideo.value?.reconstruction?.pitchCalibration)
const calibrationEvidence = computed(() => sceneVideo.value?.reconstruction?.calibration ?? null)
const calibrationFrames = computed<CalibrationFrameEvidence[]>(() => (
  calibrationEvidence.value?.frameEvidence
  ?? sceneVideo.value?.reconstruction?.calibrationFrames
  ?? []
))
const videoPathUsesReferenceCamera = computed(() => (
  !multiPassAnalysis.value
  || activePass.value?.sceneId === multiPassAnalysis.value.referenceSceneId
))
const videoPathProjectionContext = computed(() => (
  videoPathUsesReferenceCamera.value
    ? resolvePathProjectionContext(calibrationFrames.value, currentTime.value)
    : null
))
const videoPathUnavailableReason = computed<string | null>(() => {
  const subject = selectedPathSubject.value
  if (!subject || !selectedPathSegments.value.length) return null
  if (activeTab.value === 'qa') {
    return 'Hidden while calibration QA is open'
  }
  if (!videoPathUsesReferenceCamera.value) {
    return 'Unavailable for this replay angle · switch to the reference camera'
  }
  if (!videoPathProjectionContext.value) {
    return 'No trusted calibration for this frame · calibrate or move the playhead'
  }
  if (!pathHasVisibleProjection(videoPathProjectionContext.value, selectedPathSegments.value)) {
    return 'Path is outside the current camera view'
  }
  return null
})
const videoPathSurfaceNote = computed<string | null>(() => {
  const context = videoPathProjectionContext.value
  if (!context || videoPathUnavailableReason.value) return null
  const notes: string[] = []
  if (context.mode === 'interpolated') {
    notes.push(`Bounded camera interpolation · ${(context.interpolationIntervalSeconds * 1000).toFixed(0)} ms`)
  } else if (context.mode === 'nearest') {
    notes.push(`Nearest camera sample · Δ ${(context.timeOffsetSeconds * 1000).toFixed(0)} ms`)
  }
  if (context.uncertaintyMetres !== null) {
    notes.push(`camera uncertainty ±${context.uncertaintyMetres.toFixed(1)} m`)
  }
  if (selectedPathSubject.value?.kind === 'ball') {
    notes.push('ground projection on video · height remains in 3D')
  }
  return notes.length ? notes.join(' · ') : null
})
const visiblePitchSide = computed<'left' | 'right' | 'unknown'>(() => {
  const explicit = sceneVideo.value?.reconstruction?.pitchOrientation?.visiblePitchSide
  if (explicit) return explicit
  const calibrated = pitchCalibration.value?.pitchSide
  if (calibrated) return calibrated
  const landmark = pitchCalibration.value?.preset ?? pitchCalibration.value?.rectangle
  if (landmark?.endsWith('-left')) return 'left'
  if (landmark?.endsWith('-right')) return 'right'
  return 'unknown'
})
const visiblePitchSideSource = computed(() => (
  sceneVideo.value?.reconstruction?.pitchOrientation?.visiblePitchSideSource
  ?? (visiblePitchSide.value === 'unknown' ? 'unknown' : 'calibration')
))
const attackingGoalSide = computed<'left' | 'right' | 'unknown'>(() => {
  const explicit = sceneVideo.value?.reconstruction?.pitchOrientation?.attackingGoal
  return explicit ?? 'unknown'
})
const activeCalibrationQaFrame = computed<CalibrationFrameEvidence | null>(() => {
  if (activeTab.value !== 'qa' || !calibrationFrames.value.length) return null
  return calibrationFrames.value.reduce((nearest, frame) => (
    Math.abs(frame.sceneTime - currentTime.value) < Math.abs(nearest.sceneTime - currentTime.value)
      ? frame
      : nearest
  ))
})
const calibrationQaFrameSize = computed(() => ({
  width: activeCalibrationQaFrame.value?.frameWidth ?? sourceVideo.value?.videoWidth ?? 960,
  height: activeCalibrationQaFrame.value?.frameHeight ?? sourceVideo.value?.videoHeight ?? 540,
}))
const calibrationQaMarkings = computed(() => {
  const frame = activeCalibrationQaFrame.value
  if (!frame) return []
  if (frame.markings?.length) return frame.markings
  return projectPitchMarkings(frame.imageToPitch, calibrationQaFrameSize.value.width, calibrationQaFrameSize.value.height)
})
const modelComparison = computed(() => sceneVideo.value?.reconstruction?.modelComparison ?? null)
const analysisFrameCount = computed(() => {
  const reconstructed = sceneVideo.value?.reconstruction?.frameCount
  if (reconstructed !== undefined) return reconstructed
  if (sceneVideo.value?.selectedSegmentId) {
    const fps = Math.min(sceneVideo.value.analysisFps ?? 10, 5)
    return Math.max(1, Math.ceil((scene.value?.duration ?? 0) * fps))
  }
  return sceneVideo.value?.frameCount ?? 0
})
const activeFrameAnalysis = computed(() => {
  const analysis = frameAnalysis.value
  return analysis && Math.abs(currentTime.value - analysis.sceneTime) <= 0.11 ? analysis : null
})
function canonicalPersonById(canonicalPersonId: string | null | undefined) {
  if (!canonicalPersonId) return null
  return scene.value?.payload.canonicalPeople?.find(
    (person) => person.canonicalPersonId === canonicalPersonId,
  ) ?? null
}

function canonicalHasActiveDedicatedUnbind(canonicalPersonId: string) {
  const currentScene = scene.value
  if (!currentScene) return false
  const tracks = currentScene.payload.tracks.filter(
    (track) => track.canonicalPersonId === canonicalPersonId,
  )
  return hasActiveDedicatedUnbindForOwner(
    currentScene.payload.videoAsset?.reconstruction?.frameAnnotations ?? [],
    [canonicalPersonId, ...tracks.map((track) => track.id)],
    {
      canonicalPeople: currentScene.payload.canonicalPeople,
      tracks: currentScene.payload.tracks,
    },
  )
}

function renderTrackForCanonicalPerson(canonicalPersonId: string | null | undefined) {
  if (!canonicalPersonId) return null
  return scene.value?.payload.tracks.find(
    (track) => track.canonicalPersonId === canonicalPersonId,
  ) ?? null
}

function framePersonCanonicalId(person: FrameAnalysis['people'][number]) {
  const matchedTrackId = validMatchedTrackId(person)
  return person.canonicalPersonId
    ?? scene.value?.payload.tracks.find((track) => track.id === matchedTrackId)?.canonicalPersonId
    ?? null
}

function framePersonLabel(person: FrameAnalysis['people'][number]) {
  const identity = canonicalPersonById(framePersonCanonicalId(person))
  return person.annotationLabel
    || person.displayName
    || identity?.displayName
    || person.matchedTrackLabel
    || person.id
}

function framePersonSelectionDescription(person: FrameAnalysis['people'][number]) {
  const canonicalPersonId = framePersonCanonicalId(person)
  if (canonicalPersonId && renderTrackForCanonicalPerson(canonicalPersonId)) {
    return 'select linked video and 3D player'
  }
  if (canonicalPersonId) return 'select canonical person; not projected in 3D'
  return 'select unresolved video detection'
}

const selectedFramePerson = computed(() => (
  activeFrameAnalysis.value?.people.find((person) => person.id === selectedFramePersonId.value) ?? null
))
const selectedTrackPresence = computed(() => (
  selectedTrack.value ? trackPresenceAtTime(selectedTrack.value, currentTime.value) : null
))
const videoSelectionStatus = computed(() => {
  const person = selectedFramePerson.value
  const canonicalPersonId = person ? framePersonCanonicalId(person) : selectedCanonicalPersonId.value
  const identity = canonicalPersonById(canonicalPersonId)
  const linkedTrack = person
    ? renderTrackForFramePerson(person, scene.value?.payload.tracks ?? [])
    : renderTrackForCanonicalPerson(canonicalPersonId) ?? selectedTrack.value
  if (identity && !linkedTrack) {
    const visibleMatches = selectedFramePeople(
      activeFrameAnalysis.value,
      null,
      identity.canonicalPersonId,
    ).length
    return {
      state: 'identity-only' as const,
      label: visibleMatches ? 'Identity matched' : 'Canonical identity selected',
      detail: `${identity.displayName || identity.canonicalPersonId} · not projected in 3D`,
      matchCount: visibleMatches,
    }
  }
  if (person && !identity && !linkedTrack) {
    return {
      state: 'unlinked' as const,
      label: 'Video detection selected',
      detail: `${framePersonLabel(person)} · identity is not resolved yet`,
      matchCount: 0,
    }
  }
  if (person && linkedTrack) {
    const metricStatus = linkedFrameMetricSelectionStatus(
      person,
      linkedTrack.label ?? person.matchedTrackLabel,
    )
    if (metricStatus) return metricStatus
  }
  return videoTrackSelectionStatus(
    activeFrameAnalysis.value,
    selectedTrackId.value,
    selectedTrack.value?.label ?? selectedCanonicalPerson.value?.displayName ?? null,
    {
      analyzing: frameAnalyzing.value,
      observedAtCurrentTime: selectedTrackPresence.value?.observed ?? null,
      canonicalPersonId: selectedCanonicalPersonId.value,
    },
  )
})
const activeCalibrationDraft = computed(() => {
  const draft = calibrationDraft.value
  return draft && Math.abs(currentTime.value - draft.sceneTime) <= 0.11 ? draft : null
})
const activeCalibrationDiagnostics = computed(() => {
  const draft = activeCalibrationDraft.value
  return draft ? buildCalibrationFrameDiagnostics(draft, calibrationFrames.value) : null
})
const visibleCalibrationWarnings = computed(() => (
  calibrationDraft.value ? calibrationPreviewWarnings(calibrationDraft.value) : []
))
const videoReviewTransformStyle = computed(() => ({
  transform: `translate3d(${videoReviewTransform.value.x}px, ${videoReviewTransform.value.y}px, 0) scale(${videoReviewTransform.value.scale})`,
}))
const videoReviewZoomPercent = computed(() => Math.round(videoReviewTransform.value.scale * 100))
const calibrationPresets: Array<{ value: PitchCalibrationPreset; label: string }> = [
  { value: 'penalty-area-left', label: 'Left penalty area' },
  { value: 'goal-area-left', label: 'Left goal area' },
  { value: 'center-circle', label: 'Center circle' },
  { value: 'goal-area-right', label: 'Right goal area' },
  { value: 'penalty-area-right', label: 'Right penalty area' },
]
const reconstructionModels: Array<{ value: ReconstructionModel; label: string }> = [
  { value: 'yolo26n.pt', label: '26n · fast' },
  { value: 'yolo26s.pt', label: '26s' },
  { value: 'yolo26m.pt', label: '26m · balanced' },
  { value: 'yolo26l.pt', label: '26l' },
  { value: 'yolo26x.pt', label: '26x · max' },
]
const ballDetectionBackends: Array<{ value: BallDetectionBackend; label: string }> = [
  { value: 'dedicated-ultralytics', label: 'Roboflow · tiled' },
  { value: 'wasb-service', label: 'WASB · temporal' },
  { value: 'generic-ultralytics', label: 'COCO · fallback' },
]
const frameAnnotationKinds: Array<{ value: FrameAnnotationKind; label: string }> = [
  { value: 'home-player', label: 'Home player' },
  { value: 'away-player', label: 'Away player' },
  { value: 'home-goalkeeper', label: 'Home goalkeeper' },
  { value: 'away-goalkeeper', label: 'Away goalkeeper' },
  { value: 'referee', label: 'Referee' },
  { value: 'other', label: 'Other person' },
]
const frameIdentityActions: Array<{ value: FrameIdentityAction; label: string }> = [
  { value: 'confirm', label: 'Confirm in tracking' },
  { value: 'exclude', label: 'Exclude detection' },
  { value: 'merge', label: 'Merge with identity' },
  { value: 'split', label: 'Split identity here / range' },
]
const frameIdentityMergeTargets = computed(() => {
  const draft = frameAnnotationDraft.value
  if (!scene.value || !draft) return []
  const annotations = scene.value.payload.videoAsset?.reconstruction?.frameAnnotations ?? []
  const publishedOwnership = {
    canonicalPeople: scene.value.payload.canonicalPeople,
    tracks: scene.value.payload.tracks,
  }
  const sourceTrack = scene.value.payload.tracks.find((track) => track.id === draft.sourceTrackId)
  const sourceOwnerIds = [
    draft.canonicalPersonId,
    draft.sourceTrackId,
    sourceTrack?.canonicalPersonId,
  ]
  const sourceExternalPlayerId = draft.externalPlayerId
    ?? canonicalPersonById(draft.canonicalPersonId)?.externalPlayerId
    ?? sourceTrack?.externalPlayerId
    ?? null
  const canonicalTargets = (scene.value.payload.canonicalPeople ?? [])
    .filter((person) => (
      person.identityStatus !== 'excluded'
      && person.canonicalPersonId !== draft.canonicalPersonId
      && dedicatedRosterMergeCompatible(
        annotations,
        sourceOwnerIds,
        [person.canonicalPersonId],
        publishedOwnership,
      )
      && !confirmedRosterBindingsConflict(
        sourceExternalPlayerId,
        person.externalPlayerId,
      )
    ))
    .map((person) => ({
      id: person.canonicalPersonId,
      label: person.displayName || person.canonicalPersonId,
      type: 'canonical' as const,
    }))
  const canonicalIds = new Set(canonicalTargets.map((target) => target.id))
  const legacyTargets = buildIdentityMergeTargets(
    scene.value.payload.tracks,
    annotations,
    draft.annotationId,
    draft.sourceTrackId,
  ).filter((target) => {
    if (target.type === 'track') {
      const track = scene.value?.payload.tracks.find((item) => item.id === target.id)
      return (
        (!track?.canonicalPersonId || !canonicalIds.has(track.canonicalPersonId))
        && dedicatedRosterMergeCompatible(
          annotations,
          sourceOwnerIds,
          [track?.id, track?.canonicalPersonId],
          publishedOwnership,
        )
        && !confirmedRosterBindingsConflict(
          sourceExternalPlayerId,
          track?.externalPlayerId,
        )
      )
    }
    const annotation = annotations.find((item) => item.id === target.id)
    const annotationTrack = scene.value?.payload.tracks.find(
      (track) => track.id === annotation?.sourceTrackId,
    )
    return dedicatedRosterMergeCompatible(
      annotations,
      sourceOwnerIds,
      [
        annotation?.id,
        annotation?.canonicalPersonId,
        annotation?.sourceTrackId,
        annotationTrack?.canonicalPersonId,
      ],
      publishedOwnership,
    ) && !confirmedRosterBindingsConflict(
      sourceExternalPlayerId,
      annotation?.externalPlayerId,
    )
  })
  return [...canonicalTargets, ...legacyTargets]
})
const frameIdentitySaveDisabled = computed(() => {
  const draft = frameAnnotationDraft.value
  const splitRangeInvalid = draft?.action === 'split' && !identitySplitRangeIsValid({
    duration: scene.value?.duration ?? 0,
    canonicalPersonId: draft.canonicalPersonId,
    targetObservationId: draft.targetObservationId,
    rangeStart: draft.rangeStart,
    rangeEnd: draft.rangeEnd,
    targetTime: activeFrameAnalysis.value?.sceneTime,
  })
  return !draft
    || splitRangeInvalid
    || (draft.action === 'merge' && !frameIdentityMergeTargets.value.some(
      (target) => target.id === draft.mergeTargetId,
    ))
    || (draft.action === 'exclude' && draft.scope === 'identity' && !draft.canonicalPersonId && !draft.sourceTrackId)
    || splitRangeInvalid
    || (draft.action !== 'exclude' && draft.kind === 'ignore')
})
const frameIdentitySplitPreview = computed(() => {
  const draft = frameAnnotationDraft.value
  if (
    draft?.action !== 'split'
    || draft.rangeStart === null
    || draft.rangeEnd === null
  ) return null
  const identity = canonicalPersonById(draft.canonicalPersonId)
  const observations = identity?.observations ?? []
  const target = observations.find((observation) => (
    (observation.observationId ?? observation.id) === draft.targetObservationId
  ))
  const counts = identitySplitObservationCounts(
    observations,
    draft.rangeStart,
    draft.rangeEnd,
    draft.affectedPreview,
  )
  return {
    identityLabel: identity?.displayName || draft.canonicalPersonId || 'Unknown identity',
    targetTime: target?.sceneTime ?? activeFrameAnalysis.value?.sceneTime ?? null,
    affected: counts.affected,
    remaining: counts.remaining,
  }
})
const calibrationLabel = computed(() => {
  const calibration = pitchCalibration.value
  if (reconstructionQualityVerdict.value === 'pending') return 'QUALITY PENDING'
  if (reconstructionQualityVerdict.value === 'reject') return 'QUALITY REJECTED'
  if (reconstructionQualityVerdict.value === 'review') return 'METRIC · REVIEW'
  if (reconstructionQualityVerdict.value === 'pass') return 'METRIC · PASS'
  return calibration?.status === 'ready'
    ? `LEGACY METRIC? ${Math.round((calibration.confidence ?? 0) * 100)}%`
    : calibration?.status === 'approximate'
      ? `HALF-PITCH ${Math.round((calibration.confidence ?? 0) * 100)}%`
    : '2.5D FALLBACK'
})

function formatAnalysisDuration(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return 'estimating…'
  const rounded = Math.max(0, Math.ceil(seconds))
  if (rounded < 60) return `${rounded}s`
  const minutes = Math.floor(rounded / 60)
  const remainder = rounded % 60
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`
}

function progressStatusLabel(status: 'completed' | 'current' | 'pending') {
  if (status === 'completed') return 'COMPLETED'
  if (status === 'current') return 'CURRENT'
  return 'PENDING'
}

function projectedEvidencePoint(point: CalibrationEvidencePoint) {
  return point.projected ?? point.projectedImage ?? null
}

function calibrationPointResidual(point: CalibrationEvidencePoint) {
  return point.residualVector?.magnitude ?? point.residualPx ?? null
}

function calibrationRawLinePoints(line: CalibrationEvidenceLine) {
  if (line.points && line.points.length >= 2) return line.points
  return line.start && line.end ? [line.start, line.end] : []
}

function calibrationRawLineLabel(line: CalibrationEvidenceLine, index: number) {
  return line.label ?? line.name ?? line.family ?? `Semantic line ${index + 1}`
}

function calibrationPercent(value: number | null | undefined) {
  return value === null || value === undefined ? '—' : `${Math.round(value * 100)}%`
}

function calibrationPixels(value: number | null | undefined) {
  return value === null || value === undefined ? '—' : `${value.toFixed(1)} px`
}

function calibrationVisibleSideLabel(
  side: 'left' | 'right' | 'unknown' | null | undefined,
  trusted: boolean | null | undefined,
) {
  if (!side || side === 'unknown') return 'UNKNOWN'
  return trusted ? side.toUpperCase() : `CANDIDATE ${side.toUpperCase()} · UNTRUSTED`
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

function canonicalToPassTime(time: number) {
  const pass = activePass.value
  const duration = Math.max(0.01, scene.value?.duration ?? 0.01)
  const passDuration = Math.max(0.01, sourceEnd.value - sourceStart.value)
  if (!pass || pass.sceneId === multiPassAnalysis.value?.referenceSceneId) return Math.min(time, passDuration)
  const alignment = pass.alignment
  if (alignment?.overlap && alignment.anchors.length > 1) {
    return interpolateMapping(time, alignment.anchors, 'referenceTime', 'passTime')
  }
  return Math.min(passDuration, Math.max(0, time / duration * passDuration))
}

function passToCanonicalTime(time: number) {
  const pass = activePass.value
  const duration = Math.max(0.01, scene.value?.duration ?? 0.01)
  const passDuration = Math.max(0.01, sourceEnd.value - sourceStart.value)
  if (!pass || pass.sceneId === multiPassAnalysis.value?.referenceSceneId) return Math.min(time, duration)
  const alignment = pass.alignment
  if (alignment?.overlap && alignment.anchors.length > 1) {
    return interpolateMapping(time, alignment.anchors, 'passTime', 'referenceTime')
  }
  return Math.min(duration, Math.max(0, time / passDuration * duration))
}

function passRelationLabel(relation?: string) {
  if (relation === 'reference') return 'reference'
  if (relation === 'replay-overlap') return 'aligned replay'
  if (relation === 'continuation-before') return 'earlier context'
  if (relation === 'continuation-after') return 'later context'
  return 'independent'
}

function segmentRoleLabel(role?: string) {
  if (role === 'original') return 'Original'
  if (role === 'replay') return 'Replay'
  return 'Continuation'
}

function segmentGroupColor(group = 1) {
  return ['#ffd36a', '#71e2aa', '#76a9ff', '#dc89ff', '#ff8b6b', '#68d9d4'][(group - 1) % 6]
}

function alphabeticVariant(index: number) {
  let value = index
  let output = ''
  while (value >= 0) {
    output = String.fromCharCode(65 + value % 26) + output
    value = Math.floor(value / 26) - 1
  }
  return output
}

function normalizeSegmentLayout(compact = false) {
  const video = sceneVideo.value
  if (!video?.segments?.length || !video.segmentLayout) return
  const ordered = [...video.segments].sort((left, right) => left.start - right.start)
  const groupOrder = [...new Set(ordered.map((segment) => segment.layout?.group ?? 1))]
  const groupMap = new Map(groupOrder.map((group, index) => [group, compact ? index + 1 : group]))
  const grouped = new Map<number, VideoSegment[]>()
  for (const segment of ordered) {
    const group = groupMap.get(segment.layout?.group ?? 1) ?? 1
    const items = grouped.get(group) ?? []
    items.push(segment)
    grouped.set(group, items)
  }
  video.segmentLayout.groups = [...grouped.entries()].map(([group, items]) => {
    items.forEach((segment, index) => {
      const variant = alphabeticVariant(index)
      const currentRole = segment.layout?.role ?? (index === 0 ? 'original' : 'continuation')
      const role = index === 0 ? 'original' : currentRole === 'original' ? 'continuation' : currentRole
      segment.layout = {
        group,
        variant,
        label: `${group}-${variant}`,
        role,
        confidence: segment.layout?.confidence ?? 1,
        motionCost: segment.layout?.motionCost,
      }
    })
    return {
      id: `event-${group}`,
      index: group,
      label: String(group),
      segmentIds: items.map((item) => item.id),
      replayCount: items.filter((item) => item.layout?.role === 'replay').length,
    }
  })
}

function assignSegmentGroup(segment: VideoSegment, value: string) {
  const group = Number(value)
  if (!Number.isFinite(group) || group < 1) return
  segment.layout = {
    group,
    variant: segment.layout?.variant ?? 'A',
    label: segment.layout?.label ?? `${group}-A`,
    role: segment.layout?.role ?? 'continuation',
    confidence: 1,
    motionCost: segment.layout?.motionCost,
  }
  if (segmentLayout.value) segmentLayout.value.status = 'edited'
  normalizeSegmentLayout()
  saveState.value = 'Timeline layout has unsaved changes'
}

function assignSegmentRole(segment: VideoSegment, value: string) {
  if (!segment.layout || !['original', 'replay', 'continuation'].includes(value)) return
  segment.layout.role = value as 'original' | 'replay' | 'continuation'
  segment.layout.confidence = 1
  if (segmentLayout.value) segmentLayout.value.status = 'edited'
  normalizeSegmentLayout()
  saveState.value = 'Timeline layout has unsaved changes'
}

function splitSelectedIntoNewEvent() {
  const video = sceneVideo.value
  if (!video?.segments?.length || !segmentLayout.value || !canSplitSelection.value) {
    error.value = 'Select a continuous group tail, leaving its first segment in the original event.'
    return
  }
  const selectedIds = new Set(multiPassSelection.value)
  const selected = video.segments.filter((segment) => selectedIds.has(segment.id))
  const sourceGroup = selected[0].layout?.group
  if (!sourceGroup) return
  for (const segment of video.segments) {
    if (!segment.layout) continue
    if (segment.layout.group > sourceGroup) segment.layout.group += 1
  }
  for (const segment of selected) {
    if (!segment.layout) continue
    segment.layout.group = sourceGroup + 1
    segment.layout.confidence = 1
  }
  segmentLayout.value.status = 'edited'
  normalizeSegmentLayout()
  multiPassSelection.value = []
  saveState.value = `Created Event ${sourceGroup + 1}; later events shifted`
}

function toggleTimelineGroupEditing() {
  multiPassSelection.value = []
  timelineGroupEditing.value = !timelineGroupEditing.value
}

function handleTimelineSegment(segment: VideoSegment) {
  if (!timelineGroupEditing.value) {
    seekTo(segment.start)
    return
  }
  if (multiPassSelection.value.includes(segment.id)) {
    multiPassSelection.value = multiPassSelection.value.filter((id) => id !== segment.id)
  } else if (multiPassSelection.value.length < 6) {
    multiPassSelection.value = [...multiPassSelection.value, segment.id]
  }
}

async function saveTimelineGroupMap() {
  await confirmSegmentLayout()
  timelineGroupEditing.value = false
  multiPassSelection.value = []
}

async function confirmSegmentLayout() {
  if (!segmentLayout.value) return
  normalizeSegmentLayout(true)
  segmentLayout.value.status = 'confirmed'
  await saveScene()
}

async function rebuildSegmentLayout() {
  if (!sceneVideo.value || layoutRebuilding.value) return
  layoutRebuilding.value = true
  saveState.value = 'Rebuilding event map…'
  try {
    scene.value = await api.proposeSegmentLayout(sceneVideo.value.id)
    selectedTrackId.value = null
    selectedCanonicalPersonId.value = null
    currentTime.value = 0
    saveState.value = 'New event map proposed'
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not rebuild the event map'
  } finally {
    layoutRebuilding.value = false
  }
}

function chooseSourcePass(event: Event) {
  activePassSceneId.value = (event.target as HTMLSelectElement).value
  seekTo(currentTime.value)
}

function tick(timestamp: number) {
  if (!previousTime) previousTime = timestamp
  const delta = Math.min(0.05, (timestamp - previousTime) / 1000)
  previousTime = timestamp
  if (playing.value && scene.value) {
    if (sceneVideo.value && sourceVideo.value) {
      sourceVideo.value.playbackRate = playbackRate.value
      currentTime.value = Math.max(0, passToCanonicalTime(sourceVideo.value.currentTime - sourceStart.value))
    } else {
      currentTime.value += delta * playbackRate.value
    }
    if (currentTime.value >= scene.value.duration || (sourceVideo.value && sourceVideo.value.currentTime >= sourceEnd.value) || sourceVideo.value?.ended) {
      currentTime.value = 0
      playing.value = false
      sourceVideo.value?.pause()
      if (sourceVideo.value) sourceVideo.value.currentTime = sourceStart.value
    }
  }
  animationFrame = requestAnimationFrame(tick)
}

async function loadWorkspace() {
  error.value = null
  try {
    scenes.value = await api.listScenes()
    if (!scenes.value.length) throw new Error('No scenes found')
    const initialSummary = scenes.value.find((item) => item.kind === 'video') ?? scenes.value[0]
    scene.value = await api.getScene(initialSummary.id)
    selectedTrackId.value = scene.value.payload.tracks[0]?.id ?? null
    void loadIdentityReview(scene.value.id)
    resumeReconstructionPolling()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not open the workspace'
  }
}

async function loadIdentityReview(sceneId: string) {
  const requestId = ++identityReviewRequestId
  const reconstruction = scene.value?.id === sceneId
    ? scene.value.payload.videoAsset?.reconstruction
    : null
  if (reconstruction?.status === 'queued' || reconstruction?.status === 'processing') {
    identityReviewSnapshot.value = null
    identityReviewLoading.value = false
    identityReviewError.value = null
    return
  }
  identityReviewLoading.value = true
  identityReviewError.value = null
  try {
    const review = await api.identityReview(sceneId)
    if (requestId !== identityReviewRequestId || scene.value?.id !== sceneId) return
    if (scene.value.revision !== undefined && review.revision !== scene.value.revision) {
      identityReviewSnapshot.value = null
      identityReviewError.value = 'Identity review changed with the scene; reload the review.'
      return
    }
    identityReviewSnapshot.value = review
  } catch (cause) {
    if (requestId !== identityReviewRequestId || scene.value?.id !== sceneId) return
    identityReviewSnapshot.value = null
    identityReviewError.value = cause instanceof Error
      ? cause.message
      : 'Could not load identity review evidence'
  } finally {
    if (requestId === identityReviewRequestId) identityReviewLoading.value = false
  }
}

function invalidateIdentityReview() {
  identityReviewRequestId += 1
  identityReviewSnapshot.value = null
  identityReviewLoading.value = false
  identityReviewError.value = null
}

async function switchScene(id: string) {
  window.clearTimeout(reconstructionTimer)
  reconstructing.value = false
  playing.value = false
  sourceVideo.value?.pause()
  currentTime.value = 0
  invalidateIdentityReview()
  scene.value = await api.getScene(id)
  selectedTrackId.value = scene.value.payload.tracks[0]?.id ?? null
  void loadIdentityReview(scene.value.id)
  resumeReconstructionPolling()
}

async function saveScene() {
  if (
    !scene.value
    || reconstructing.value
    || reconstructionRunning.value
    || frameAnnotationSaving.value
    || rosterBindingSaving.value
    || identityDecisionSaving.value
    || matchSnapshotRefreshing.value
    || manualRosterImporting.value
    || playerActionSaving.value
  ) return
  saving.value = true
  saveState.value = 'Saving…'
  try {
    scene.value = await api.saveScene(scene.value)
    saveState.value = 'All changes saved'
  } catch (cause) {
    saveState.value = cause instanceof Error ? cause.message : 'Save failed'
  } finally {
    saving.value = false
  }
}

function togglePlay() {
  if (!scene.value) return
  if (currentTime.value >= scene.value.duration) currentTime.value = 0
  playing.value = !playing.value
  if (sceneVideo.value && sourceVideo.value) {
    sourceVideo.value.currentTime = sourceStart.value + canonicalToPassTime(currentTime.value)
    sourceVideo.value.playbackRate = playbackRate.value
    if (playing.value) void sourceVideo.value.play()
    else sourceVideo.value.pause()
  }
}

function seekTo(time: number) {
  currentTime.value = Math.max(0, Math.min(scene.value?.duration ?? time, time))
  if (sourceVideo.value) sourceVideo.value.currentTime = sourceStart.value + canonicalToPassTime(currentTime.value)
}

function onTimelineInput() {
  playing.value = false
  sourceVideo.value?.pause()
  seekTo(currentTime.value)
}

function roundActionTime(time: number) {
  return Number(time.toFixed(3))
}

function createPlayerActionId() {
  const uuid = globalThis.crypto?.randomUUID?.()
  return `action-${uuid ?? `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`}`
}

function buildManualPlayerAction(
  canonicalPersonId: string,
  playheadTime: number,
  type: PlayerActionType = 'pass',
): PlayerAction | null {
  const sceneDuration = scene.value?.duration ?? 0
  if (sceneDuration <= 0.001) return null
  const intervalDuration = Math.min(defaultPlayerActionDuration(type), sceneDuration)
  const keypointTime = Math.max(0, Math.min(sceneDuration, playheadTime))
  let startTime = Math.max(0, keypointTime - intervalDuration * 0.42)
  let endTime = startTime + intervalDuration
  if (endTime > sceneDuration) {
    endTime = sceneDuration
    startTime = Math.max(0, endTime - intervalDuration)
  }
  startTime = roundActionTime(startTime)
  endTime = roundActionTime(endTime)
  if (endTime <= startTime) endTime = Math.min(sceneDuration, startTime + 0.001)
  if (endTime <= startTime) return null
  return {
    id: createPlayerActionId(),
    canonicalPersonId,
    type,
    startTime,
    endTime,
    keypoints: [{
      kind: defaultPlayerActionKeypointKind(type),
      time: roundActionTime(Math.max(startTime, Math.min(endTime, keypointTime))),
    }],
    confidence: 1,
    status: 'confirmed',
    source: 'manual',
  }
}

async function persistPlayerAction(action: PlayerAction, successMessage: string) {
  const activeScene = scene.value
  if (
    !activeScene
    || playerActionSaving.value
    || reconstructionMutationLocked.value
    || action.source !== 'manual'
  ) return
  const previousActions = [...(activeScene.payload.playerActions ?? [])]
  const nextAction: PlayerAction = {
    ...action,
    canonicalPersonId: action.canonicalPersonId,
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
  selectedPlayerActionId.value = nextAction.id
  playerActionSaving.value = true
  saveState.value = 'Saving player action…'
  error.value = null
  try {
    const updated = await api.upsertPlayerAction(activeScene.id, nextAction)
    if (scene.value?.id !== activeScene.id) return
    scene.value = updated
    selectedPlayerActionId.value = nextAction.id
    saveState.value = successMessage
  } catch (cause) {
    if (scene.value?.id === activeScene.id) scene.value.payload.playerActions = previousActions
    error.value = cause instanceof Error ? cause.message : 'Could not save the player action'
    saveState.value = 'Player action change was not saved'
  } finally {
    playerActionSaving.value = false
  }
}

function addPlayerActionAt(time: number) {
  const canonicalPersonId = selectedActionActorId.value
  if (!canonicalPersonId) {
    error.value = 'Select a resolved player before adding an action'
    return
  }
  const action = buildManualPlayerAction(canonicalPersonId, time)
  if (!action) {
    error.value = 'This scene is too short for an action interval'
    return
  }
  playing.value = false
  sourceVideo.value?.pause()
  selectedPlayerActionId.value = action.id
  void persistPlayerAction(action, `Added ${action.type} action at ${time.toFixed(2)}s`)
}

function selectPlayerAction(actionId: string) {
  const action = selectedActorActions.value.find((item) => item.id === actionId)
  if (!action) return
  selectedPlayerActionId.value = action.id
}

function seekPlayerAction(time: number) {
  playing.value = false
  sourceVideo.value?.pause()
  seekTo(time)
}

function updatePlayerAction(action: PlayerAction) {
  if (action.canonicalPersonId !== selectedActionActorId.value) {
    error.value = 'The action belongs to a different canonical player'
    return
  }
  void persistPlayerAction(action, `Saved ${action.type} action`)
}

async function removePlayerAction(actionId: string) {
  const activeScene = scene.value
  const action = selectedActorActions.value.find((item) => item.id === actionId)
  if (
    !activeScene
    || !action
    || action.source !== 'manual'
    || playerActionSaving.value
    || reconstructionMutationLocked.value
  ) return
  const previousActions = [...(activeScene.payload.playerActions ?? [])]
  activeScene.payload.playerActions = previousActions.filter((item) => item.id !== actionId)
  selectedPlayerActionId.value = null
  playerActionSaving.value = true
  saveState.value = 'Removing player action…'
  error.value = null
  try {
    const updated = await api.deletePlayerAction(activeScene.id, actionId)
    if (scene.value?.id !== activeScene.id) return
    scene.value = updated
    saveState.value = `Removed ${action.type} action`
  } catch (cause) {
    if (scene.value?.id === activeScene.id) scene.value.payload.playerActions = previousActions
    selectedPlayerActionId.value = actionId
    error.value = cause instanceof Error ? cause.message : 'Could not remove the player action'
    saveState.value = 'Player action removal was not saved'
  } finally {
    playerActionSaving.value = false
  }
}

function videoReviewSize() {
  const viewport = videoReviewViewport.value
  return {
    width: viewport?.clientWidth ?? 0,
    height: viewport?.clientHeight ?? 0,
  }
}

function commitVideoReviewTransform(transform: VideoReviewTransform) {
  const { width, height } = videoReviewSize()
  videoReviewTransform.value = clampVideoReviewTransform(transform, width, height)
}

function resetVideoReviewView() {
  videoReviewPanDrag.value = null
  videoReviewTransform.value = { scale: VIDEO_REVIEW_MIN_SCALE, x: 0, y: 0 }
}

function setVideoReviewZoom(nextScale: number, clientX?: number, clientY?: number) {
  const viewport = videoReviewViewport.value
  if (!viewport) return
  const rect = viewport.getBoundingClientRect()
  const focalX = clientX === undefined ? 0 : clientX - rect.left - rect.width / 2
  const focalY = clientY === undefined ? 0 : clientY - rect.top - rect.height / 2
  videoReviewTransform.value = zoomVideoReviewTransform(
    videoReviewTransform.value,
    nextScale,
    focalX,
    focalY,
    rect.width,
    rect.height,
  )
}

function adjustVideoReviewZoom(delta: number) {
  setVideoReviewZoom(videoReviewTransform.value.scale + delta)
}

function onVideoReviewWheel(event: WheelEvent) {
  if (draggedCalibrationAnchor.value || frameAnnotationDrag.value) return
  const factor = Math.exp(-event.deltaY * 0.0015)
  setVideoReviewZoom(videoReviewTransform.value.scale * factor, event.clientX, event.clientY)
}

function startVideoReviewPan(event: PointerEvent) {
  if (
    videoReviewTransform.value.scale <= VIDEO_REVIEW_MIN_SCALE
    || frameAnnotationMode.value
    || draggedCalibrationAnchor.value
    || event.button !== 0
  ) return
  const target = event.target
  if (target instanceof Element && target.closest('button, input, select, .calibration-anchor, .frame-person-box, .frame-ignore-box')) return
  videoReviewPanDrag.value = {
    pointerId: event.pointerId,
    clientX: event.clientX,
    clientY: event.clientY,
    transform: { ...videoReviewTransform.value },
  }
  try {
    videoReviewViewport.value?.setPointerCapture(event.pointerId)
  } catch {
    // Pointer capture is optional for synthetic accessibility input.
  }
  event.preventDefault()
}

function updateVideoReviewPan(event: PointerEvent) {
  const drag = videoReviewPanDrag.value
  const viewport = videoReviewViewport.value
  if (!drag || !viewport || drag.pointerId !== event.pointerId) return
  const { width, height } = videoReviewSize()
  videoReviewTransform.value = panVideoReviewTransform(
    drag.transform,
    event.clientX - drag.clientX,
    event.clientY - drag.clientY,
    width,
    height,
  )
  event.preventDefault()
}

function finishVideoReviewPan(event: PointerEvent) {
  const drag = videoReviewPanDrag.value
  if (!drag || drag.pointerId !== event.pointerId) return
  videoReviewPanDrag.value = null
  try {
    if (videoReviewViewport.value?.hasPointerCapture(event.pointerId)) {
      videoReviewViewport.value.releasePointerCapture(event.pointerId)
    }
  } catch {
    // The pointer may already have been released by the browser.
  }
}

function onVideoReviewKeydown(event: KeyboardEvent) {
  if (event.target !== event.currentTarget) return
  if (event.key === '+' || event.key === '=') {
    event.preventDefault()
    adjustVideoReviewZoom(0.25)
    return
  }
  if (event.key === '-') {
    event.preventDefault()
    adjustVideoReviewZoom(-0.25)
    return
  }
  if (event.key === '0' || event.key === 'Home') {
    event.preventDefault()
    resetVideoReviewView()
    return
  }
  const amount = event.shiftKey ? 64 : 24
  const movement = {
    ArrowLeft: [-amount, 0],
    ArrowRight: [amount, 0],
    ArrowUp: [0, -amount],
    ArrowDown: [0, amount],
  }[event.key]
  if (!movement || videoReviewTransform.value.scale <= VIDEO_REVIEW_MIN_SCALE) return
  event.preventDefault()
  const { width, height } = videoReviewSize()
  videoReviewTransform.value = panVideoReviewTransform(
    videoReviewTransform.value,
    movement[0],
    movement[1],
    width,
    height,
  )
}

function setCamera(name: CameraName) {
  activeCamera.value = name
  viewport.value?.cameraPreset(name)
}

function onCameraPresetChange(event: Event) {
  setCamera((event.target as HTMLSelectElement).value as CameraName)
}

function moveSelected(position: { x: number; z: number }) {
  if (!selectedTrack.value) return
  selectedTrack.value.keyframes = upsertKeyframe(selectedTrack.value.keyframes, {
    t: Number(currentTime.value.toFixed(2)),
    x: Number(position.x.toFixed(2)),
    z: Number(position.z.toFixed(2)),
    confidence: 1,
  })
  saveState.value = 'Unsaved changes'
}

function normalizeManualBallKeyframes(keyframes: Keyframe[]) {
  const duration = scene.value?.duration ?? 0
  const length = scene.value?.payload.pitch.length ?? 105
  const width = scene.value?.payload.pitch.width ?? 68
  const normalized: Keyframe[] = []
  for (const frame of [...keyframes].sort((left, right) => left.t - right.t)) {
    const next: Keyframe = {
      ...frame,
      t: Number(Math.max(0, Math.min(duration, frame.t)).toFixed(3)),
      x: Number(Math.max(-length / 2, Math.min(length / 2, frame.x)).toFixed(2)),
      y: Number(Math.max(0.24, frame.y ?? 0.24).toFixed(2)),
      z: Number(Math.max(-width / 2, Math.min(width / 2, frame.z)).toFixed(2)),
      confidence: 1,
      observed: true,
      state: 'observed',
      projectionSource: 'manual',
      projection: { source: 'manual', uncertaintyMetres: 0 },
      positionUncertaintyMetres: 0,
    }
    const duplicate = normalized.findIndex((item) => Math.abs(item.t - next.t) < MANUAL_BALL_TIME_TOLERANCE)
    if (duplicate >= 0) normalized[duplicate] = next
    else normalized.push(next)
  }
  return normalized.sort((left, right) => left.t - right.t)
}

function manualBallKeyframeAt(time: number, position?: { x: number; z: number }): Keyframe {
  const source = manualBallKeyframes.value.length
    ? manualBallKeyframes.value
    : automaticBallKeyframes.value
  const interpolated = interpolateKeyframes(source, time)
  return normalizeManualBallKeyframes([{
    t: time,
    x: position?.x ?? interpolated.x,
    y: position ? 0.24 : interpolated.y ?? 0.24,
    z: position?.z ?? interpolated.z,
    confidence: 1,
  }])[0]
}

async function persistManualBallKeyframes(
  keyframes: Keyframe[],
  message: string,
  selectionTime: number | null,
) {
  if (!scene.value || ballTrajectorySaving.value) return
  const activeScene = scene.value
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
  const normalized = normalizeManualBallKeyframes(keyframes)
  activeScene.payload.ball = {
    ...activeScene.payload.ball,
    mode: 'manual',
    manualKeyframes: normalized,
    keyframes: normalized,
  }
  selectedBallKeyframeTime.value = selectionTime
  ballTrajectorySaving.value = true
  saveState.value = 'Saving manual ball trajectory…'
  try {
    const updated = await api.updateBallTrajectory(activeScene.id, 'manual', normalized)
    if (scene.value?.id !== activeScene.id) return
    scene.value = updated
    if (selectionTime !== null) {
      selectedBallKeyframeTime.value = manualBallKeyframes.value.find(
        (frame) => Math.abs(frame.t - selectionTime) < 0.0011,
      )?.t ?? null
    }
    saveState.value = message
  } catch (cause) {
    if (scene.value?.id === activeScene.id) scene.value.payload.ball = previousBall
    error.value = cause instanceof Error ? cause.message : 'Could not save the manual ball trajectory'
    saveState.value = 'Manual ball change was not saved'
  } finally {
    ballTrajectorySaving.value = false
  }
}

function selectBallObject() {
  ballSelected.value = true
  selectedTrackId.value = null
  selectedCanonicalPersonId.value = null
  selectedFramePersonId.value = null
  selectedPlayerActionId.value = null
  editMode.value = false
  activeTab.value = 'binding'
  viewOptions.value.ball = true
  if (ballTrajectoryMode.value === 'manual') {
    ballEditMode.value = true
    if (viewMode.value === 'video') viewMode.value = sceneVideo.value ? 'split' : '3d'
  }
}

function toggleManualBallPlacement() {
  if (ballTrajectoryMode.value !== 'manual') {
    void setBallTrajectoryMode('manual')
    return
  }
  const next = !ballEditMode.value
  selectBallObject()
  ballEditMode.value = next
  saveState.value = next
    ? 'Click anywhere on the 3D pitch to place the ball'
    : 'Ball placement paused'
}

async function setBallTrajectoryMode(mode: BallTrajectoryMode) {
  if (!scene.value || ballTrajectorySaving.value || mode === ballTrajectoryMode.value) {
    if (mode === 'manual') selectBallObject()
    return
  }
  const sceneId = scene.value.id
  ballTrajectorySaving.value = true
  saveState.value = mode === 'manual'
    ? 'Opening manual ball trajectory…'
    : 'Restoring automatic ball trajectory…'
  try {
    const updated = await api.updateBallTrajectory(sceneId, mode)
    if (scene.value?.id !== sceneId) return
    scene.value = updated
    ballSelected.value = true
    selectedTrackId.value = null
    selectedCanonicalPersonId.value = null
    selectedFramePersonId.value = null
    editMode.value = false
    activeTab.value = 'binding'
    viewOptions.value.ball = true
    ballEditMode.value = mode === 'manual'
    selectedBallKeyframeTime.value = null
    if (mode === 'manual' && viewMode.value === 'video') viewMode.value = sceneVideo.value ? 'split' : '3d'
    saveState.value = mode === 'manual'
      ? 'Manual ball mode · add a keypoint or click the pitch'
      : 'Automatic ball trajectory restored'
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not switch the ball trajectory mode'
  } finally {
    ballTrajectorySaving.value = false
  }
}

function changeBallTrajectoryMode(event: Event) {
  void setBallTrajectoryMode((event.target as HTMLSelectElement).value as BallTrajectoryMode)
}

function selectManualBallKeypoint(time: number) {
  const keyframe = manualBallKeyframes.value.find((frame) => Math.abs(frame.t - time) < 0.0011)
  if (!keyframe) return
  selectBallObject()
  selectedBallKeyframeTime.value = keyframe.t
  ballEditMode.value = true
  playing.value = false
  sourceVideo.value?.pause()
  seekTo(keyframe.t)
  saveState.value = `Ball keypoint selected at ${keyframe.t.toFixed(3)}s`
}

function addManualBallKeypoint(time = currentTime.value) {
  if (ballTrajectorySaving.value) return
  const timestamp = Number(Math.max(0, Math.min(scene.value?.duration ?? time, time)).toFixed(3))
  const existing = manualBallKeyframes.value.find((frame) => Math.abs(frame.t - timestamp) < MANUAL_BALL_TIME_TOLERANCE)
  if (existing) {
    selectManualBallKeypoint(existing.t)
    return
  }
  selectBallObject()
  playing.value = false
  sourceVideo.value?.pause()
  const next = manualBallKeyframeAt(timestamp)
  void persistManualBallKeyframes(
    [...manualBallKeyframes.value, next],
    `Ball keypoint added at ${timestamp.toFixed(3)}s`,
    timestamp,
  )
}

function moveManualBall(position: { x: number; z: number }) {
  if (ballTrajectoryMode.value !== 'manual' || ballTrajectorySaving.value) return
  const timestamp = selectedBallKeyframeTime.value ?? Number(currentTime.value.toFixed(3))
  const current = manualBallKeyframes.value.find((frame) => Math.abs(frame.t - timestamp) < MANUAL_BALL_TIME_TOLERANCE)
    ?? manualBallKeyframeAt(timestamp, position)
  const moved = manualBallKeyframeAt(timestamp, position)
  const next = manualBallKeyframes.value.filter((frame) => Math.abs(frame.t - timestamp) >= MANUAL_BALL_TIME_TOLERANCE)
  next.push({ ...current, ...moved })
  ballSelected.value = true
  ballEditMode.value = true
  playing.value = false
  sourceVideo.value?.pause()
  void persistManualBallKeyframes(
    next,
    `Ball placed at X ${moved.x.toFixed(2)} · Z ${moved.z.toFixed(2)}`,
    moved.t,
  )
}

function removeManualBallKeypoint(time: number) {
  if (ballTrajectorySaving.value) return
  const next = manualBallKeyframes.value.filter((frame) => Math.abs(frame.t - time) >= MANUAL_BALL_TIME_TOLERANCE)
  const nearest = [...next].sort(
    (left, right) => Math.abs(left.t - time) - Math.abs(right.t - time),
  )[0] ?? null
  if (nearest) seekTo(nearest.t)
  void persistManualBallKeyframes(
    next,
    `Ball keypoint removed from ${time.toFixed(3)}s`,
    nearest?.t ?? null,
  )
}

function updateManualBallKeypointTime(payload: { from: number; to: number }) {
  if (ballTrajectorySaving.value) return
  const current = manualBallKeyframes.value.find((frame) => Math.abs(frame.t - payload.from) < MANUAL_BALL_TIME_TOLERANCE)
  if (!current) return
  const next = manualBallKeyframes.value.filter((frame) => Math.abs(frame.t - payload.from) >= MANUAL_BALL_TIME_TOLERANCE)
  const moved = manualBallKeyframeAt(payload.to, { x: current.x, z: current.z })
  next.push({ ...current, ...moved })
  seekTo(moved.t)
  void persistManualBallKeyframes(
    next,
    `Ball keypoint moved to ${moved.t.toFixed(3)}s`,
    moved.t,
  )
}

function updateManualBallCoordinate(axis: 'x' | 'z', value: string) {
  const selected = selectedManualBallKeyframe.value
  const numeric = Number(value)
  if (!selected || !Number.isFinite(numeric)) return
  moveManualBall({
    x: axis === 'x' ? numeric : selected.x,
    z: axis === 'z' ? numeric : selected.z,
  })
}

function updateTrackPosition(axis: 'x' | 'z', value: string) {
  if (!selectedTrack.value) return
  const position = interpolateKeyframes(selectedTrack.value.keyframes, currentTime.value)
  moveSelected({ x: axis === 'x' ? Number(value) : position.x, z: axis === 'z' ? Number(value) : position.z })
}

async function loadMatchDataProviders() {
  providerCatalogLoading.value = true
  try {
    const catalog = await api.matchDataProviders()
    catalogProviders.value = catalog.providers
    selectedCatalogProvider.value = resolveMatchDataProvider(
      catalog,
      scene.value?.payload.matchBinding?.source,
    )
  } catch {
    // Old API servers only expose the original TheSportsDB catalog. Keep the
    // editor usable while making the missing API-Football adapter explicit.
    catalogProviders.value = [...LEGACY_MATCH_DATA_PROVIDERS.providers]
    selectedCatalogProvider.value = resolveMatchDataProvider(
      LEGACY_MATCH_DATA_PROVIDERS,
      scene.value?.payload.matchBinding?.source,
    )
  } finally {
    providerCatalogLoading.value = false
  }
}

async function openMatchSettings() {
  catalogOpen.value = true
  catalogEvents.value = []
  catalogError.value = null
  await loadMatchDataProviders()
  if (catalogOpen.value && selectedMatchDataProviderReady.value) await loadCatalog()
}

function changeCatalogProvider() {
  catalogEvents.value = []
  catalogError.value = null
}

async function loadCatalog() {
  if (!selectedMatchDataProviderReady.value) {
    catalogEvents.value = []
    catalogError.value = selectedMatchDataProviderReason.value
    return
  }
  catalogLoading.value = true
  catalogEvents.value = []
  catalogError.value = null
  try {
    catalogEvents.value = await api.eventsByDate(
      catalogDate.value,
      selectedCatalogProvider.value,
    )
  } catch (cause) {
    catalogError.value = cause instanceof Error ? cause.message : 'Football catalog is unavailable'
  } finally {
    catalogLoading.value = false
  }
}

async function loadCatalogSearch() {
  const query = catalogQuery.value.trim()
  if (query.length < 3) return
  if (!selectedMatchDataProviderReady.value) {
    catalogEvents.value = []
    catalogError.value = selectedMatchDataProviderReason.value
    return
  }
  catalogLoading.value = true
  catalogEvents.value = []
  catalogError.value = null
  try {
    catalogEvents.value = await api.searchEvents(
      query,
      selectedCatalogProvider.value,
    )
  } catch (cause) {
    catalogError.value = cause instanceof Error ? cause.message : 'Football catalog is unavailable'
  } finally {
    catalogLoading.value = false
  }
}

async function bindMatch(event: ExternalEvent) {
  if (
    !scene.value
    || reconstructionMutationLocked.value
    || bundleLoading.value
    || manualRosterImporting.value
    || matchSnapshotRefreshing.value
  ) return
  const sceneId = scene.value.id
  bundleLoading.value = event.id
  catalogError.value = null
  try {
    const result = await api.bindSceneMatch(
      sceneId,
      event.id,
      event.provider ?? selectedCatalogProvider.value,
    )
    if (scene.value?.id !== sceneId) return
    invalidateIdentityReview()
    scene.value = result.scene
    catalogOpen.value = false
    clearStaleFrameAnalysis()
    const status = result.scene.payload.videoAsset?.reconstruction?.status
    if (status === 'queued' || status === 'processing') {
      reconstructing.value = true
      saveState.value = 'Project match data saved · rebuilding identity with the new roster…'
      void pollReconstruction(sceneId)
    } else {
      saveState.value = 'Project match data saved'
      void loadIdentityReview(sceneId)
    }
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not bind this match'
  } finally {
    bundleLoading.value = null
  }
}

async function refreshMatchSnapshot() {
  if (
    !scene.value
    || !matchSnapshotRefreshAvailable.value
    || matchSnapshotRefreshing.value
    || manualRosterImporting.value
    || reconstructionMutationLocked.value
  ) return
  const sceneId = scene.value.id
  matchSnapshotRefreshing.value = true
  error.value = null
  saveState.value = 'Refreshing project match snapshot…'
  try {
    const result = await api.refreshSceneMatchBinding(sceneId)
    if (scene.value?.id !== sceneId) return
    invalidateIdentityReview()
    scene.value = result.scene
    clearStaleFrameAnalysis()
    const status = result.scene.payload.videoAsset?.reconstruction?.status
    if (status === 'queued' || status === 'processing') {
      reconstructing.value = true
      saveState.value = 'Project match snapshot refreshed · rebuilding identity…'
      void pollReconstruction(sceneId)
    } else {
      saveState.value = 'Project match snapshot refreshed'
      void loadIdentityReview(sceneId)
    }
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not refresh match data'
  } finally {
    matchSnapshotRefreshing.value = false
  }
}

function chooseManualRosterFile() {
  manualRosterImportError.value = null
  if (manualRosterFileInput.value) manualRosterFileInput.value.value = ''
  manualRosterFileInput.value?.click()
}

async function importManualRosterFile(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file || !scene.value) return
  if (reconstructionMutationLocked.value || manualRosterImporting.value || matchSnapshotRefreshing.value) {
    manualRosterImportError.value = 'Wait for reconstruction or project match refresh to finish before importing a roster.'
    input.value = ''
    return
  }
  const sceneId = scene.value.id
  manualRosterImportError.value = null
  if (!file.name.toLowerCase().endsWith('.json')) {
    manualRosterImportError.value = 'Choose a .json roster file.'
    input.value = ''
    return
  }
  if (file.size > 2 * 1024 * 1024) {
    manualRosterImportError.value = 'The roster JSON is larger than the 2 MB import limit.'
    input.value = ''
    return
  }
  manualRosterImporting.value = true
  saveState.value = `Importing ${file.name}…`
  try {
    const payload = parseManualMatchImport(await file.text())
    if (scene.value?.id !== sceneId) return
    const result = await api.importSceneMatchBinding(sceneId, payload)
    if (scene.value?.id !== sceneId) return
    invalidateIdentityReview()
    scene.value = result.scene
    clearStaleFrameAnalysis()
    const status = result.scene.payload.videoAsset?.reconstruction?.status
    if (status === 'queued' || status === 'processing') {
      reconstructing.value = true
      saveState.value = 'Project roster imported · rebuilding identity…'
      void pollReconstruction(sceneId)
    } else {
      saveState.value = 'Project roster imported'
      void loadIdentityReview(sceneId)
    }
  } catch (cause) {
    manualRosterImportError.value = cause instanceof Error
      ? cause.message
      : 'Could not import the roster JSON.'
    saveState.value = 'Manual roster was not imported'
  } finally {
    manualRosterImporting.value = false
    input.value = ''
  }
}

async function confirmCanonicalRoster(payload: { canonicalPersonId: string; externalPlayerId: string }) {
  if (!scene.value || reconstructing.value || reconstructionRunning.value || rosterBindingSaving.value) return
  const identity = canonicalPersonById(payload.canonicalPersonId)
  const rosterPlayer = rosterPlayers.value.find(
    (player) => player.id === payload.externalPlayerId,
  )
  if (!identity || !rosterPlayer) {
    error.value = 'The selected roster candidate is no longer available'
    return
  }
  await updateCanonicalRosterBinding(payload.canonicalPersonId, rosterPlayer.id)
}

async function rejectIdentityCandidate(payload: IdentityReviewCandidateDecision) {
  if (payload.kind !== 'roster') {
    error.value = 'Identity-link rejection is not available from this review snapshot'
    return
  }
  if (
    !scene.value
    || identityDecisionSaving.value
    || rosterBindingSaving.value
    || reconstructionMutationLocked.value
  ) return
  if (!canonicalPersonById(payload.canonicalPersonId)) {
    error.value = 'The selected canonical identity is no longer available'
    return
  }
  if (!rosterPlayers.value.some((player) => player.id === payload.externalPlayerId)) {
    error.value = 'The roster candidate is absent from the saved match snapshot'
    return
  }
  const sceneId = scene.value.id
  identityDecisionSaving.value = true
  error.value = null
  saveState.value = 'Rejecting roster hypothesis…'
  try {
    const updated = await api.rejectRosterCandidate(
      sceneId,
      payload.canonicalPersonId,
      payload.externalPlayerId,
    )
    if (scene.value?.id !== sceneId) return
    invalidateIdentityReview()
    scene.value = updated
    clearStaleFrameAnalysis()
    selectedCanonicalPersonId.value = payload.canonicalPersonId
    selectedTrackId.value = renderTrackForCanonicalPerson(payload.canonicalPersonId)?.id ?? null
    const status = updated.payload.videoAsset?.reconstruction?.status
    if (status === 'queued' || status === 'processing') {
      reconstructing.value = true
      saveState.value = 'Roster hypothesis rejected · rebuilding identity…'
      void pollReconstruction(sceneId)
    } else {
      saveState.value = 'Roster hypothesis rejected'
      void loadIdentityReview(sceneId)
    }
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not reject roster candidate'
    saveState.value = 'Roster rejection was not saved'
  } finally {
    identityDecisionSaving.value = false
  }
}

function inspectIdentityFrame(payload: IdentityReviewInspectFrame) {
  if (!scene.value?.payload.canonicalPeople?.some(
    (person) => person.canonicalPersonId === payload.canonicalPersonId,
  )) return
  playing.value = false
  sourceVideo.value?.pause()
  selectedCanonicalPersonId.value = payload.canonicalPersonId
  selectedTrackId.value = renderTrackForCanonicalPerson(payload.canonicalPersonId)?.id ?? null
  viewMode.value = sceneVideo.value ? 'split' : viewMode.value
  activeTab.value = 'binding'
  seekTo(payload.sceneTime)
  saveState.value = `Identity observation · frame ${payload.frameIndex}`
  void nextTick().then(() => analyzeCurrentFrame())
}

async function unbindCanonicalRoster(payload: { canonicalPersonId: string }) {
  if (!canonicalPersonById(payload.canonicalPersonId)) {
    error.value = 'The selected canonical identity is no longer available'
    return
  }
  await updateCanonicalRosterBinding(payload.canonicalPersonId, null)
}

async function clearCanonicalRosterBinding(payload: { canonicalPersonId: string }) {
  if (
    !scene.value
    || saving.value
    || reconstructionMutationLocked.value
    || rosterBindingSaving.value
  ) return
  if (!canonicalHasActiveDedicatedUnbind(payload.canonicalPersonId)) {
    error.value = 'There is no active manual Unbind decision to clear for this identity'
    return
  }
  const sceneId = scene.value.id
  rosterBindingSaving.value = true
  try {
    const queued = await api.clearCanonicalRosterBinding(sceneId, payload.canonicalPersonId)
    if (scene.value?.id !== sceneId) return
    invalidateIdentityReview()
    scene.value = queued
    clearStaleFrameAnalysis()
    selectedCanonicalPersonId.value = payload.canonicalPersonId
    selectedTrackId.value = renderTrackForCanonicalPerson(payload.canonicalPersonId)?.id ?? null
    selectedFramePersonId.value = null
    const status = queued.payload.videoAsset?.reconstruction?.status
    if (status === 'queued' || status === 'processing') {
      reconstructing.value = true
      saveState.value = 'Manual Unbind cleared · rebuilding identity…'
      void pollReconstruction(sceneId)
    } else {
      saveState.value = 'Manual Unbind cleared'
      void loadIdentityReview(sceneId)
    }
  } catch (cause) {
    error.value = cause instanceof Error
      ? cause.message
      : 'Could not clear the manual roster Unbind decision'
  } finally {
    rosterBindingSaving.value = false
  }
}

async function updateCanonicalRosterBinding(
  canonicalPersonId: string,
  externalPlayerId: string | null,
) {
  if (!scene.value || reconstructing.value || reconstructionRunning.value || rosterBindingSaving.value) return
  const sceneId = scene.value.id
  rosterBindingSaving.value = true
  try {
    const queued = await api.updateCanonicalRosterBinding(
      sceneId,
      canonicalPersonId,
      externalPlayerId,
    )
    if (scene.value?.id !== sceneId) return
    invalidateIdentityReview()
    scene.value = queued
    clearStaleFrameAnalysis()
    selectedCanonicalPersonId.value = canonicalPersonId
    selectedTrackId.value = renderTrackForCanonicalPerson(canonicalPersonId)?.id ?? null
    selectedFramePersonId.value = null
    const status = queued.payload.videoAsset?.reconstruction?.status
    if (status === 'queued' || status === 'processing') {
      reconstructing.value = true
      saveState.value = externalPlayerId
        ? 'Roster binding saved · rebuilding identity…'
        : 'Roster binding removed · rebuilding identity…'
      void pollReconstruction(sceneId)
    } else {
      saveState.value = externalPlayerId ? 'Roster binding saved' : 'Roster binding removed'
      void loadIdentityReview(sceneId)
    }
  } catch (cause) {
    error.value = cause instanceof Error
      ? cause.message
      : externalPlayerId
        ? 'Could not save roster binding'
        : 'Could not remove roster binding'
  } finally {
    rosterBindingSaving.value = false
  }
}

function addEventBinding(item: TimelineEvent) {
  if (!scene.value) return
  scene.value.payload.eventBindings.push({
    sceneTime: Number(currentTime.value.toFixed(2)),
    externalEventId: item.id,
    label: item.label,
    type: item.type,
  })
  saveState.value = 'Unsaved event marker'
}

function removeEventBinding(index: number) {
  scene.value?.payload.eventBindings.splice(index, 1)
  saveState.value = 'Unsaved changes'
}

async function openProcessedVideo(asset: VideoAsset) {
  if (!asset.scene_id) return
  const parentScene = await api.getScene(asset.scene_id)
  scenes.value = [
    { id: parentScene.id, title: parentScene.title, duration: parentScene.duration, kind: 'video' },
    ...scenes.value.filter((item) => item.id !== parentScene.id),
  ]
  scene.value = parentScene
  currentTime.value = 0
  selectedTrackId.value = null
  selectedCanonicalPersonId.value = null
  invalidateIdentityReview()
  void loadIdentityReview(parentScene.id)
  videoIngestOpen.value = false
  viewMode.value = 'split'
  saveState.value = 'Video timeline ready'
}

async function createSceneFromSegment(segment: VideoSegment) {
  if (!sceneVideo.value) return
  segmentCreating.value = segment.id
  try {
    const createdScene = segment.sceneId
      ? await api.getScene(segment.sceneId)
      : await api.createSegmentScene(sceneVideo.value.id, segment.id)
    scene.value = createdScene
    selectedTrackId.value = createdScene.payload.tracks[0]?.id ?? null
    currentTime.value = 0
    invalidateIdentityReview()
    void loadIdentityReview(createdScene.id)
    saveState.value = 'Shot scene created'
    await nextTick()
    seekTo(0)
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not create shot scene'
  } finally {
    segmentCreating.value = null
  }
}

async function startMultiPass() {
  if (!sceneVideo.value || multiPassSelection.value.length < 2 || multiPassStarting.value) return
  const selectedCount = multiPassSelection.value.length
  multiPassStarting.value = true
  reconstructing.value = true
  invalidateIdentityReview()
  saveState.value = `Analyzing ${selectedCount} camera angles…`
  try {
    const createdScene = await api.createMultiPass(sceneVideo.value.id, multiPassSelection.value)
    scene.value = createdScene
    selectedTrackId.value = null
    selectedCanonicalPersonId.value = null
    currentTime.value = 0
    multiPassSelection.value = []
    invalidateIdentityReview()
    await pollReconstruction(createdScene.id)
  } catch (cause) {
    reconstructing.value = false
    error.value = cause instanceof Error ? cause.message : 'Could not start multi-angle analysis'
  } finally {
    multiPassStarting.value = false
  }
}

async function pollReconstruction(sceneId: string) {
  try {
    const canonicalSelection = selectedCanonicalPersonId.value
    const updated = await api.getScene(sceneId)
    if (scene.value?.id !== sceneId) return
    const status = updated.payload.videoAsset?.reconstruction?.status
    scene.value = updated
    clearStaleFrameAnalysis()
    if (status === 'queued' || status === 'processing') {
      reconstructing.value = true
      const progress = updated.payload.videoAsset?.reconstruction?.progress
      saveState.value = progress
        ? `${progress.label} · ${progress.overallPercent}% · ${progress.etaSeconds === null ? 'estimating time' : `${formatAnalysisDuration(progress.etaSeconds)} left`}`
        : status === 'queued' ? 'Analysis queued…' : 'Analysis running…'
      reconstructionTimer = window.setTimeout(() => pollReconstruction(sceneId), 700)
      return
    }
    reconstructing.value = false
    const canonicalStillExists = canonicalSelection
      ? updated.payload.canonicalPeople?.some(
        (person) => person.canonicalPersonId === canonicalSelection,
      ) === true
      : false
    const selectedRenderTrack = canonicalStillExists
      ? updated.payload.tracks.find((track) => track.canonicalPersonId === canonicalSelection) ?? null
      : null
    selectedTrackId.value = canonicalStillExists
      ? selectedRenderTrack?.id ?? null
      : updated.payload.tracks[0]?.id ?? null
    selectedCanonicalPersonId.value = canonicalStillExists
      ? canonicalSelection
      : selectedRenderTrack?.canonicalPersonId
        ?? updated.payload.tracks[0]?.canonicalPersonId
        ?? null
    const multiPass = updated.payload.videoAsset?.multiPass
    const verdict = updated.payload.videoAsset?.reconstruction?.qualityVerdict
      ?? updated.payload.videoAsset?.reconstruction?.quality?.verdict
      ?? 'unknown'
    if (status === 'ready' && verdict !== 'pass') activeTab.value = 'qa'
    saveState.value = status === 'ready'
      ? multiPass?.consensus
        ? `${multiPass.consensus.passesAnalyzed} angles · ${Math.round(multiPass.consensus.evidenceScore * 100)}% evidence`
        : `Compute complete · quality ${verdict} · ${updated.payload.tracks.length} automatic tracks`
      : updated.payload.videoAsset?.reconstruction?.error || 'Reconstruction needs review'
    void loadIdentityReview(sceneId)
  } catch (cause) {
    reconstructing.value = false
    error.value = cause instanceof Error ? cause.message : 'Could not read reconstruction status'
  }
}

function resumeReconstructionPolling() {
  window.clearTimeout(reconstructionTimer)
  if (!scene.value || !reconstructionRunning.value) {
    reconstructing.value = false
    return
  }
  reconstructing.value = true
  const progress = reconstructionProgress.value
  saveState.value = progress
    ? `${progress.label} · ${progress.overallPercent}%`
    : 'Analysis running…'
  void pollReconstruction(scene.value.id)
}

async function reconstructCurrentScene() {
  if (!scene.value || !sceneVideo.value?.selectedSegmentId || reconstructing.value) return
  clearStaleFrameAnalysis()
  invalidateIdentityReview()
  reconstructing.value = true
  saveState.value = `Detecting people with ${selectedReconstructionModel.value.replace('.pt', '')} · ball ${selectedBallBackend.value}…`
  try {
    scene.value = await api.reconstructScene(
      scene.value.id,
      selectedReconstructionModel.value,
      selectedBallBackend.value,
    )
    await pollReconstruction(scene.value.id)
  } catch (cause) {
    reconstructing.value = false
    error.value = cause instanceof Error ? cause.message : 'Could not start reconstruction'
  }
}

async function queueIdentityCorrectionRebuild(sceneId: string, sceneTime: number, label: string) {
  invalidateIdentityReview()
  reconstructing.value = true
  if (scene.value?.id !== sceneId) return
  seekTo(sceneTime)
  saveState.value = `${label} saved · rebuilding the affected tracking…`
  // The annotation mutation and queued reconstruction are one server-side
  // compare-and-swap. This client only observes that already-started run.
  void pollReconstruction(sceneId)
}

function validMatchedTrackId(person: FrameAnalysis['people'][number]) {
  const trackId = person.matchedTrackId
  if (!trackId || !scene.value?.payload.tracks.some((track) => track.id === trackId)) return null
  return trackId
}

function bestFramePersonForTrack(
  analysis: FrameAnalysis | null,
  trackId: string | null,
  canonicalPersonId: string | null = null,
) {
  if (!analysis || (!trackId && !canonicalPersonId)) return null
  const sourcePriority: Record<NonNullable<FrameAnalysis['people'][number]['matchSource']>, number> = {
    'persisted-observation': 0,
    'manual-identity': 1,
    'legacy-observed-frame': 2,
  }
  return [...selectedFramePeople(analysis, trackId, canonicalPersonId)].sort((left, right) => {
    const sourceDelta = (left.matchSource ? sourcePriority[left.matchSource] : 9)
      - (right.matchSource ? sourcePriority[right.matchSource] : 9)
    if (sourceDelta) return sourceDelta
    const distanceDelta = (left.matchDistance ?? Number.POSITIVE_INFINITY)
      - (right.matchDistance ?? Number.POSITIVE_INFINITY)
    if (distanceDelta) return distanceDelta
    return right.confidence - left.confidence || left.id.localeCompare(right.id)
  })[0] ?? null
}

function selectTrack(trackId: string) {
  ballSelected.value = false
  ballEditMode.value = false
  selectedBallKeyframeTime.value = null
  const track = scene.value?.payload.tracks.find((item) => item.id === trackId) ?? null
  selectedTrackId.value = trackId
  selectedCanonicalPersonId.value = track?.canonicalPersonId ?? null
  selectedFramePersonId.value = bestFramePersonForTrack(
    activeFrameAnalysis.value,
    trackId,
    track?.canonicalPersonId ?? null,
  )?.id ?? null
  if (track) saveState.value = `${track.label} selected in video and 3D`
}

function selectCanonicalPerson(canonicalPersonId: string) {
  ballSelected.value = false
  ballEditMode.value = false
  selectedBallKeyframeTime.value = null
  const renderTrack = renderTrackForCanonicalPerson(canonicalPersonId)
  selectedCanonicalPersonId.value = canonicalPersonId
  selectedTrackId.value = renderTrack?.id ?? null
  selectedFramePersonId.value = bestFramePersonForTrack(
    activeFrameAnalysis.value,
    renderTrack?.id ?? null,
    canonicalPersonId,
  )?.id ?? null
  const identity = canonicalPersonById(canonicalPersonId)
  saveState.value = renderTrack
    ? `${identity?.displayName ?? canonicalPersonId} selected in video and 3D`
    : `${identity?.displayName ?? canonicalPersonId} selected · not projected in 3D`
}

async function runCurrentFrameAnalysis(preserveTrackId?: string): Promise<FrameAnalysis | null> {
  if (!scene.value || !sceneVideo.value?.selectedSegmentId) return null
  const sceneId = scene.value.id
  const requestedTime = currentTime.value
  if (
    frameAnalyzing.value
    && activeFrameAnalysisRequest?.sceneId === sceneId
    && Math.abs(activeFrameAnalysisRequest.sceneTime - requestedTime) <= 0.11
  ) return null
  const requestId = ++frameAnalysisRequestId
  activeFrameAnalysisRequest = { sceneId, sceneTime: requestedTime }
  const selectionAtStart = selectedTrackId.value
  const canonicalSelectionAtStart = selectedCanonicalPersonId.value
  const framePersonSelectionAtStart = selectedFramePersonId.value
  playing.value = false
  sourceVideo.value?.pause()
  frameAnalyzing.value = true
  saveState.value = preserveTrackId
    ? `Matching ${selectedTrack.value?.label ?? preserveTrackId} in source frame…`
    : `Analyzing frame at ${requestedTime.toFixed(2)}s…`
  try {
    const result = await api.analyzeFrame(sceneId, requestedTime)
    if (requestId !== frameAnalysisRequestId) return null
    if (scene.value?.id !== sceneId || Math.abs(currentTime.value - requestedTime) > 0.11) {
      saveState.value = 'Frame changed · discarded stale analysis result'
      return null
    }
    frameAnalysis.value = result
    seekTo(result.sceneTime)
    const firstPerson = result.people.find((item) => item.canonicalPersonId || item.matchedTrackId) ?? null
    const firstCanonicalPersonId = firstPerson ? framePersonCanonicalId(firstPerson) : null
    const firstMatch = renderTrackForCanonicalPerson(firstCanonicalPersonId)?.id
      ?? (firstPerson ? validMatchedTrackId(firstPerson) : null)
    const canonicalSelectionChanged = selectedCanonicalPersonId.value !== canonicalSelectionAtStart
    selectedTrackId.value = selectionAfterFrameAnalysis(
      selectionAtStart,
      selectedTrackId.value,
      canonicalSelectionAtStart || canonicalSelectionChanged ? null : firstMatch,
      preserveTrackId,
    )
    const selectedRenderTrack = scene.value?.payload.tracks.find(
      (track) => track.id === selectedTrackId.value,
    )
    selectedCanonicalPersonId.value = canonicalSelectionAfterFrameAnalysis(
      canonicalSelectionAtStart,
      selectedCanonicalPersonId.value,
      selectedTrackId.value,
      selectedRenderTrack?.canonicalPersonId,
      firstCanonicalPersonId,
    )
    const matchedPerson = bestFramePersonForTrack(
      result,
      selectedTrackId.value,
      selectedCanonicalPersonId.value,
    )
    if (matchedPerson) {
      selectedFramePersonId.value = matchedPerson.id
    } else if (!selectedTrackId.value) {
      const requestedPersonId = selectedFramePersonId.value !== framePersonSelectionAtStart
        ? selectedFramePersonId.value
        : framePersonSelectionAtStart
      selectedFramePersonId.value = result.people.some((person) => person.id === requestedPersonId)
        ? requestedPersonId
        : null
    } else {
      selectedFramePersonId.value = null
    }
    frameDetectionHitCycle = null
    activeTab.value = 'binding'
    const selectedMatches = selectedTrackId.value || selectedCanonicalPersonId.value
      ? selectedFramePeople(result, selectedTrackId.value, selectedCanonicalPersonId.value).length
      : 0
    saveState.value = preserveTrackId
      ? selectedMatches
        ? `${selectedTrack.value?.label ?? selectedTrackId.value} linked across video and 3D`
        : `${selectedTrack.value?.label ?? selectedTrackId.value} is not visible in this source frame`
      : `${result.people.length} people · ${result.matchedTracks} matched at ${result.sceneTime.toFixed(2)}s`
    return result
  } catch (cause) {
    if (requestId === frameAnalysisRequestId) {
      error.value = cause instanceof Error ? cause.message : 'Could not analyze this frame'
    }
    return null
  } finally {
    if (requestId === frameAnalysisRequestId) {
      frameAnalyzing.value = false
      activeFrameAnalysisRequest = null
    }
  }
}

async function analyzeCurrentFrame() {
  await runCurrentFrameAnalysis()
}

async function selectTrackFromThree(trackId: string) {
  selectTrack(trackId)
  const currentAnalysis = activeFrameAnalysis.value
  if (currentAnalysis) {
    const matches = selectedFramePeople(currentAnalysis, trackId, selectedCanonicalPersonId.value)
    saveState.value = matches.length
      ? `${selectedTrack.value?.label ?? trackId} selected in video and 3D`
      : `${selectedTrack.value?.label ?? trackId} selected · no visible source detection`
    return
  }
  await runCurrentFrameAnalysis(trackId)
}

function syncFrameAnnotations(result: FrameAnalysis) {
  const reconstruction = scene.value?.payload.videoAsset?.reconstruction
  if (!reconstruction) return
  reconstruction.frameAnnotations = [
    ...(reconstruction.frameAnnotations ?? []).filter((item) => item.frameIndex !== result.frameIndex),
    ...result.annotations,
  ].sort((left, right) => left.frameIndex - right.frameIndex || left.id.localeCompare(right.id))
}

function clearStaleFrameAnalysis() {
  frameAnalysisRequestId += 1
  activeFrameAnalysisRequest = null
  frameAnalyzing.value = false
  frameAnalysis.value = null
  selectedFramePersonId.value = null
  frameAnnotationMode.value = false
  frameAnnotationDraft.value = null
  frameAnnotationDrag.value = null
  frameDetectionHitCycle = null
}

async function toggleFrameAnnotationMode() {
  if (frameAnnotationMode.value) {
    frameAnnotationMode.value = false
    frameAnnotationDraft.value = null
    frameAnnotationDrag.value = null
    saveState.value = 'Frame labeling closed'
    return
  }
  if (!activeFrameAnalysis.value) await analyzeCurrentFrame()
  if (!frameAnalysis.value) return
  frameAnnotationMode.value = true
  viewMode.value = 'split'
  playing.value = false
  sourceVideo.value?.pause()
  activeTab.value = 'binding'
  saveState.value = 'Select a box or drag around any person'
}

function defaultAnnotationKind(person: FrameAnalysis['people'][number]): FrameAnnotationKind {
  if (person.kind) return person.kind
  if (person.teamId === 'home') return 'home-player'
  if (person.teamId === 'away') return 'away-player'
  if (person.teamId === 'officials') return 'referee'
  return 'other'
}

function selectDetectedPerson(person: FrameAnalysis['people'][number]) {
  ballSelected.value = false
  ballEditMode.value = false
  selectedBallKeyframeTime.value = null
  selectedFramePersonId.value = person.id
  const canonicalPersonId = framePersonCanonicalId(person)
  const linkedTrackId = renderTrackForFramePerson(
    person,
    scene.value?.payload.tracks ?? [],
  )?.id ?? null
  selectedCanonicalPersonId.value = canonicalPersonId
  selectedTrackId.value = linkedTrackId
  if (!frameAnnotationMode.value) {
    saveState.value = linkedTrackId
      ? `${framePersonLabel(person)} selected in video and 3D`
      : canonicalPersonId
        ? `${framePersonLabel(person)} selected · identity exists, not projected in 3D`
        : `${framePersonLabel(person)} selected · identity is not resolved yet`
    return
  }
  const annotations = scene.value?.payload.videoAsset?.reconstruction?.frameAnnotations ?? []
  const persistedAnnotation = semanticAnnotationForEdit(person, annotations)
  const semanticAnnotationId = persistedAnnotation?.id ?? null
  const hasDedicatedRosterCorrection = [
    ...(person.annotationIds ?? []),
    ...(person.annotationId ? [person.annotationId] : []),
  ].some((annotationId) => annotations.find(
    (item) => item.id === annotationId,
  )?.correctionKind === 'canonical-roster-binding-v1') || annotations.some((annotation) => (
    annotation.correctionKind === 'canonical-roster-binding-v1'
    && annotation.canonicalPersonId === canonicalPersonId
  ))
  frameAnnotationDraft.value = {
    annotationId: semanticAnnotationId,
    bbox: { ...(persistedAnnotation?.bbox ?? person.bbox) },
    kind: persistedAnnotation?.kind ?? defaultAnnotationKind(person),
    label: persistedAnnotation
      ? persistedAnnotation.label ?? ''
      : hasDedicatedRosterCorrection
        ? ''
        : person.annotationLabel || person.displayName || person.matchedTrackLabel || '',
    externalPlayerId: null,
    action: persistedAnnotation
      ? annotationIdentityAction(persistedAnnotation)
      : hasDedicatedRosterCorrection
        ? 'confirm'
        : person.correctionAction ?? 'confirm',
    scope: persistedAnnotation?.scope
      ?? (hasDedicatedRosterCorrection
        ? (linkedTrackId || canonicalPersonId ? 'identity' : 'observation')
        : person.correctionScope ?? (linkedTrackId ? 'identity' : 'observation')),
    mergeTargetId: persistedAnnotation?.mergeTargetId
      ?? (hasDedicatedRosterCorrection ? null : person.mergeTargetId),
    sourceTrackId: persistedAnnotation?.sourceTrackId ?? person.sourceTrackId ?? linkedTrackId,
    canonicalPersonId: persistedAnnotation?.canonicalPersonId ?? canonicalPersonId,
    targetObservationId: persistedAnnotation?.targetObservationId
      ?? person.targetObservationId
      ?? person.observationId
      ?? null,
    rangeStart: persistedAnnotation?.rangeStart
      ?? person.rangeStart
      ?? activeFrameAnalysis.value?.sceneTime
      ?? null,
    rangeEnd: persistedAnnotation?.rangeEnd ?? person.rangeEnd ?? scene.value?.duration ?? null,
    affectedPreview: persistedAnnotation?.affectedPreview ?? person.affectedPreview ?? null,
  }
  saveState.value = persistedAnnotation
    ? 'Editing saved frame label'
    : hasDedicatedRosterCorrection
      ? 'New semantic correction · roster Bind / Unbind remains separate'
      : 'Detection selected for labeling'
}

function selectFrameAnnotation(annotation: FrameAnnotation) {
  if (!frameAnnotationMode.value) return
  const dedicatedRosterCorrection = annotation.correctionKind === 'canonical-roster-binding-v1'
  frameAnnotationDraft.value = {
    annotationId: dedicatedRosterCorrection ? null : annotation.id,
    bbox: { ...annotation.bbox },
    kind: annotation.kind,
    label: dedicatedRosterCorrection ? '' : annotation.label || '',
    externalPlayerId: null,
    action: dedicatedRosterCorrection ? 'confirm' : annotationIdentityAction(annotation),
    scope: annotation.scope ?? 'observation',
    mergeTargetId: dedicatedRosterCorrection ? null : annotation.mergeTargetId ?? null,
    sourceTrackId: annotation.sourceTrackId ?? null,
    canonicalPersonId: annotation.canonicalPersonId
      ?? scene.value?.payload.tracks.find(
        (track) => track.id === annotation.sourceTrackId,
      )?.canonicalPersonId
      ?? canonicalPersonById(annotation.sourceTrackId)?.canonicalPersonId
      ?? null,
    targetObservationId: annotation.targetObservationId ?? null,
    rangeStart: annotation.rangeStart ?? annotation.sceneTime,
    rangeEnd: annotation.rangeEnd ?? scene.value?.duration ?? null,
    affectedPreview: annotation.affectedPreview ?? null,
  }
  saveState.value = dedicatedRosterCorrection
    ? 'New semantic correction · roster Bind / Unbind remains separate'
    : 'Editing saved frame label'
}

function onFrameIdentityActionChange() {
  const draft = frameAnnotationDraft.value
  if (!draft) return
  if (draft.action !== 'exclude' && draft.kind === 'ignore') {
    draft.kind = 'other'
    draft.label = ''
    draft.externalPlayerId = null
  }
  if (draft.action === 'merge') draft.scope = 'identity'
  if (draft.action === 'split') {
    draft.scope = 'range'
    draft.externalPlayerId = null
    draft.rangeStart ??= activeFrameAnalysis.value?.sceneTime ?? null
    draft.rangeEnd ??= scene.value?.duration ?? null
  }
  if (draft.action !== 'merge') draft.mergeTargetId = null
}

function frameAnalysisImagePoint(event: MouseEvent | PointerEvent) {
  const svg = frameAnalysisOverlay.value
  const analysis = activeFrameAnalysis.value
  if (!svg || !analysis) return null
  const rect = svg.getBoundingClientRect()
  return clientPointToContainedMedia(
    event.clientX,
    event.clientY,
    rect,
    analysis.frameWidth,
    analysis.frameHeight,
  )
}

function selectFramePersonAtPoint(event: MouseEvent) {
  if (suppressNextFrameOverlayClick || frameAnnotationDrag.value) {
    suppressNextFrameOverlayClick = false
    return
  }
  const analysis = activeFrameAnalysis.value
  const svg = frameAnalysisOverlay.value
  const point = frameAnalysisImagePoint(event)
  if (!analysis || !svg || !point) return
  const rect = svg.getBoundingClientRect()
  const renderedScale = Math.min(
    rect.width / Math.max(1, analysis.frameWidth),
    rect.height / Math.max(1, analysis.frameHeight),
  )
  const minimumTargetSize = 24 / Math.max(0.001, renderedScale)
  const previousCycle = frameDetectionHitCycle
  const sameHitCluster = previousCycle?.frameIndex === analysis.frameIndex
    && (previousCycle.x - point.x) ** 2 + (previousCycle.y - point.y) ** 2 <= (minimumTargetSize / 2) ** 2
  const person = selectFrameDetectionHit(analysis.people, point, {
    minimumTargetSize,
    previousCandidateId: sameHitCluster ? previousCycle.personId : null,
  })
  if (!person) {
    frameDetectionHitCycle = null
    return
  }
  frameDetectionHitCycle = {
    frameIndex: analysis.frameIndex,
    x: point.x,
    y: point.y,
    personId: person.id,
  }
  selectDetectedPerson(person)
}

function startFrameAnnotationDrag(event: PointerEvent) {
  if (!frameAnnotationMode.value || event.button !== 0 || event.target !== event.currentTarget) return
  const point = frameAnalysisImagePoint(event)
  if (!point) return
  frameAnnotationDrag.value = { ...point, pointerId: event.pointerId }
  frameAnnotationDraft.value = {
    annotationId: null,
    bbox: { x: point.x, y: point.y, width: 4, height: 4 },
    kind: 'home-player',
    label: '',
    externalPlayerId: null,
    action: 'confirm',
    scope: 'identity',
    mergeTargetId: null,
    sourceTrackId: null,
    canonicalPersonId: null,
    targetObservationId: null,
    rangeStart: null,
    rangeEnd: null,
    affectedPreview: null,
  }
  try {
    frameAnalysisOverlay.value?.setPointerCapture(event.pointerId)
  } catch {
    // Pointer capture is optional for synthetic input.
  }
}

function updateFrameAnnotationDrag(event: PointerEvent) {
  const start = frameAnnotationDrag.value
  const draft = frameAnnotationDraft.value
  const point = frameAnalysisImagePoint(event)
  if (!start || !draft || !point || start.pointerId !== event.pointerId) return
  draft.bbox = {
    x: Math.min(start.x, point.x),
    y: Math.min(start.y, point.y),
    width: Math.max(4, Math.abs(point.x - start.x)),
    height: Math.max(4, Math.abs(point.y - start.y)),
  }
}

function finishFrameAnnotationDrag(event: PointerEvent) {
  const start = frameAnnotationDrag.value
  const point = frameAnalysisImagePoint(event)
  if (!start || !point || start.pointerId !== event.pointerId) return
  suppressNextFrameOverlayClick = event.type === 'pointerup'
  if (suppressNextFrameOverlayClick) {
    window.setTimeout(() => {
      suppressNextFrameOverlayClick = false
    }, 0)
  }
  updateFrameAnnotationDrag(event)
  frameAnnotationDrag.value = null
  try {
    if (frameAnalysisOverlay.value?.hasPointerCapture(event.pointerId)) {
      frameAnalysisOverlay.value.releasePointerCapture(event.pointerId)
    }
  } catch {
    // The browser may release capture before pointerup reaches the overlay.
  }
  if (Math.abs(point.x - start.x) < 6 || Math.abs(point.y - start.y) < 10) {
    frameAnnotationDraft.value = null
    saveState.value = 'Draw a larger box around the full person'
  } else {
    saveState.value = 'New manual person box ready'
  }
}

async function saveFrameAnnotation() {
  const draft = frameAnnotationDraft.value
  const analysis = activeFrameAnalysis.value
  if (!scene.value || !draft || !analysis || frameAnnotationSaving.value || reconstructing.value || reconstructionStatus.value === 'queued' || reconstructionStatus.value === 'processing') return
  const sceneId = scene.value.id
  frameAnnotationSaving.value = true
  try {
    const result = await api.saveFrameAnnotation(sceneId, {
      annotationId: draft.annotationId,
      sceneTime: analysis.sceneTime,
      bbox: draft.bbox,
      kind: draft.action === 'exclude' ? 'ignore' : draft.kind,
      label: ['exclude', 'split'].includes(draft.action) ? null : draft.label.trim() || null,
      externalPlayerId: null,
      action: draft.action,
      scope: draft.action === 'merge' ? 'identity' : draft.action === 'split' ? 'range' : draft.scope,
      mergeTargetId: draft.action === 'merge' ? draft.mergeTargetId : null,
      sourceTrackId: draft.sourceTrackId,
      canonicalPersonId: draft.canonicalPersonId,
      targetObservationId: draft.action === 'split' ? draft.targetObservationId : null,
      rangeStart: draft.action === 'split' ? draft.rangeStart : null,
      rangeEnd: draft.action === 'split' ? draft.rangeEnd : null,
    })
    if (scene.value?.id !== sceneId) return
    scene.value = mergeFrameReconstructionMetadata(scene.value, result)
    syncFrameAnnotations(result)
    clearStaleFrameAnalysis()
    seekTo(result.sceneTime)
    const correctionLabel = draft.action === 'merge'
      ? 'Identity merge'
      : draft.action === 'split'
        ? 'Identity split'
        : draft.action === 'exclude'
          ? 'Exclusion'
          : 'Tracking confirmation'
    if (draft.action === 'merge' && draft.mergeTargetId) {
      const targetTrack = scene.value.payload.tracks.find((track) => (
        track.id === draft.mergeTargetId || track.canonicalPersonId === draft.mergeTargetId
      )) ?? null
      selectedTrackId.value = targetTrack?.id ?? null
      selectedCanonicalPersonId.value = canonicalPersonById(draft.mergeTargetId)?.canonicalPersonId
        ?? targetTrack?.canonicalPersonId
        ?? null
    }
    if (draft.action === 'exclude' && draft.scope === 'identity') {
      selectedTrackId.value = null
      selectedCanonicalPersonId.value = null
    }
    await queueIdentityCorrectionRebuild(sceneId, result.sceneTime, correctionLabel)
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not save frame label'
  } finally {
    frameAnnotationSaving.value = false
  }
}

async function deleteFrameAnnotation() {
  const annotationId = frameAnnotationDraft.value?.annotationId
  if (!scene.value || !annotationId || frameAnnotationSaving.value || reconstructing.value || reconstructionStatus.value === 'queued' || reconstructionStatus.value === 'processing') return
  const sceneId = scene.value.id
  frameAnnotationSaving.value = true
  try {
    const result = await api.deleteFrameAnnotation(sceneId, annotationId)
    if (scene.value?.id !== sceneId) return
    scene.value = mergeFrameReconstructionMetadata(scene.value, result)
    syncFrameAnnotations(result)
    clearStaleFrameAnalysis()
    seekTo(result.sceneTime)
    await queueIdentityCorrectionRebuild(sceneId, result.sceneTime, 'Correction removal')
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not remove frame label'
  } finally {
    frameAnnotationSaving.value = false
  }
}

async function compareCurrentSceneModels() {
  if (!scene.value || !sceneVideo.value?.selectedSegmentId || modelComparing.value) return
  playing.value = false
  sourceVideo.value?.pause()
  modelComparing.value = true
  saveState.value = `Comparing yolo26n and yolo26m on ${analysisFrameCount.value} frames…`
  try {
    const report = await api.compareModels(scene.value.id)
    const reconstruction = scene.value.payload.videoAsset?.reconstruction
    if (reconstruction) reconstruction.modelComparison = report
    selectedTrackId.value ??= scene.value.payload.tracks[0]?.id ?? null
    activeTab.value = 'binding'
    const gain = report.comparison.inPitchObservationGain
    saveState.value = `Model comparison ready · in-pitch observation delta ${gain >= 0 ? '+' : ''}${gain}`
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not compare recognition models'
  } finally {
    modelComparing.value = false
  }
}

async function openPitchCalibration(preset?: PitchCalibrationPreset) {
  if (!scene.value || !sceneVideo.value?.selectedSegmentId || calibrationLoading.value) return
  playing.value = false
  sourceVideo.value?.pause()
  viewMode.value = 'split'
  clearStaleFrameAnalysis()
  calibrationLoading.value = true
  if (preset) calibrationPreset.value = preset
  saveState.value = `Preparing pitch overlay at ${currentTime.value.toFixed(2)}s…`
  try {
    const draft = await api.autoPitchCalibration(scene.value.id, currentTime.value, preset)
    calibrationDraft.value = draft
    calibrationPreset.value = draft.preset
    seekTo(draft.sceneTime)
    saveState.value = draft.alignmentError === null
      ? 'Pitch anchors ready for review'
      : `Pitch overlay · ${draft.alignmentError.toFixed(1)} px error`
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not prepare pitch calibration'
  } finally {
    calibrationLoading.value = false
  }
}

async function changePitchCalibrationPreset(event: Event) {
  const preset = (event.target as HTMLSelectElement).value as PitchCalibrationPreset
  await openPitchCalibration(preset)
}

function calibrationImagePoint(event: PointerEvent) {
  const svg = calibrationOverlay.value
  const draft = calibrationDraft.value
  if (!svg || !draft) return null
  const rect = svg.getBoundingClientRect()
  return clientPointToContainedMedia(
    event.clientX,
    event.clientY,
    rect,
    draft.frameWidth,
    draft.frameHeight,
  )
}

function updateDraggedCalibrationAnchor(event: PointerEvent) {
  const draft = calibrationDraft.value
  const anchorId = draggedCalibrationAnchor.value
  const point = calibrationImagePoint(event)
  if (!draft || !anchorId || !point) return
  const anchor = draft.anchors.find((item) => item.id === anchorId)
  if (!anchor) return
  anchor.image.x = point.x
  anchor.image.y = point.y
}

function startCalibrationAnchorDrag(event: PointerEvent, anchorId: string) {
  draggedCalibrationAnchor.value = anchorId
  try {
    calibrationOverlay.value?.setPointerCapture(event.pointerId)
  } catch {
    // Synthetic accessibility input may not own a native pointer capture.
  }
  updateDraggedCalibrationAnchor(event)
}

async function refreshPitchCalibration() {
  if (!scene.value || !calibrationDraft.value || calibrationLoading.value) return
  calibrationLoading.value = true
  try {
    calibrationDraft.value = await api.previewPitchCalibration(
      scene.value.id,
      calibrationDraft.value.sceneTime,
      calibrationPreset.value,
      calibrationDraft.value.anchors,
    )
    saveState.value = calibrationDraft.value.alignmentError === null
      ? 'Pitch overlay updated'
      : `Pitch overlay · ${calibrationDraft.value.alignmentError.toFixed(1)} px error`
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not update pitch overlay'
  } finally {
    calibrationLoading.value = false
  }
}

function finishCalibrationAnchorDrag(event: PointerEvent) {
  if (!draggedCalibrationAnchor.value) return
  updateDraggedCalibrationAnchor(event)
  draggedCalibrationAnchor.value = null
  try {
    if (calibrationOverlay.value?.hasPointerCapture(event.pointerId)) {
      calibrationOverlay.value.releasePointerCapture(event.pointerId)
    }
  } catch {
    // The drag is already complete when the browser releases capture first.
  }
  void refreshPitchCalibration()
}

function nudgeCalibrationAnchor(event: KeyboardEvent, anchorId: string) {
  const draft = calibrationDraft.value
  const anchor = draft?.anchors.find((item) => item.id === anchorId)
  if (!draft || !anchor) return
  const amount = event.shiftKey ? 16 : 4
  const movement = {
    ArrowLeft: [-amount, 0],
    ArrowRight: [amount, 0],
    ArrowUp: [0, -amount],
    ArrowDown: [0, amount],
  }[event.key]
  if (!movement) return
  anchor.image.x = Math.max(0, Math.min(draft.frameWidth, anchor.image.x + movement[0]))
  anchor.image.y = Math.max(0, Math.min(draft.frameHeight, anchor.image.y + movement[1]))
  void refreshPitchCalibration()
}

function closePitchCalibration() {
  calibrationDraft.value = null
  draggedCalibrationAnchor.value = null
  saveState.value = 'Pitch calibration cancelled'
}

async function applyPitchCalibration() {
  if (!scene.value || !calibrationDraft.value || calibrationApplying.value) return
  const sceneId = scene.value.id
  calibrationApplying.value = true
  reconstructing.value = true
  invalidateIdentityReview()
  clearStaleFrameAnalysis()
  saveState.value = 'Applying pitch calibration and rebuilding tracks…'
  try {
    const updated = await api.applyPitchCalibration(
      sceneId,
      calibrationDraft.value.sceneTime,
      calibrationPreset.value,
      calibrationDraft.value.anchors,
    )
    if (scene.value?.id !== sceneId) return
    scene.value = updated
    calibrationDraft.value = null
    selectedTrackId.value = null
    selectedCanonicalPersonId.value = null
    await pollReconstruction(sceneId)
  } catch (cause) {
    reconstructing.value = false
    error.value = cause instanceof Error ? cause.message : 'Could not apply pitch calibration'
  } finally {
    calibrationApplying.value = false
  }
}

async function calibrateQaFrame(sceneTime: number) {
  seekTo(sceneTime)
  await openPitchCalibration()
}

async function changeAttackingGoal(event: Event | 'left' | 'right') {
  const side = typeof event === 'string'
    ? event
    : (event.target as HTMLSelectElement).value as 'left' | 'right'
  if (!scene.value || !['left', 'right'].includes(side) || pitchSideSaving.value) return
  pitchSideSaving.value = true
  playing.value = false
  sourceVideo.value?.pause()
  try {
    const canonicalSelection = selectedCanonicalPersonId.value
    scene.value = await api.setAttackingGoal(scene.value.id, side)
    clearStaleFrameAnalysis()
    calibrationDraft.value = null
    const canonicalStillExists = canonicalSelection
      ? scene.value.payload.canonicalPeople?.some(
        (person) => person.canonicalPersonId === canonicalSelection,
      ) === true
      : false
    selectedTrackId.value = canonicalStillExists
      ? scene.value.payload.tracks.find(
        (track) => track.canonicalPersonId === canonicalSelection,
      )?.id ?? null
      : scene.value.payload.tracks.some((track) => track.id === selectedTrackId.value)
        ? selectedTrackId.value
        : scene.value.payload.tracks[0]?.id ?? null
    selectedCanonicalPersonId.value = canonicalStillExists
      ? canonicalSelection
      : scene.value.payload.tracks.find((track) => track.id === selectedTrackId.value)?.canonicalPersonId ?? null
    saveState.value = `Attack direction set to ${side} · calibration unchanged`
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not change pitch side'
  } finally {
    pitchSideSaving.value = false
  }
}

function confidenceFor(track: Track) {
  return Math.round(interpolateKeyframes(track.keyframes, currentTime.value).confidence * 100)
}

function trackQualityFor(track: Track) {
  const observed = track.keyframes.filter(
    (keyframe) => keyframe.observed !== false && keyframe.confidence > 0.12,
  )
  if (!observed.length) return 0
  return Math.round(
    observed.reduce((total, keyframe) => total + keyframe.confidence, 0) / observed.length * 100,
  )
}

function onKeydown(event: KeyboardEvent) {
  if ((event.target as HTMLElement)?.matches('input, select, textarea, button, [role="button"]')) return
  if (event.code === 'Space') {
    event.preventDefault()
    togglePlay()
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
    event.preventDefault()
    void saveScene()
  }
}

onMounted(async () => {
  const savedViewPreferences = loadThreeViewPreferences(window.localStorage)
  if (savedViewPreferences) {
    viewOptions.value = savedViewPreferences.options
    renderQuality.value = savedViewPreferences.renderQuality
  }
  await loadWorkspace()
  await nextTick()
  animationFrame = requestAnimationFrame(tick)
  window.addEventListener('keydown', onKeydown)
})

onBeforeUnmount(() => {
  frameAnalysisRequestId += 1
  cancelAnimationFrame(animationFrame)
  window.clearTimeout(reconstructionTimer)
  videoReviewResizeObserver?.disconnect()
  window.removeEventListener('keydown', onKeydown)
})

watch(videoReviewViewport, (viewport) => {
  videoReviewResizeObserver?.disconnect()
  videoReviewResizeObserver = null
  if (!viewport) return
  videoReviewResizeObserver = new ResizeObserver(() => {
    commitVideoReviewTransform(videoReviewTransform.value)
  })
  videoReviewResizeObserver.observe(viewport)
})

watch([viewOptions, renderQuality], ([options, quality]) => {
  saveThreeViewPreferences(window.localStorage, {
    options: { ...options },
    renderQuality: quality,
  })
}, { deep: true })

watch(currentTime, (time) => {
  if (
    selectedBallKeyframeTime.value !== null
    && Math.abs(selectedBallKeyframeTime.value - time) > 0.0011
  ) selectedBallKeyframeTime.value = null
  const desiredTime = sourceStart.value + canonicalToPassTime(time)
  if (!playing.value && sourceVideo.value && Math.abs(sourceVideo.value.currentTime - desiredTime) > 0.08) {
    sourceVideo.value.currentTime = desiredTime
  }
})

watch(selectedActionActorId, () => {
  selectedPlayerActionId.value = null
})

watch(() => scene.value?.id, () => {
  frameAnalysisRequestId += 1
  activeFrameAnalysisRequest = null
  frameAnalyzing.value = false
  if (!sceneVideo.value) viewMode.value = '3d'
  trackQuery.value = ''
  resetVideoReviewView()
  activeTab.value = 'binding'
  frameAnalysis.value = null
  selectedFramePersonId.value = null
  ballSelected.value = false
  ballEditMode.value = false
  ballTrajectorySaving.value = false
  selectedBallKeyframeTime.value = null
  selectedPlayerActionId.value = null
  playerActionSaving.value = false
  manualRosterImportError.value = null
  frameDetectionHitCycle = null
  suppressNextFrameOverlayClick = false
  frameAnnotationMode.value = false
  frameAnnotationDraft.value = null
  frameAnnotationDrag.value = null
  calibrationDraft.value = null
  multiPassSelection.value = []
  timelineGroupEditing.value = false
  activePassSceneId.value = multiPassAnalysis.value?.referenceSceneId ?? null
  selectedReconstructionModel.value = sceneVideo.value?.reconstruction?.model ?? 'yolo26m.pt'
  selectedBallBackend.value = sceneVideo.value?.reconstruction?.ballBackend ?? 'dedicated-ultralytics'
})

watch(() => multiPassAnalysis.value?.referenceSceneId, (referenceSceneId) => {
  if (!activePassSceneId.value && referenceSceneId) activePassSceneId.value = referenceSceneId
})

watch(activePassSceneId, resetVideoReviewView)
</script>

<template>
  <main class="app-shell">
    <header class="topbar">
      <div class="brand-block">
        <div class="brand-mark"><span>R</span></div>
        <div>
          <p class="eyebrow">Interactive football lab</p>
          <h1>Replay Studio <span>α</span></h1>
        </div>
      </div>

      <div v-if="scene" class="moment-title">
        <input v-model="scene.title" aria-label="Moment title" @input="saveState = 'Unsaved changes'" />
      </div>

      <div class="top-actions">
        <span class="save-state">{{ saveState }}</span>
        <button class="button import-button" @click="videoIngestOpen = true">＋ Import clip</button>
        <button class="button ghost" :disabled="reconstructionMutationLocked || matchSnapshotRefreshing || manualRosterImporting" @click="openMatchSettings">Match settings</button>
        <button class="button primary" :disabled="saving || reconstructing || reconstructionRunning || frameAnnotationSaving || rosterBindingSaving || identityDecisionSaving || matchSnapshotRefreshing || manualRosterImporting" @click="saveScene">
          {{ saving ? 'Saving' : 'Save moment' }}
        </button>
      </div>
    </header>

    <section v-if="error && !scene" class="fatal-state">
      <div class="fatal-card">
        <span class="fatal-code">API OFFLINE</span>
        <h2>The studio could not reach its workspace.</h2>
        <p>{{ error }}</p>
        <code>uvicorn app.main:app --app-dir apps/api</code>
        <button class="button primary" @click="loadWorkspace">Try again</button>
      </div>
    </section>

    <section v-else-if="scene" class="studio-grid">
      <aside class="panel left-panel">
        <div class="panel-section scene-switcher">
          <div class="section-heading">
            <span>Video projects</span>
          </div>
          <select :value="activeProjectId ?? ''" aria-label="Video project" @change="switchScene(($event.target as HTMLSelectElement).value)">
            <option v-for="item in projects" :key="item.id" :value="item.id">{{ item.title }}</option>
          </select>
          <div v-if="internalSceneLabel && activeProjectId" class="project-context">
            <span>{{ internalSceneLabel }}</span>
            <strong>{{ scene.title }}</strong>
            <button @click="switchScene(activeProjectId)">← Back to full timeline</button>
          </div>
        </div>

        <div class="panel-section teams-card">
          <p class="section-label">Project match roster</p>
          <div class="score-row">
            <div>
              <i :style="{ background: projectMatchTeams.home.color }" />
              <span>{{ projectMatchTeams.home.name }}</span>
            </div>
            <strong>VS</strong>
            <div>
              <span>{{ projectMatchTeams.away.name }}</span>
              <i :style="{ background: projectMatchTeams.away.color }" />
            </div>
          </div>
          <small>{{ projectMatchContext.label }}</small>
          <small v-if="scene.payload.matchBinding">Data provider · {{ boundMatchProviderLabel }}</small>
          <div class="match-snapshot-refresh" role="group" aria-label="Project match roster tools">
            <small>Applies to the full timeline and every shot in this video project.</small>
            <small v-if="matchSnapshotRefreshAvailable">
              {{ scene.payload.matchBinding?.schemaVersion === 2 ? 'Saved roster is partial.' : 'Legacy match binding has no offline roster snapshot.' }}
            </small>
            <div class="match-snapshot-buttons">
              <button
                v-if="matchSnapshotRefreshAvailable"
                type="button"
                :disabled="matchSnapshotRefreshing || manualRosterImporting || reconstructionMutationLocked"
                @click="refreshMatchSnapshot"
              >{{ matchSnapshotRefreshing ? 'Refreshing…' : 'Refresh project roster' }}</button>
              <button
                type="button"
                :disabled="manualRosterImporting || matchSnapshotRefreshing || reconstructionMutationLocked"
                @click="chooseManualRosterFile"
              >{{ manualRosterImporting ? 'Importing…' : 'Import project roster JSON' }}</button>
            </div>
            <input
              ref="manualRosterFileInput"
              hidden
              type="file"
              accept=".json,application/json"
              aria-label="Choose roster JSON file"
              @change="importManualRosterFile"
            />
            <small>Example format: <code>data/matches/spain-belgium-2026-qf.json</code>. Choose the file explicitly; it updates the whole video project.</small>
            <small v-if="manualRosterImportError" class="match-import-error" role="alert">{{ manualRosterImportError }}</small>
          </div>
        </div>

        <div class="tracks-header">
          <span>Tracked objects</span>
          <span>{{ scene.payload.tracks.length + canonicalPeopleWithoutRender.length + 1 }}</span>
        </div>
        <div v-if="scene.payload.tracks.length || canonicalPeopleWithoutRender.length" class="track-search" role="search" aria-label="Tracked objects">
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" /><path d="m15.5 15.5 5 5" /></svg>
          <input v-model="trackQuery" type="search" aria-label="Search tracked objects" placeholder="Find player, number or track…" />
          <button v-if="trackQuery" type="button" aria-label="Clear tracked object search" @click="trackQuery = ''">×</button>
        </div>
        <div class="track-list">
          <button
            v-for="track in filteredTracks"
            :key="track.id"
            class="track-row"
            :class="{ selected: selectedTrackId === track.id }"
            @click="selectTrack(track.id)"
          >
            <span class="jersey-dot" :style="{ background: track.color }">{{ track.number }}</span>
            <span class="track-copy">
              <strong>{{ track.label }}</strong>
              <small>{{ track.externalPlayerId ? 'Linked player' : 'Unbound track' }}</small>
            </span>
            <span
              class="confidence"
              :class="{ weak: trackQualityFor(track) < 85 }"
              title="Average confidence across observed frames"
            >{{ trackQualityFor(track) }}</span>
          </button>
          <button
            v-for="identity in filteredCanonicalPeopleWithoutRender"
            :key="identity.canonicalPersonId"
            class="track-row identity-only-row"
            :class="{ selected: selectedCanonicalPersonId === identity.canonicalPersonId && !selectedTrackId }"
            @click="selectCanonicalPerson(identity.canonicalPersonId)"
          >
            <span class="jersey-dot">{{ identity.jerseyNumber || '?' }}</span>
            <span class="track-copy">
              <strong>{{ identity.displayName }}</strong>
              <small>Canonical identity · not projected in 3D</small>
            </span>
            <span class="confidence" :class="{ weak: (identity.identityConfidence ?? 0) < .85 }">{{ identity.identityConfidence == null ? '—' : Math.round(identity.identityConfidence * 100) }}</span>
          </button>
          <p v-if="trackQuery && !filteredTracks.length && !filteredCanonicalPeopleWithoutRender.length && !ballMatchesTrackQuery" class="track-search-empty">No tracked objects match “{{ trackQuery }}”.</p>
          <button
            v-if="ballMatchesTrackQuery"
            class="track-row ball-row"
            :class="{ selected: ballSelected }"
            type="button"
            @click="selectBallObject"
          >
            <span class="ball-icon">●</span>
            <span class="track-copy">
              <strong>Match ball</strong>
              <small>{{ ballTrajectoryMode === 'manual' ? 'Manual' : 'Automatic' }} · {{ scene.payload.ball.keyframes.length }} keypoints</small>
            </span>
          </button>
        </div>
      </aside>

      <section
        class="stage-column"
        :class="{
          'has-segment-map': segmentLayout && sceneVideo?.segments?.length,
          'has-ball-timeline': ballTrajectoryMode === 'manual',
          'has-player-action-timeline': showPlayerActionTimeline,
        }"
      >
        <div class="stage-toolbar">
          <div class="stage-view-controls">
            <label class="toolbar-select camera-selector">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 7.5h10.5v9H4zM14.5 10l5-2.5v9l-5-2.5" />
              </svg>
              <span>Camera</span>
              <select :value="activeCamera" aria-label="3D camera preset" :disabled="viewMode === 'video' || Boolean(calibrationDraft)" @change="onCameraPresetChange">
                <option value="broadcast">Broadcast</option>
                <option value="orbit">Orbit</option>
                <option value="tactical">Tactical</option>
                <option value="goal">Goal line</option>
              </select>
            </label>
            <label v-if="sceneVideo" class="toolbar-select layout-selector">
              <span>Layout</span>
              <select v-model="viewMode" aria-label="Workspace layout" :disabled="Boolean(calibrationDraft)">
                <option value="split">Video + 3D</option>
                <option value="3d">3D only</option>
                <option value="video">Video only</option>
              </select>
            </label>
          </div>
          <div class="stage-tools">
            <label v-if="multiPassAnalysis?.status === 'ready' && activePass" class="angle-switcher">
              <span>Source</span>
              <select :value="activePass.sceneId" aria-label="Replay angle" @change="chooseSourcePass">
                <option v-for="item in multiPassAnalysis.passes.filter((pass) => pass.status === 'ready')" :key="item.sceneId" :value="item.sceneId">
                  {{ item.label }} · {{ passRelationLabel(item.relation) }}
                </option>
              </select>
            </label>
            <span v-if="multiPassAnalysis" class="multi-pass-badge">{{ multiPassAnalysis.status }} · {{ multiPassAnalysis.selectedSegmentIds.length }} angles</span>
            <button
              v-if="sceneVideo?.selectedSegmentId"
              class="tool-toggle frame-analysis-toggle primary-tool"
              :class="{ active: Boolean(activeFrameAnalysis) }"
              :disabled="frameAnalyzing || reconstructing || reconstructionStatus === 'processing' || reconstructionStatus === 'queued'"
              @click="analyzeCurrentFrame"
            >
              {{ frameAnalyzing ? 'Reading frame…' : 'Analyze frame' }}
            </button>
            <ToolbarDisclosure
              v-if="sceneVideo?.selectedSegmentId && !multiPassAnalysis"
              label="Reconstruction"
              :active="reconstructionRunning || activeTab === 'qa' || Boolean(calibrationDraft)"
            >
              <template #icon>
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h14v14H5zM8 8h8v8H8zM3 9h2m14 0h2M3 15h2m14 0h2M9 3v2m6-2v2M9 19v2m6-2v2" /></svg>
              </template>
              <template #default="{ closeMenu }">
                <div class="reconstruction-menu-content">
                  <div class="reconstruction-menu-heading">
                    <div><span>Scene reconstruction</span><strong>{{ scene.payload.tracks.length }} tracked objects</strong></div>
                    <b :class="`verdict-${reconstructionQualityVerdict}`">{{ reconstructionRunning ? `${reconstructionProgress?.overallPercent ?? 0}%` : reconstructionQualityVerdict.toUpperCase() }}</b>
                  </div>
                  <label class="reconstruction-model-field">
                    <span>People detector</span>
                    <select
                      v-model="selectedReconstructionModel"
                      aria-label="Reconstruction model"
                      :disabled="reconstructing || reconstructionRunning"
                    >
                      <option v-for="item in reconstructionModels" :key="item.value" :value="item.value">{{ item.label }}</option>
                    </select>
                  </label>
                  <label class="reconstruction-model-field">
                    <span>Ball detector</span>
                    <select
                      v-model="selectedBallBackend"
                      aria-label="Ball detection backend"
                      :disabled="reconstructing || reconstructionRunning"
                    >
                      <option v-for="item in ballDetectionBackends" :key="item.value" :value="item.value">{{ item.label }}</option>
                    </select>
                  </label>
                  <label class="reconstruction-model-field">
                    <span>Ball trajectory</span>
                    <select
                      :value="ballTrajectoryMode"
                      aria-label="Ball trajectory mode"
                      :disabled="ballTrajectorySaving || reconstructing || reconstructionRunning"
                      @change="changeBallTrajectoryMode"
                    >
                      <option value="automatic">Automatic detection</option>
                      <option value="manual">Manual keypoints</option>
                    </select>
                  </label>
                  <button class="reconstruction-menu-primary" :disabled="reconstructing || reconstructionRunning" @click="closeMenu(); reconstructCurrentScene()">
                    {{ reconstructionRunning ? `Analyzing · ${reconstructionProgress?.overallPercent ?? 0}%` : scene.payload.tracks.length ? 'Reconstruct scene' : 'Build scene' }}
                  </button>
                  <div class="reconstruction-menu-actions">
                    <button :class="{ active: Boolean(modelComparison) }" :disabled="modelComparing || reconstructing || reconstructionRunning" @click="closeMenu(); compareCurrentSceneModels()">
                      <span>Compare detection models</span><small>{{ modelComparing ? 'Comparing 26n and 26m…' : modelComparison ? 'Result ready' : '26n versus 26m' }}</small>
                    </button>
                    <button :class="{ active: Boolean(calibrationDraft) }" :disabled="calibrationLoading || calibrationApplying || reconstructing || reconstructionRunning" @click="closeMenu(); calibrationDraft ? closePitchCalibration() : openPitchCalibration()">
                      <span>{{ calibrationDraft ? 'Close calibration' : 'Calibrate current frame' }}</span><small>Inspect pitch geometry and anchors</small>
                    </button>
                    <button v-if="sceneVideo.reconstruction" @click="closeMenu(); activeTab = 'qa'">
                      <span>Open calibration quality</span><small>View {{ visiblePitchSide.toUpperCase() }} · attack {{ attackingGoalSide.toUpperCase() }}</small>
                    </button>
                  </div>
                </div>
              </template>
            </ToolbarDisclosure>
            <ThreeViewMenu
              v-model="viewOptions"
              v-model:render-quality="renderQuality"
              :disabled="Boolean(calibrationDraft)"
            />
          </div>
        </div>

        <div class="viewport-wrap" :class="{ 'split-view': viewMode === 'split' && sceneVideo && !calibrationDraft, 'video-only': viewMode === 'video' && sceneVideo && !calibrationDraft, calibrating: Boolean(calibrationDraft) }">
          <section v-if="reconstructionRunning" class="analysis-progress-panel" aria-live="polite" aria-label="Video analysis progress">
            <div class="analysis-progress-heading">
              <div>
                <span>ANALYSIS PIPELINE · PHASE {{ reconstructionProgress?.phaseIndex ?? 1 }} OF {{ reconstructionProgress?.phaseCount ?? reconstructionPhases.length }}</span>
                <strong>{{ reconstructionProgress?.label ?? 'Waiting to start' }}</strong>
                <small>{{ reconstructionProgress?.detail ?? `Queued ${analysisFrameCount} sampled frames for analysis.` }}</small>
              </div>
              <b>{{ reconstructionProgress?.overallPercent ?? 0 }}%</b>
            </div>
            <div class="analysis-overall-track" aria-hidden="true">
              <i :style="{ width: `${reconstructionProgress?.overallPercent ?? 0}%` }" />
            </div>
            <div class="analysis-progress-meta">
              <span v-if="reconstructionProgress?.total">{{ reconstructionProgress.completed }} / {{ reconstructionProgress.total }} in current phase</span>
              <span>{{ formatAnalysisDuration(reconstructionProgress?.elapsedSeconds ?? 0) }} elapsed</span>
              <span>{{ reconstructionProgress?.etaSeconds === null || reconstructionProgress?.etaSeconds === undefined ? 'Estimating remaining time…' : `≈ ${formatAnalysisDuration(reconstructionProgress.etaSeconds)} remaining` }}</span>
            </div>
            <ol class="analysis-phase-list">
              <li v-for="(phase, index) in reconstructionPhases" :key="phase.id" :class="phase.status">
                <i aria-hidden="true">{{ phase.status === 'completed' ? '✓' : index + 1 }}</i>
                <span><strong>{{ phase.label }}</strong><small>{{ progressStatusLabel(phase.status) }}</small></span>
              </li>
            </ol>
          </section>
          <div v-if="sceneVideo && (viewMode !== '3d' || calibrationDraft)" class="reference-pane">
            <div
              ref="videoReviewViewport"
              class="video-review-viewport"
              :class="{ pannable: videoReviewTransform.scale > VIDEO_REVIEW_MIN_SCALE, panning: Boolean(videoReviewPanDrag) }"
              tabindex="0"
              role="region"
              aria-label="Video review frame. Use plus and minus to zoom, arrow keys to pan, and zero or Home to reset."
              aria-keyshortcuts="+ - 0 Home ArrowLeft ArrowRight ArrowUp ArrowDown"
              @wheel.prevent="onVideoReviewWheel"
              @pointerdown="startVideoReviewPan"
              @pointermove="updateVideoReviewPan"
              @pointerup="finishVideoReviewPan"
              @pointercancel="finishVideoReviewPan"
              @keydown="onVideoReviewKeydown"
            >
              <div class="video-review-transform" :style="videoReviewTransformStyle">
                <video
                  ref="sourceVideo"
                  :src="sceneVideo.mediaUrl"
                  :poster="sceneVideo.posterUrl"
                  muted
                  playsinline
                  preload="auto"
                  @loadedmetadata="seekTo(currentTime)"
                />
                <VideoPathTrackingOverlay
                  :enabled="viewOptions.pathTracking && !calibrationDraft && activeTab !== 'qa' && videoPathUsesReferenceCamera"
                  :keyframes="selectedPathKeyframes"
                  :projection-context="videoPathProjectionContext"
                  :current-time="currentTime"
                  :subject-kind="selectedPathSubject?.kind"
                  :color="selectedPathSubject?.color"
                  :subject-label="selectedPathSubject?.label"
                />
                <svg
              v-if="activeCalibrationDraft"
              ref="calibrationOverlay"
              class="pitch-calibration-overlay"
              :viewBox="`0 0 ${activeCalibrationDraft.frameWidth} ${activeCalibrationDraft.frameHeight}`"
              preserveAspectRatio="xMidYMid meet"
              aria-label="Pitch calibration overlay"
              @click.stop
              @pointermove.stop.prevent="updateDraggedCalibrationAnchor"
              @pointerup.stop.prevent="finishCalibrationAnchorDrag"
              @pointercancel.stop.prevent="finishCalibrationAnchorDrag"
            >
              <polyline
                v-for="marking in activeCalibrationDraft.markings"
                :key="marking.id"
                class="calibration-marking"
                :class="marking.kind"
                :points="marking.points.map((point) => `${point.x},${point.y}`).join(' ')"
              />
              <polyline
                v-for="(line, index) in activeCalibrationDiagnostics?.lines ?? []"
                :key="`source-line-${line.id ?? index}`"
                class="calibration-source-line"
                :class="{ accepted: line.accepted ?? line.inlier, rejected: line.accepted === false || line.inlier === false }"
                :points="calibrationRawLinePoints(line).map((point) => `${point.x},${point.y}`).join(' ')"
              />
              <line
                v-if="activeCalibrationDraft.horizon"
                class="calibration-horizon"
                :x1="activeCalibrationDraft.horizon.start.x"
                :y1="activeCalibrationDraft.horizon.start.y"
                :x2="activeCalibrationDraft.horizon.end.x"
                :y2="activeCalibrationDraft.horizon.end.y"
              />
              <g
                v-for="(point, index) in activeCalibrationDiagnostics?.points ?? []"
                :key="`detected-${point.id ?? index}`"
                class="calibration-detected-keypoint"
                :class="{ inlier: point.inlier, outlier: point.inlier === false }"
              >
                <line
                  v-if="projectedEvidencePoint(point)"
                  :x1="point.image.x"
                  :y1="point.image.y"
                  :x2="projectedEvidencePoint(point)!.x"
                  :y2="projectedEvidencePoint(point)!.y"
                />
                <circle class="source" :cx="point.image.x" :cy="point.image.y" r="6" />
                <circle
                  v-if="projectedEvidencePoint(point)"
                  class="projected"
                  :cx="projectedEvidencePoint(point)!.x"
                  :cy="projectedEvidencePoint(point)!.y"
                  r="4"
                />
                <text :x="point.image.x + 9" :y="point.image.y - 8">
                  {{ point.label || `KP ${index + 1}` }}<template v-if="calibrationPointResidual(point) !== null"> · {{ calibrationPointResidual(point)!.toFixed(1) }}px</template>
                </text>
              </g>
              <g
                v-for="(anchor, index) in activeCalibrationDraft.anchors"
                :key="anchor.id"
                class="calibration-anchor"
                @pointerdown.stop.prevent="startCalibrationAnchorDrag($event, anchor.id)"
              >
                <circle
                  :cx="anchor.image.x"
                  :cy="anchor.image.y"
                  r="13"
                  role="button"
                  tabindex="0"
                  :aria-label="`Calibration anchor ${index + 1}: ${anchor.label}`"
                  @keydown.prevent="nudgeCalibrationAnchor($event, anchor.id)"
                />
                <text :x="anchor.image.x" :y="anchor.image.y + 4">{{ index + 1 }}</text>
                <text class="anchor-label" :x="anchor.image.x + 18" :y="anchor.image.y - 15">{{ anchor.label }}</text>
              </g>
            </svg>
            <svg
              v-else-if="activeCalibrationQaFrame && calibrationQaMarkings.length"
              class="calibration-qa-overlay"
              :class="activeCalibrationQaFrame.status"
              :viewBox="`0 0 ${calibrationQaFrameSize.width} ${calibrationQaFrameSize.height}`"
              preserveAspectRatio="xMidYMid meet"
              aria-label="Exact stored calibration evidence overlay"
              @click.stop
            >
              <polyline
                v-for="marking in calibrationQaMarkings"
                :key="marking.id"
                class="calibration-qa-marking"
                :class="marking.kind"
                :points="marking.points.map((point) => `${point.x},${point.y}`).join(' ')"
              />
              <g
                v-for="(point, index) in activeCalibrationQaFrame.keypoints ?? []"
                :key="point.id ?? index"
                class="calibration-qa-keypoint"
                :class="{ inlier: point.inlier, outlier: point.inlier === false }"
              >
                <circle :cx="point.image.x" :cy="point.image.y" r="6" />
                <line v-if="projectedEvidencePoint(point)" :x1="point.image.x" :y1="point.image.y" :x2="projectedEvidencePoint(point)!.x" :y2="projectedEvidencePoint(point)!.y" />
                <circle v-if="projectedEvidencePoint(point)" class="projected" :cx="projectedEvidencePoint(point)!.x" :cy="projectedEvidencePoint(point)!.y" r="4" />
              </g>
              <g class="calibration-qa-axis">
                <line x1="24" :y1="calibrationQaFrameSize.height - 28" x2="96" :y2="calibrationQaFrameSize.height - 28" />
                <path :d="`M 96 ${calibrationQaFrameSize.height - 28} l -12 -7 l 0 14 z`" />
                <text x="24" :y="calibrationQaFrameSize.height - 39">PITCH X · LEFT → RIGHT</text>
              </g>
            </svg>
            <svg
              v-if="activeFrameAnalysis && !calibrationDraft && activeTab !== 'qa'"
              ref="frameAnalysisOverlay"
              class="frame-analysis-overlay"
              :class="{ labeling: frameAnnotationMode }"
              :viewBox="`0 0 ${activeFrameAnalysis.frameWidth} ${activeFrameAnalysis.frameHeight}`"
              preserveAspectRatio="xMidYMid meet"
              aria-label="Current frame detections"
              @pointerdown="startFrameAnnotationDrag"
              @pointermove="updateFrameAnnotationDrag"
              @pointerup="finishFrameAnnotationDrag"
              @pointercancel="finishFrameAnnotationDrag"
              @click="selectFramePersonAtPoint"
            >
              <g
                v-for="person in activeFrameAnalysis.people"
                :key="person.id"
                class="frame-person-box"
                :class="{ matched: framePersonCanonicalId(person) || person.matchedTrackId, selected: person.id === selectedFramePersonId, manual: person.annotationId, confirmed: person.previewState === 'confirmed', merged: person.previewState === 'merged', split: person.previewState === 'split' }"
                @pointerdown.stop
              >
                <rect
                  :x="person.bbox.x"
                  :y="person.bbox.y"
                  :width="person.bbox.width"
                  :height="person.bbox.height"
                  :stroke="framePersonCanonicalId(person) || person.matchedTrackId ? '#71e2aa' : '#ff8f63'"
                  role="button"
                  tabindex="0"
                  :aria-pressed="person.id === selectedFramePersonId"
                  :aria-label="`${framePersonLabel(person)}, ${Math.round(person.confidence * 100)} percent, ${framePersonSelectionDescription(person)}`"
                  @pointerdown.stop
                  @keydown.enter.prevent.stop="selectDetectedPerson(person)"
                  @keydown.space.prevent.stop="selectDetectedPerson(person)"
                />
                <rect
                  class="jersey-swatch"
                  :x="person.bbox.x"
                  :y="Math.max(1, person.bbox.y - 16)"
                  width="12"
                  height="12"
                  :fill="person.jerseyColor"
                />
                <text :x="person.bbox.x + 16" :y="Math.max(11, person.bbox.y - 6)">
                  {{ framePersonLabel(person) }} · {{ person.previewState === 'merged' ? `MERGED → ${person.mergeTargetId}` : person.previewState === 'split' ? `SPLIT [${person.rangeStart?.toFixed(2)}, ${person.rangeEnd?.toFixed(2)})` : person.previewState === 'confirmed' ? 'CONFIRMED' : `${Math.round(person.confidence * 100)}%` }}
                </text>
                <text v-if="person.id === selectedFramePersonId" class="pitch-position" :class="{ uncertain: frameMetricBadge(person) === 'UNCERTAIN' }" :x="person.bbox.x" :y="person.bbox.y + person.bbox.height + 12">
                  {{ frameMetricBadge(person) === 'UNCERTAIN' ? '3D position uncertain' : `x ${person.pitch.x.toFixed(1)} · z ${person.pitch.z.toFixed(1)}` }}
                </text>
              </g>
              <g
                v-for="annotation in activeFrameAnalysis.annotations.filter((item) => annotationIdentityAction(item) === 'exclude')"
                :key="annotation.id"
                class="frame-ignore-box"
                @pointerdown.stop
                @click.stop="selectFrameAnnotation(annotation)"
              >
                <rect
                  :x="annotation.bbox.x"
                  :y="annotation.bbox.y"
                  :width="annotation.bbox.width"
                  :height="annotation.bbox.height"
                />
                <path
                  :d="`M ${annotation.bbox.x} ${annotation.bbox.y} L ${annotation.bbox.x + annotation.bbox.width} ${annotation.bbox.y + annotation.bbox.height} M ${annotation.bbox.x + annotation.bbox.width} ${annotation.bbox.y} L ${annotation.bbox.x} ${annotation.bbox.y + annotation.bbox.height}`"
                />
                <text :x="annotation.bbox.x" :y="Math.max(14, annotation.bbox.y - 7)">EXCLUDED</text>
              </g>
              <g
                v-for="annotation in activeFrameAnalysis.annotations.filter((item) => annotationIdentityAction(item) === 'split')"
                :key="annotation.id"
                class="frame-split-box"
                @pointerdown.stop
                @click.stop="selectFrameAnnotation(annotation)"
              >
                <rect
                  :x="annotation.bbox.x"
                  :y="annotation.bbox.y"
                  :width="annotation.bbox.width"
                  :height="annotation.bbox.height"
                />
                <line
                  :x1="annotation.bbox.x + annotation.bbox.width / 2"
                  :y1="annotation.bbox.y - 5"
                  :x2="annotation.bbox.x + annotation.bbox.width / 2"
                  :y2="annotation.bbox.y + annotation.bbox.height + 5"
                />
                <text :x="annotation.bbox.x" :y="Math.max(14, annotation.bbox.y - 7)">
                  SPLIT · {{ annotation.rangeStart?.toFixed(2) }}–{{ annotation.rangeEnd?.toFixed(2) }}s
                </text>
              </g>
              <rect
                v-if="frameAnnotationMode && frameAnnotationDraft"
                class="frame-annotation-draft"
                :x="frameAnnotationDraft.bbox.x"
                :y="frameAnnotationDraft.bbox.y"
                :width="frameAnnotationDraft.bbox.width"
                :height="frameAnnotationDraft.bbox.height"
              />
              <g
                v-for="ball in activeFrameAnalysis.ballCandidates"
                :key="ball.id"
                class="frame-ball-candidate"
                :class="{ primary: ball.primary }"
                @click.stop
              >
                <circle :cx="ball.image.x" :cy="ball.image.y" :r="ball.primary ? 10 : 7" />
                <text v-if="ball.primary" :x="ball.image.x + 12" :y="ball.image.y - 8">BALL {{ Math.round(ball.confidence * 100) }}%</text>
              </g>
                </svg>
              </div>
            </div>
            <div class="video-review-controls" role="group" aria-label="Video review zoom controls">
              <button
                type="button"
                aria-label="Zoom out video review"
                :disabled="videoReviewTransform.scale <= VIDEO_REVIEW_MIN_SCALE"
                @click="adjustVideoReviewZoom(-0.25)"
              >−</button>
              <output aria-live="polite" aria-label="Video review zoom">{{ videoReviewZoomPercent }}%</output>
              <button
                type="button"
                aria-label="Zoom in video review"
                :disabled="videoReviewTransform.scale >= VIDEO_REVIEW_MAX_SCALE"
                @click="adjustVideoReviewZoom(0.25)"
              >+</button>
              <button
                type="button"
                class="reset"
                aria-label="Reset video review zoom and pan"
                :disabled="videoReviewTransform.scale === VIDEO_REVIEW_MIN_SCALE && videoReviewTransform.x === 0 && videoReviewTransform.y === 0"
                @click="resetVideoReviewView"
              >Reset</button>
            </div>
            <div
              v-if="videoSelectionStatus && !calibrationDraft && activeTab !== 'qa'"
              class="video-selection-status"
              :class="videoSelectionStatus.state"
              role="status"
              aria-live="polite"
            >
              <i aria-hidden="true" />
              <span>
                <strong>{{ videoSelectionStatus.label }}</strong>
                <small>{{ videoSelectionStatus.detail }}</small>
              </span>
            </div>
            <PathTrackingLegend
              v-if="!calibrationDraft"
              :enabled="viewOptions.pathTracking"
              :subject-kind="selectedPathSubject?.kind"
              :subject-label="selectedPathSubject?.label"
              :subject-color="selectedPathSubject?.color"
              :sample-count="selectedPathSubject?.sampleCount"
              :has-drawable-path="selectedPathSegments.length > 0"
              :unavailable-label="unavailablePathSubjectLabel"
              :surface-unavailable-reason="videoPathUnavailableReason"
              :surface-note="videoPathSurfaceNote"
              align="left"
              top-offset="stacked"
              surface-label="video review"
            />
            <div v-if="calibrationDraft" class="pitch-calibration-panel" :class="{ left: calibrationPreset.endsWith('right') }" @click.stop>
              <div class="calibration-panel-heading">
                <div><span>Current frame calibration</span><strong>{{ calibrationDraft.sceneTime.toFixed(2) }}s · frame {{ calibrationDraft.frameIndex }}</strong></div>
                <i :class="activeCalibrationDiagnostics?.status ?? calibrationDraft.quality">{{ activeCalibrationDiagnostics?.status ?? calibrationDraft.quality }}</i>
              </div>
              <div v-if="activeCalibrationDiagnostics" class="calibration-diagnostics" aria-live="polite">
                <div><span>Method</span><strong>{{ activeCalibrationDiagnostics.method }}</strong></div>
                <div><span>Confidence</span><strong>{{ calibrationPercent(calibrationDraft.confidence) }}</strong></div>
                <div><span>Detected keypoints</span><strong>{{ activeCalibrationDiagnostics.keypointCount ?? '—' }}</strong></div>
                <div><span>Inliers</span><strong>{{ activeCalibrationDiagnostics.inlierCount ?? '—' }}<template v-if="activeCalibrationDiagnostics.inlierRatio !== null"> · {{ calibrationPercent(activeCalibrationDiagnostics.inlierRatio) }}</template></strong></div>
                <div><span>Projected markings · p50</span><strong>{{ calibrationPixels(activeCalibrationDiagnostics.residualP50) }}</strong></div>
                <div><span>Projected markings · p95</span><strong>{{ calibrationPixels(activeCalibrationDiagnostics.residualP95) }}</strong></div>
                <div><span>Projected markings · precision</span><strong>{{ calibrationPercent(activeCalibrationDiagnostics.precision) }}</strong></div>
                <div><span>Projected markings · recall</span><strong>{{ calibrationPercent(activeCalibrationDiagnostics.recall) }}</strong></div>
                <div><span>Projected markings · F1</span><strong>{{ calibrationPercent(activeCalibrationDiagnostics.f1) }}</strong></div>
                <div v-if="activeCalibrationDiagnostics.sourceStatus"><span>Previous rebuild evidence</span><strong>{{ activeCalibrationDiagnostics.sourceStatus }}</strong></div>
              </div>
              <div v-if="activeCalibrationDiagnostics?.lines.length" class="calibration-source-line-list">
                <strong>Observed semantic lines · {{ activeCalibrationDiagnostics.lines.length }}</strong>
                <span v-for="(line, index) in activeCalibrationDiagnostics.lines.slice(0, 8)" :key="`line-label-${line.id ?? index}`">
                  <i :class="{ accepted: line.accepted ?? line.inlier, rejected: line.accepted === false || line.inlier === false }" />
                  <span>{{ calibrationRawLineLabel(line, index) }}</span>
                  <b v-if="(line.confidence !== null && line.confidence !== undefined) || calibrationLineResidualLabel(line)">
                    <template v-if="line.confidence !== null && line.confidence !== undefined">{{ calibrationPercent(line.confidence) }}</template>
                    <em v-if="calibrationLineResidualLabel(line)">{{ calibrationLineResidualLabel(line) }}</em>
                  </b>
                </span>
                <small v-if="activeCalibrationDiagnostics.lines.length > 8">+{{ activeCalibrationDiagnostics.lines.length - 8 }} more lines</small>
              </div>
              <label>
                <span>Visible landmark</span>
                <select :value="calibrationPreset" :disabled="calibrationLoading" aria-label="Pitch landmark preset" @change="changePitchCalibrationPreset">
                  <option v-for="preset in calibrationPresets" :key="preset.value" :value="preset.value">{{ preset.label }}</option>
                </select>
              </label>
              <div v-if="!activeCalibrationDiagnostics" class="calibration-score">
                <span>Projected-markings residual p50</span>
                <strong>{{ calibrationDraft.alignmentError === null ? 'NO SCORE' : `${calibrationDraft.alignmentError.toFixed(1)} px` }}</strong>
              </div>
              <div class="calibration-score">
                <span>Visible pitch side</span>
                <strong :class="{ untrusted: activeCalibrationDiagnostics && !activeCalibrationDiagnostics.visibleSideTrusted }">
                  {{ calibrationVisibleSideLabel(activeCalibrationDiagnostics?.visibleSide, activeCalibrationDiagnostics?.visibleSideTrusted) }}
                </strong>
              </div>
              <div v-if="activeCalibrationDiagnostics?.rejectionReasons.length" class="calibration-rejections">
                <strong>{{ activeCalibrationDiagnostics.status === 'rejected' ? 'Rejection reasons' : 'Source candidate rejection reasons' }}</strong>
                <span v-for="reason in activeCalibrationDiagnostics.rejectionReasons" :key="reason">{{ calibrationRejectionReasonLabel(reason) }}</span>
              </div>
              <p v-if="!activeCalibrationDraft">Move the playhead back to {{ calibrationDraft.sceneTime.toFixed(2) }}s to edit these anchors.</p>
              <p v-else>Yellow lines are the projected pitch. Colored dots are detected source points; red vectors end at their projected positions. Drag numbered anchors to refine this frame manually.</p>
              <small v-for="warning in visibleCalibrationWarnings" :key="warning">{{ warning }}</small>
              <small class="calibration-preview-note">Preview only · diagnostics are recorded, but anchors and tracks stay unchanged until Apply & rebuild.</small>
              <div class="calibration-actions">
                <button :disabled="calibrationLoading" @click="openPitchCalibration(calibrationPreset)">{{ calibrationLoading ? 'Updating…' : 'Calibrate again' }}</button>
                <button v-if="!activeCalibrationDraft" @click="seekTo(calibrationDraft.sceneTime)">Return to frame</button>
                <button class="apply" :disabled="calibrationLoading || calibrationApplying" @click="applyPitchCalibration">{{ calibrationApplying ? 'Applying…' : 'Apply & rebuild' }}</button>
              </div>
            </div>
            <div class="reference-label"><i /> Original clip <template v-if="reconstructionRunning">· {{ reconstructionProgress?.overallPercent ?? 0 }}% · {{ reconstructionProgress?.label ?? 'STARTING ANALYSIS' }}</template><template v-else-if="scene.payload.tracks.length">· AUTO {{ scene.payload.tracks.length }} · {{ calibrationLabel }}</template></div>
            <div class="reference-meta">{{ sceneVideo.filename }} · {{ sceneVideo.fps.toFixed(2) }} FPS</div>
          </div>
          <div v-show="viewMode !== 'video'" class="three-pane">
            <ThreeViewport
              ref="viewport"
              :scene="reconstructionPreviewScene ?? scene"
              :current-time="currentTime"
              :selected-track-id="selectedTrackId"
              :edit-mode="editMode"
              :ball-edit-mode="ballTrajectoryMode === 'manual' && ballEditMode"
              :selected-ball-keyframe-time="selectedBallKeyframeTime"
              :show-models="viewOptions.models"
              :show-trails="viewOptions.trajectory"
              :show-path-tracking="viewOptions.pathTracking"
              :ball-selected="ballSelected"
              :show-labels="viewOptions.labels"
              :show-ball="viewOptions.ball"
              :show-analysis-markers="viewOptions.analysisMarkers"
              :render-quality="renderQuality"
              :frame-analysis="activeFrameAnalysis"
              :active-player-action="selectedTrack ? activePlayerActionPlayback : null"
              @select="selectTrackFromThree"
              @select-ball="selectBallObject"
              @move-track="moveSelected"
              @move-ball="moveManualBall"
            />
            <PathTrackingLegend
              :enabled="viewOptions.pathTracking"
              :subject-kind="selectedPathSubject?.kind"
              :subject-label="selectedPathSubject?.label"
              :subject-color="selectedPathSubject?.color"
              :sample-count="selectedPathSubject?.sampleCount"
              :has-drawable-path="selectedPathSegments.length > 0"
              :unavailable-label="unavailablePathSubjectLabel"
              surface-label="3D scene"
            />
          </div>
        </div>

        <div
          class="timeline-panel"
          :class="{
            'has-segment-map': segmentLayout && sceneVideo?.segments?.length,
            'has-ball-timeline': ballTrajectoryMode === 'manual',
            'has-player-action-timeline': showPlayerActionTimeline,
          }"
        >
          <div v-if="segmentLayout && sceneVideo?.segments?.length" class="master-timeline">
            <div class="master-timeline-heading">
              <div>
                <strong>Full video timeline</strong>
                <span>{{ segmentLayout.groups.length }} events · {{ timelineGroupEditing ? `${multiPassSelection.length} selected` : 'click any segment to seek' }}</span>
              </div>
              <div class="master-timeline-actions">
                <button :class="{ active: timelineGroupEditing }" @click="toggleTimelineGroupEditing">{{ timelineGroupEditing ? 'Close edit' : 'Edit groups' }}</button>
                <button v-if="timelineGroupEditing" class="split" :disabled="!canSplitSelection" @click="splitSelectedIntoNewEvent">Split selected</button>
                <button v-if="timelineGroupEditing" class="save" @click="saveTimelineGroupMap">Save map</button>
              </div>
            </div>
            <div class="master-timeline-track" :class="{ editing: timelineGroupEditing }" aria-label="Full video event timeline">
              <button
                v-for="segment in sceneVideo.segments"
                :key="segment.id"
                :class="[segment.layout?.role, { selected: multiPassSelection.includes(segment.id) }]"
                :style="{
                  width: `${(segment.duration / scene.duration) * 100}%`,
                  backgroundColor: `${segmentGroupColor(segment.layout?.group)}20`,
                  borderColor: segmentGroupColor(segment.layout?.group),
                }"
                :aria-label="`${segment.layout?.label ?? segment.label}, ${segmentRoleLabel(segment.layout?.role)}, ${segment.start.toFixed(2)} to ${segment.end.toFixed(2)} seconds`"
                @click="handleTimelineSegment(segment)"
              >
                <strong>{{ segment.layout?.label ?? segment.label }}</strong>
                <small>{{ segment.start.toFixed(1) }}–{{ segment.end.toFixed(1) }}s</small>
                <i>{{ segmentRoleLabel(segment.layout?.role) }}</i>
              </button>
              <span class="master-playhead" :style="{ left: `${(currentTime / scene.duration) * 100}%` }" />
            </div>
          </div>
          <ManualBallTimeline
            v-if="ballTrajectoryMode === 'manual'"
            :duration="scene.duration"
            :current-time="currentTime"
            :keyframes="manualBallKeyframes"
            :selected-time="selectedBallKeyframeTime"
            :saving="ballTrajectorySaving"
            :disabled="reconstructionRunning"
            @seek="seekTo"
            @add="addManualBallKeypoint"
            @select="selectManualBallKeypoint"
            @remove="removeManualBallKeypoint"
            @update-time="updateManualBallKeypointTime"
          />
          <PlayerActionTimeline
            v-if="showPlayerActionTimeline && selectedActionActorId"
            :canonical-person-id="selectedActionActorId"
            :person-label="selectedActionActorLabel"
            :duration="scene.duration"
            :current-time="currentTime"
            :actions="selectedActorActions"
            :selected-action-id="selectedPlayerActionId"
            :saving="playerActionSaving"
            :disabled="reconstructionMutationLocked"
            @seek="seekPlayerAction"
            @add="addPlayerActionAt"
            @select="selectPlayerAction"
            @update="updatePlayerAction"
            @remove="removePlayerAction"
          />
          <div class="transport">
            <button class="play-button" @click="togglePlay">{{ playing ? 'Ⅱ' : '▶' }}</button>
            <span class="timecode">{{ timeLabel }}</span>
            <select v-model="playbackRate" aria-label="Playback speed">
              <option :value="0.25">0.25×</option>
              <option :value="0.5">0.5×</option>
              <option :value="1">1×</option>
              <option :value="2">2×</option>
            </select>
          </div>
          <div class="timeline-track">
            <input v-model.number="currentTime" type="range" min="0" :max="scene.duration" step="0.01" @input="onTimelineInput" />
            <div class="event-markers">
              <button
                v-for="(binding, index) in scene.payload.eventBindings"
                :key="`${binding.externalEventId}-${index}`"
                :style="{ left: `${(binding.sceneTime / scene.duration) * 100}%` }"
                :title="binding.label"
                @click="seekTo(binding.sceneTime)"
              />
            </div>
            <div class="timeline-scale">
              <span>{{ timelineTick(0) }}</span>
              <span>{{ timelineTick(scene.duration / 3) }}</span>
              <span>{{ timelineTick((scene.duration * 2) / 3) }}</span>
              <span>{{ timelineTick(scene.duration) }}</span>
            </div>
          </div>
          <span class="duration">{{ scene.duration.toFixed(2) }}s</span>
        </div>
      </section>

      <aside class="panel inspector-panel">
        <div class="inspector-tabs">
          <button :class="{ active: activeTab === 'binding' }" @click="activeTab = 'binding'">Inspector</button>
          <button v-if="sceneVideo?.reconstruction" :class="[`verdict-${reconstructionQualityVerdict}`, { active: activeTab === 'qa' }]" @click="activeTab = 'qa'">Quality <span>{{ reconstructionQualityVerdict === 'unknown' ? '?' : reconstructionQualityVerdict.toUpperCase() }}</span></button>
          <button :class="{ active: activeTab === 'events' }" @click="activeTab = 'events'">Events <span>{{ scene.payload.eventBindings.length }}</span></button>
        </div>

        <div v-if="activeTab === 'binding' && (ballSelected || selectedCanonicalPerson || selectedTrack || frameAnalysis || modelComparison || multiPassAnalysis)" class="inspector-body">
          <div v-if="ballSelected" class="manual-ball-inspector">
            <div class="player-identity">
              <span class="large-jersey ball-inspector-icon">●</span>
              <div>
                <p>Selected object</p>
                <h2>Match ball</h2>
                <small>{{ ballTrajectoryMode === 'manual' ? 'Human-authored trajectory' : 'Detector trajectory' }}</small>
              </div>
            </div>

            <div class="field-group">
              <label>Trajectory source</label>
              <select
                :value="ballTrajectoryMode"
                :disabled="ballTrajectorySaving || reconstructionRunning"
                aria-label="Ball trajectory source"
                @change="changeBallTrajectoryMode"
              >
                <option value="automatic">Automatic detection</option>
                <option value="manual">Manual keypoints</option>
              </select>
              <small>Both versions are retained, so switching modes never destroys the other trajectory.</small>
            </div>

            <template v-if="ballTrajectoryMode === 'manual'">
              <div class="field-group">
                <div class="label-row">
                  <label>{{ selectedManualBallKeyframe ? `Keypoint at ${selectedManualBallKeyframe.t.toFixed(3)}s` : `Playhead at ${currentTime.toFixed(2)}s` }}</label>
                  <span>metres</span>
                </div>
                <div v-if="selectedManualBallKeyframe" class="position-grid">
                  <label>X <input type="number" step="0.1" :value="selectedManualBallKeyframe.x.toFixed(2)" :disabled="ballTrajectorySaving" @change="updateManualBallCoordinate('x', ($event.target as HTMLInputElement).value)" /></label>
                  <label>Z <input type="number" step="0.1" :value="selectedManualBallKeyframe.z.toFixed(2)" :disabled="ballTrajectorySaving" @change="updateManualBallCoordinate('z', ($event.target as HTMLInputElement).value)" /></label>
                </div>
                <small v-else>Add a keypoint at the playhead, or enable placement and click the 3D pitch. The first click creates the keypoint automatically.</small>
                <button class="wide-action" :disabled="ballTrajectorySaving || reconstructionRunning" @click="addManualBallKeypoint(currentTime)">
                  + Add keypoint at {{ currentTime.toFixed(2) }}s
                </button>
                <button class="wide-action" :class="{ active: ballEditMode }" :disabled="ballTrajectorySaving || reconstructionRunning" @click="toggleManualBallPlacement">
                  ◎ {{ ballEditMode ? 'Click the 3D pitch…' : 'Place ball on pitch' }}
                </button>
              </div>
            </template>

            <div class="quality-card ball-trajectory-summary">
              <div><span>Active source</span><strong>{{ ballTrajectoryMode.toUpperCase() }}</strong></div>
              <div><span>Manual keypoints</span><strong>{{ manualBallKeyframes.length }}</strong></div>
              <div><span>Automatic samples</span><strong>{{ automaticBallKeyframes.length }}</strong></div>
              <small v-if="ballTrajectoryMode === 'manual'">Positions are linearly interpolated between consecutive manual keypoints.</small>
              <small v-else>The latest detector result is active. Your manual keypoints remain stored.</small>
            </div>
          </div>

          <div v-if="selectedIdentityReviewPerson" class="canonical-identity-panel">
            <div v-if="identityReviewLoading" class="identity-review-load-state" role="status">
              Loading identity crops and worker readiness…
            </div>
            <div v-else-if="identityReviewError" class="identity-review-load-state error" role="alert">
              <span>{{ identityReviewError }}</span>
              <button type="button" @click="loadIdentityReview(scene.id)">Retry identity review</button>
            </div>
            <IdentityReviewPanel
              :identity="selectedIdentityReviewPerson"
              :roster-players="rosterPlayers"
              :observations="selectedIdentityReviewObservations"
              :worker-states="identityReviewWorkers"
              :dedicated-unbind-active="selectedCanonicalDedicatedUnbindActive"
              :disabled="saving || frameAnnotationSaving || rosterBindingSaving || identityDecisionSaving || reconstructing || reconstructionRunning"
              @bind-candidate="confirmCanonicalRoster"
              @reject-candidate="rejectIdentityCandidate"
              @inspect-frame="inspectIdentityFrame"
              @unbind-roster="unbindCanonicalRoster"
              @clear-roster-binding="clearCanonicalRosterBinding"
            />
            <div v-if="!selectedTrack" class="identity-projection-note" role="status">
              <strong>Not projected in 3D</strong>
              <small>The video identity is preserved, but no trajectory passed metric projection QA for this person.</small>
            </div>
          </div>
          <template v-if="selectedTrack">
            <div class="player-identity">
              <span class="large-jersey" :style="{ background: selectedTrack.color }">{{ selectedTrack.number }}</span>
              <div><p>Selected player</p><h2>{{ selectedTrack.label }}</h2><small>{{ selectedTeam?.name }}</small></div>
            </div>

            <div class="field-grid">
              <div class="field-group"><label>Display name</label><input v-model="selectedTrack.label" @input="saveState = 'Unsaved changes'" /></div>
              <div class="field-group"><label>Number</label><input v-model.number="selectedTrack.number" type="number" min="0" max="99" @input="saveState = 'Unsaved changes'" /></div>
            </div>

            <div class="field-group">
              <div class="label-row"><label>Position at {{ currentTime.toFixed(2) }}s</label><span>metres</span></div>
              <div class="position-grid">
                <label>X <input type="number" step="0.1" :value="interpolateKeyframes(selectedTrack.keyframes, currentTime).x.toFixed(2)" @change="updateTrackPosition('x', ($event.target as HTMLInputElement).value)" /></label>
                <label>Z <input type="number" step="0.1" :value="interpolateKeyframes(selectedTrack.keyframes, currentTime).z.toFixed(2)" @change="updateTrackPosition('z', ($event.target as HTMLInputElement).value)" /></label>
              </div>
              <button class="wide-action" :class="{ active: editMode }" @click="editMode = !editMode">◎ {{ editMode ? 'Click position on pitch…' : 'Place on pitch' }}</button>
            </div>

            <TrackPresenceCard :track="selectedTrack" :current-time="currentTime" />

            <div class="quality-card">
              <div><span>Frame confidence</span><strong>{{ confidenceFor(selectedTrack) }}%</strong></div>
              <div class="quality-bar"><i :style="{ width: `${confidenceFor(selectedTrack)}%` }" /></div>
              <small>{{ selectedTrack.keyframes.length }} keyframes · linear interpolation</small>
              <div><span>Compute status</span><strong>{{ reconstructionProcessingStatus.toUpperCase() }}</strong></div>
              <div><span>Quality verdict</span><strong :class="`quality-${reconstructionQualityVerdict}`">{{ reconstructionQualityVerdict.toUpperCase() }}</strong></div>
              <div><span>Pitch calibration</span><strong>{{ calibrationLabel }}</strong></div>
              <div><span>Visible pitch side</span><strong>{{ visiblePitchSide.toUpperCase() }}</strong></div>
              <div><span>Attacking goal</span><strong>{{ attackingGoalSide.toUpperCase() }}</strong></div>
              <small v-if="pitchCalibration?.status === 'ready'">{{ pitchCalibration.supportedLines }} markings · {{ pitchCalibration.rectangle }}</small>
              <small v-else>{{ pitchCalibration?.reason || 'Screen-relative coordinates' }}</small>
              <small v-if="sceneVideo?.reconstruction?.diagnostics" class="reconstruction-diagnostics">
                {{ analysisFrameCount }} frames · {{ sceneVideo.reconstruction.diagnostics.meanPersonDetections }} detections/frame ·
                {{ sceneVideo.reconstruction.diagnostics.rawTrackCount }} → {{ sceneVideo.reconstruction.diagnostics.stableTrackCount }} → {{ sceneVideo.reconstruction.diagnostics.acceptedTrackCount }} tracks
              </small>
              <small
                v-if="sceneVideo?.reconstruction?.diagnostics?.identityObservationCoverage != null"
                class="reconstruction-diagnostics"
              >
                Video observations retained {{ Math.round(sceneVideo.reconstruction.diagnostics.identityObservationCoverage * 100) }}% (not identity accuracy) ·
                metric {{ Math.round((sceneVideo.reconstruction.diagnostics.metricObservationCoverage ?? 0) * 100) }}% ·
                {{ sceneVideo.reconstruction.diagnostics.discardedProjectedObservationCount ?? 0 }} rejected metric observations
              </small>
              <small class="reconstruction-diagnostics">
                {{ identityValidationSummary(sceneVideo?.reconstruction?.quality?.identityValidation) }}
              </small>
              <small
                v-if="sceneVideo?.reconstruction?.diagnostics?.identity"
                class="reconstruction-diagnostics"
              >
                ReID {{ sceneVideo.reconstruction.diagnostics.identity.reidSelectedIndependentSampleCount ?? 0 }} independent / {{ sceneVideo.reconstruction.diagnostics.identity.reidUsableObservationCount ?? 0 }} usable crops ·
                jersey OCR {{ sceneVideo.reconstruction.diagnostics.identity.jerseyReliablePersonCount ?? 0 }} reliable, {{ sceneVideo.reconstruction.diagnostics.identity.jerseyConflictPersonCount ?? 0 }} conflicts ·
                association p10 {{ sceneVideo.reconstruction.diagnostics.identity.associationConfidenceP10 == null ? 'n/a' : `${Math.round(sceneVideo.reconstruction.diagnostics.identity.associationConfidenceP10 * 100)}%` }}
              </small>
            </div>
          </template>

          <div v-if="frameAnalysis" class="frame-analysis-card" :class="{ stale: !activeFrameAnalysis }">
            <div class="frame-analysis-heading">
              <div><span>Frame recognition</span><strong>{{ frameAnalysis.sceneTime.toFixed(2) }}s · #{{ frameAnalysis.frameIndex }}</strong></div>
              <div class="frame-analysis-actions">
                <i>{{ activeFrameAnalysis ? 'CURRENT' : 'MOVE PLAYHEAD BACK' }}</i>
                <button :class="{ active: frameAnnotationMode }" :disabled="!activeFrameAnalysis" @click="toggleFrameAnnotationMode">
                  {{ frameAnnotationMode ? 'CLOSE LABELS' : 'LABEL FRAME' }}
                </button>
              </div>
            </div>
            <div v-if="frameAnnotationMode" class="frame-annotation-editor">
              <p v-if="!frameAnnotationDraft">Click an existing box, including an unmatched detection, or drag a new box around any person in the video.</p>
              <template v-else>
                <label>
                  <span>Identity correction</span>
                  <select v-model="frameAnnotationDraft.action" aria-label="Frame identity correction" @change="onFrameIdentityActionChange">
                    <option v-for="item in frameIdentityActions" :key="item.value" :value="item.value" :disabled="item.value === 'merge' && !frameIdentityMergeTargets.length">{{ item.label }}</option>
                  </select>
                </label>
                <label v-if="frameAnnotationDraft.action === 'merge'">
                  <span>Merge into</span>
                  <select v-model="frameAnnotationDraft.mergeTargetId" aria-label="Identity merge target">
                    <option :value="null">Choose existing identity</option>
                    <option v-for="target in frameIdentityMergeTargets" :key="`${target.type}-${target.id}`" :value="target.id">
                      {{ target.type === 'canonical' ? 'Canonical person' : target.type === 'track' ? 'Legacy track' : 'Manual person' }} · {{ target.label }}
                    </option>
                  </select>
                </label>
                <small v-if="frameAnnotationDraft.action === 'merge' && frameIdentityMergeTargets.length" class="split-warning">
                  Targets with incompatible dedicated Bind / Unbind decisions are unavailable. Resolve roster decisions in the canonical identity inspector first.
                </small>
                <small v-else-if="frameAnnotationDraft.action === 'merge'" class="split-warning">
                  No compatible merge targets. Choose another correction or resolve roster decisions in the canonical identity inspector.
                </small>
                <div v-if="frameAnnotationDraft.action === 'split'" class="identity-split-preview" role="status">
                  <div>
                    <strong>Split {{ frameIdentitySplitPreview?.identityLabel }}</strong>
                    <small>
                      Selected observation {{ frameAnnotationDraft.targetObservationId ? 'is pinned' : 'is unavailable' }}
                      <template v-if="frameIdentitySplitPreview?.targetTime != null"> at {{ frameIdentitySplitPreview.targetTime.toFixed(2) }}s</template>.
                    </small>
                  </div>
                  <div class="identity-split-range">
                    <label>
                      <span>Range start</span>
                      <input v-model.number="frameAnnotationDraft.rangeStart" type="number" min="0" :max="scene.duration" step="0.01" aria-label="Identity split range start" />
                    </label>
                    <label>
                      <span>Range end · exclusive</span>
                      <input v-model.number="frameAnnotationDraft.rangeEnd" type="number" min="0" :max="scene.duration" step="0.01" aria-label="Identity split range end" />
                    </label>
                  </div>
                  <small v-if="frameIdentitySplitPreview?.affected != null">
                    Preview · {{ frameIdentitySplitPreview.affected }} observation(s) become a new identity; {{ frameIdentitySplitPreview.remaining }} stay on the current identity.
                  </small>
                  <small v-else>Preview counts require a rebuilt canonical observation graph.</small>
                  <small class="split-warning">[start, end) is a cannot-link barrier. Ambiguous remapping aborts the rebuild instead of splitting a nearby player.</small>
                </div>
                <label v-if="frameAnnotationDraft.action === 'exclude'">
                  <span>Exclude scope</span>
                  <select v-model="frameAnnotationDraft.scope" aria-label="Identity exclusion scope">
                    <option value="observation">This observation only</option>
                    <option value="identity" :disabled="!frameAnnotationDraft.canonicalPersonId && !frameAnnotationDraft.sourceTrackId">Whole canonical identity</option>
                  </select>
                </label>
                <label v-if="frameAnnotationDraft.action === 'confirm'">
                  <span>Meaning</span>
                  <select v-model="frameAnnotationDraft.kind" aria-label="Frame person meaning">
                    <option v-for="item in frameAnnotationKinds" :key="item.value" :value="item.value">{{ item.label }}</option>
                  </select>
                </label>
                <label v-if="frameAnnotationDraft.action === 'confirm' && frameAnnotationDraft.kind !== 'ignore'">
                  <span>Label</span>
                  <input v-model="frameAnnotationDraft.label" aria-label="Frame person label" placeholder="Player A, Player B…" />
                </label>
                <small v-if="frameAnnotationDraft.action === 'confirm' && (frameAnnotationDraft.kind.startsWith('home-') || frameAnnotationDraft.kind.startsWith('away-'))">
                  Confirm the observation here; bind a roster player from the canonical identity inspector.
                </small>
                <small>Box {{ Math.round(frameAnnotationDraft.bbox.x) }}, {{ Math.round(frameAnnotationDraft.bbox.y) }} · {{ Math.round(frameAnnotationDraft.bbox.width) }}×{{ Math.round(frameAnnotationDraft.bbox.height) }} px</small>
                <div class="frame-annotation-buttons">
                  <button v-if="frameAnnotationDraft.annotationId" class="delete" :disabled="frameAnnotationSaving || reconstructing || reconstructionStatus === 'queued' || reconstructionStatus === 'processing'" @click="deleteFrameAnnotation">Delete</button>
                  <button class="save" :disabled="frameIdentitySaveDisabled || frameAnnotationSaving || reconstructing || reconstructionStatus === 'queued' || reconstructionStatus === 'processing'" @click="saveFrameAnnotation">{{ frameAnnotationSaving ? 'Saving…' : 'Save correction' }}</button>
                </div>
              </template>
              <small>Preview updates immediately. Saving queues a revision-safe tracking rebuild. Split uses an exclusive-end range and can be undone by deleting its correction.</small>
            </div>
            <div class="frame-analysis-stats">
              <span><strong>{{ frameAnalysis.people.length }}</strong> people</span>
              <span><strong>{{ frameAnalysis.matchedTracks }}</strong> matched</span>
              <span><strong>{{ frameAnalysis.ballCandidates.length }}</strong> ball candidates</span>
            </div>
            <div class="frame-detection-list">
              <button
                v-for="person in frameAnalysis.people"
                :key="person.id"
                :class="{ selected: person.id === selectedFramePersonId && Boolean(activeFrameAnalysis) }"
                @click="selectDetectedPerson(person); seekTo(frameAnalysis.sceneTime)"
              >
                <i :style="{ background: person.jerseyColor }" />
                <span><strong>{{ framePersonLabel(person) }}</strong><small>{{ person.previewState === 'merged' ? `merge → ${person.mergeTargetId} · ` : person.previewState === 'split' ? `split [${person.rangeStart?.toFixed(2)}, ${person.rangeEnd?.toFixed(2)}) · ` : person.previewState === 'confirmed' ? 'confirmed · ' : '' }}{{ person.kind ? `${person.kind} · ` : '' }}{{ framePersonCanonicalId(person) ? `${framePersonCanonicalId(person)} · ` : '' }}x {{ person.pitch.x.toFixed(1) }} · z {{ person.pitch.z.toFixed(1) }}{{ person.matchDistance !== null ? ` · Δ${person.matchDistance.toFixed(1)}m` : '' }} <em v-if="frameMetricBadge(person)" :class="{ uncertain: frameMetricBadge(person) === 'UNCERTAIN' }">{{ frameMetricBadge(person) }}</em></small></span>
                <b :class="person.previewState">{{ person.previewState === 'merged' ? 'MERGED' : person.previewState === 'split' ? 'SPLIT' : person.previewState === 'confirmed' ? 'CONFIRMED' : `${Math.round(person.confidence * 100)}%` }}</b>
              </button>
            </div>
            <div v-if="frameAnalysis.ballCandidates.length" class="frame-ball-list">
              <span
                v-for="ball in frameAnalysis.ballCandidates"
                :key="ball.id"
                :class="{ primary: ball.primary }"
              >
                <i /> {{ ball.primary ? 'Selected ball' : 'Candidate' }} · {{ Math.round(ball.confidence * 100) }}% · x {{ ball.pitch.x.toFixed(1) }} · z {{ ball.pitch.z.toFixed(1) }}
              </span>
            </div>
            <small v-for="warning in frameAnalysis.warnings" :key="warning" class="frame-analysis-warning">{{ warning }}</small>
          </div>

          <div v-if="modelComparison" class="model-comparison-card">
            <div class="model-comparison-heading">
              <div><span>Recognition benchmark</span><strong>{{ modelComparison.frameCount }} identical frames</strong></div>
              <i :class="modelComparison.comparison.verdict">
                {{ modelComparison.comparison.verdict === 'candidate' ? 'M LEADS' : modelComparison.comparison.verdict === 'baseline' ? 'N LEADS' : 'REVIEW' }}
              </i>
            </div>
            <div class="model-run-grid">
              <div>
                <b>{{ modelComparison.baseline.model.replace('.pt', '') }}</b><small>BASELINE</small>
                <strong>{{ modelComparison.baseline.meanDetectionsPerFrame }}</strong><span>people / frame</span>
                <dl>
                  <div><dt>In pitch</dt><dd>{{ modelComparison.baseline.inPitchDetections }}</dd></div>
                  <div><dt>Outside</dt><dd>{{ modelComparison.baseline.outsidePitchDetections }}</dd></div>
                  <div><dt>Stable → accepted</dt><dd>{{ modelComparison.baseline.stableTrackCount }} → {{ modelComparison.baseline.acceptedTrackCount }}</dd></div>
                  <div><dt>Boundary risk</dt><dd>{{ modelComparison.baseline.boundaryRiskTrackCount }}</dd></div>
                  <div><dt>Inference</dt><dd>{{ modelComparison.baseline.inferenceSeconds.toFixed(1) }}s</dd></div>
                </dl>
              </div>
              <div class="candidate">
                <b>{{ modelComparison.candidate.model.replace('.pt', '') }}</b><small>CANDIDATE</small>
                <strong>{{ modelComparison.candidate.meanDetectionsPerFrame }}</strong><span>people / frame</span>
                <dl>
                  <div><dt>In pitch</dt><dd>{{ modelComparison.candidate.inPitchDetections }}</dd></div>
                  <div><dt>Outside</dt><dd>{{ modelComparison.candidate.outsidePitchDetections }}</dd></div>
                  <div><dt>Stable → accepted</dt><dd>{{ modelComparison.candidate.stableTrackCount }} → {{ modelComparison.candidate.acceptedTrackCount }}</dd></div>
                  <div><dt>Boundary risk</dt><dd>{{ modelComparison.candidate.boundaryRiskTrackCount }}</dd></div>
                  <div><dt>Inference</dt><dd>{{ modelComparison.candidate.inferenceSeconds.toFixed(1) }}s</dd></div>
                </dl>
              </div>
            </div>
            <div class="model-comparison-deltas">
              <span><strong>{{ modelComparison.comparison.sharedDetections }}</strong> shared</span>
              <span><strong>+{{ modelComparison.comparison.candidateOnlyInPitchDetections }}</strong> M-only in field</span>
              <span><strong>{{ modelComparison.comparison.baselineOnlyInPitchDetections }}</strong> N-only in field</span>
            </div>
            <p>
              In-field delta
              <strong>{{ modelComparison.comparison.inPitchObservationGain >= 0 ? '+' : '' }}{{ modelComparison.comparison.inPitchObservationGain }}</strong>
              · outside delta
              <strong>{{ modelComparison.comparison.outsidePitchDetectionDelta >= 0 ? '+' : '' }}{{ modelComparison.comparison.outsidePitchDetectionDelta }}</strong>
              · stable tracks
              <strong>{{ modelComparison.comparison.stableTrackDelta >= 0 ? '+' : '' }}{{ modelComparison.comparison.stableTrackDelta }}</strong>
            </p>
            <small>{{ modelComparison.warnings[0] }}</small>
          </div>

          <div v-if="multiPassAnalysis" class="multi-pass-card">
            <div><span>Reconstruction evidence</span><strong>{{ Math.round((multiPassAnalysis.consensus?.evidenceScore ?? 0) * 100) }}%</strong></div>
            <small>{{ multiPassAnalysis.consensus?.passesAnalyzed ?? 0 }} passes · {{ multiPassAnalysis.consensus?.metricPasses ?? 0 }} metric · {{ multiPassAnalysis.consensus?.ballPasses ?? 0 }} with ball</small>
            <small v-if="multiPassAnalysis.ballSupport" class="aligned-support">
              {{ multiPassAnalysis.consensus?.overlappingPasses ?? 0 }} aligned replay · {{ multiPassAnalysis.ballSupport.supportedSamples }}/{{ multiPassAnalysis.ballSupport.referenceSamples }} ball samples supported
            </small>
            <div class="pass-list">
              <span v-for="item in multiPassAnalysis.passes" :key="item.segmentId" :class="{ reference: item.sceneId === multiPassAnalysis.referenceSceneId }">
                {{ item.label }} · {{ Math.round(item.quality * 100) }}% <i>{{ passRelationLabel(item.relation).toUpperCase() }} · QA {{ (item.qualityVerdict ?? 'legacy').toUpperCase() }}</i>
              </span>
            </div>
            <small class="evidence-note">Motion alignment verifies overlapping replays. Continuation shots extend the event but are not fused into the reference trajectories.</small>
          </div>
        </div>

        <div v-else-if="activeTab === 'qa'" class="inspector-body calibration-qa-body">
          <CalibrationQaPanel
            :processing-status="reconstructionProcessingStatus"
            :quality-verdict="reconstructionQualityVerdict"
            :coordinate-space="sceneVideo?.reconstruction?.coordinateSpace ?? null"
            :quality="sceneVideo?.reconstruction?.quality ?? null"
            :calibration="calibrationEvidence"
            :ball-detection="sceneVideo?.reconstruction?.ballDetection ?? null"
            :ball-diagnostics="scene.payload.ball.diagnostics ?? null"
            :frames="calibrationFrames"
            :current-time="currentTime"
            :visible-pitch-side="visiblePitchSide"
            :visible-pitch-side-source="visiblePitchSideSource"
            :attacking-goal="attackingGoalSide"
            :direction-saving="pitchSideSaving || Boolean(calibrationDraft) || reconstructionRunning"
            :legacy-calibration="pitchCalibration ?? null"
            @seek="seekTo"
            @calibrate="calibrateQaFrame"
            @change-attacking-goal="changeAttackingGoal"
          />
        </div>

        <div v-else-if="activeTab === 'events'" class="inspector-body events-body">
          <div v-if="boundEvent" class="bound-match">
            <p>{{ projectMatchContext.inherited ? 'Project match · inherited by this shot' : 'Project match' }}</p>
            <strong>{{ boundEvent.home.name }} {{ boundEvent.home_score ?? '–' }} — {{ boundEvent.away_score ?? '–' }} {{ boundEvent.away.name }}</strong>
            <small>{{ boundEvent.date }} · {{ boundEvent.league }} · {{ boundEvent.status || 'status unknown' }}</small>
            <button
              v-if="matchSnapshotRefreshAvailable"
              type="button"
              :disabled="matchSnapshotRefreshing || reconstructionMutationLocked"
              @click="refreshMatchSnapshot"
            >{{ matchSnapshotRefreshing ? 'Refreshing…' : 'Refresh project snapshot' }}</button>
          </div>
          <div v-else-if="scene.payload.matchBinding" class="bound-match">
            <p>Saved legacy project match</p>
            <strong>{{ scene.payload.matchBinding.source }} · #{{ scene.payload.matchBinding.eventId }}</strong>
            <small>Refresh once to persist the project match, roster and timeline for offline identity review.</small>
            <button
              v-if="matchSnapshotRefreshAvailable"
              type="button"
              :disabled="matchSnapshotRefreshing || reconstructionMutationLocked"
              @click="refreshMatchSnapshot"
            >{{ matchSnapshotRefreshing ? 'Refreshing…' : 'Refresh project snapshot' }}</button>
          </div>
          <button v-else class="empty-bind" @click="openMatchSettings">＋ Set project match data</button>

          <div v-if="eventBundle?.timeline.length" class="source-events">
            <p class="section-label">Source timeline</p>
            <button v-for="item in eventBundle.timeline" :key="item.id" @click="addEventBinding(item)">
              <span>{{ item.minute ?? '—' }}′</span><strong>{{ item.label }}</strong><i>＋</i>
            </button>
          </div>

          <div v-if="eventBundle?.warnings.length" class="source-warnings">
            <p v-for="warning in eventBundle.warnings" :key="warning">{{ warning }}</p>
          </div>

          <div class="scene-events">
            <p class="section-label">Scene markers</p>
            <div v-for="(item, index) in scene.payload.eventBindings" :key="`${item.externalEventId}-${index}`">
              <button class="marker-time" @click="currentTime = item.sceneTime">{{ item.sceneTime.toFixed(2) }}s</button>
              <span>{{ item.label }}</span>
              <button class="remove" @click="removeEventBinding(index)">×</button>
            </div>
            <small v-if="!scene.payload.eventBindings.length">Add source events at the current playhead position.</small>
          </div>
        </div>
        <div v-else class="inspector-body empty-reconstruction">
          <span>{{ reconstructionRunning ? `AI ANALYSIS · ${reconstructionProgress?.overallPercent ?? 0}%` : reconstructionStatus === 'failed' ? 'ANALYSIS NEEDS REVIEW' : segmentLayout ? 'EVENT MAP READY' : 'FRAME SET READY' }}</span>
          <h2>{{ reconstructionRunning ? reconstructionProgress?.label ?? (multiPassAnalysis ? `Analyzing ${multiPassAnalysis.selectedSegmentIds.length} camera angles…` : 'Preparing analysis…') : segmentLayout ? 'Review detected events' : 'No reconstructed tracks yet' }}</h2>
          <p v-if="reconstructionRunning">{{ reconstructionProgress?.detail ?? (multiPassAnalysis ? `Pass ${multiPassAnalysis.currentPass || 1} of ${multiPassAnalysis.selectedSegmentIds.length}: reconstructing each angle before choosing a canonical view.` : 'Preparing sampled frames for the detector.') }}</p>
          <p v-else>{{ segmentLayout ? 'Check which shots belong to the same event, correct replay roles, then confirm the proposed map.' : sceneVideo?.reconstruction?.error || 'Run automatic reconstruction to populate player and ball tracks from the extracted frames.' }}</p>
          <div v-if="sceneVideo" class="frame-summary">
            <strong>{{ analysisFrameCount }}</strong><small>{{ sceneVideo.selectedSegmentId ? 'analysis frames' : 'detector frames' }}</small>
            <strong>{{ sceneVideo.fps.toFixed(2) }}</strong><small>source FPS</small>
          </div>
          <div v-if="segmentLayout && sceneVideo?.segments?.length" class="layout-editor-card">
            <div class="layout-editor-heading">
              <div><span>Suggested event map</span><strong>{{ segmentLayout.groups.length }} events · {{ Math.round(segmentLayout.confidence * 100) }}%</strong></div>
              <i :class="segmentLayout.status">{{ segmentLayout.status }}</i>
            </div>
            <p>{{ segmentLayout.method === 'scoreboard-change+motion-dtw' ? `Score changes at ${segmentLayout.scoreChangeTimes.map((time) => `${time.toFixed(0)}s`).join(', ')}; replay roles use motion alignment.` : 'No stable scoreboard found; review the order-based grouping.' }}</p>
            <div class="layout-editor-actions">
              <button :disabled="layoutRebuilding" @click="rebuildSegmentLayout">{{ layoutRebuilding ? 'Analyzing…' : '↻ Rebuild' }}</button>
              <button class="confirm" @click="confirmSegmentLayout">✓ Confirm map</button>
              <button class="split" :disabled="!canSplitSelection" @click="splitSelectedIntoNewEvent">
                ＋ Split {{ multiPassSelection.length || 'selected' }} into new event
              </button>
            </div>
          </div>
          <button v-if="sceneVideo?.selectedSegmentId" class="wide-action" :disabled="reconstructing || reconstructionStatus === 'processing' || reconstructionStatus === 'queued'" @click="reconstructCurrentScene">
            {{ reconstructionStatus === 'processing' || reconstructionStatus === 'queued' ? 'Analyzing…' : '◎ Build automatic tracks' }}
          </button>
          <div v-if="sceneVideo?.segments?.length" class="shot-candidates">
            <div class="label-row"><label>Multi-angle passes</label><span>{{ multiPassSelection.length }}/6 selected</span></div>
            <p class="multi-pass-copy">Select a continuous tail such as 1-B + 1-C and split it into a new event, or select 2–6 variants for reconstruction.</p>
            <div v-for="segment in sceneVideo.segments" :key="segment.id" class="shot-candidate" :class="{ recommended: segment.recommended, selected: multiPassSelection.includes(segment.id) }">
              <div class="shot-candidate-main">
                <label class="shot-selector">
                  <input v-model="multiPassSelection" type="checkbox" :value="segment.id" :disabled="!multiPassSelection.includes(segment.id) && multiPassSelection.length >= 6" />
                  <b v-if="segment.layout" class="segment-layout-label" :style="{ borderColor: segmentGroupColor(segment.layout.group), color: segmentGroupColor(segment.layout.group) }">{{ segment.layout.label }}</b>
                  <span><strong>{{ segment.recommended ? '★ ' : '' }}{{ segment.label }}</strong><small>{{ segment.start.toFixed(2) }}–{{ segment.end.toFixed(2) }}s</small></span>
                </label>
                <div v-if="segment.layout" class="shot-layout-controls">
                  <select :value="segment.layout.group" :aria-label="`Event for ${segment.layout.label}`" @change="assignSegmentGroup(segment, ($event.target as HTMLSelectElement).value)">
                    <option v-for="group in layoutGroupOptions" :key="group" :value="group">Event {{ group }}</option>
                  </select>
                  <select :value="segment.layout.role" :aria-label="`Role for ${segment.layout.label}`" @change="assignSegmentRole(segment, ($event.target as HTMLSelectElement).value)">
                    <option value="original">Original</option>
                    <option value="replay">Replay</option>
                    <option value="continuation">Continuation</option>
                  </select>
                </div>
              </div>
              <button :disabled="segmentCreating === segment.id" @click="createSceneFromSegment(segment)">{{ segmentCreating === segment.id ? '…' : segment.sceneId ? 'OPEN' : '→' }}</button>
            </div>
            <button class="wide-action multi-pass-action" :disabled="multiPassSelection.length < 2 || multiPassStarting" @click="startMultiPass">
              {{ multiPassStarting ? 'Starting analysis…' : `◎ Analyze ${multiPassSelection.length || 'selected'} angles` }}
            </button>
          </div>
          <button class="wide-action" @click="videoIngestOpen = true">Import another clip</button>
        </div>
      </aside>
    </section>

    <transition name="drawer">
      <div v-if="catalogOpen" class="drawer-backdrop" @click.self="catalogOpen = false">
        <aside class="catalog-drawer">
          <div class="drawer-heading"><div><p class="eyebrow">{{ matchDataProviderLabel(selectedCatalogProvider) }}</p><h2>Project match data</h2></div><button class="icon-button" @click="catalogOpen = false">×</button></div>
          <p class="drawer-copy">Choose one match for the full video project. Teams, roster, lineup and match events are inherited by every shot.</p>
          <div class="provider-picker">
            <label for="match-data-provider">Data provider</label>
            <select
              id="match-data-provider"
              v-model="selectedCatalogProvider"
              :disabled="providerCatalogLoading || catalogLoading || Boolean(bundleLoading)"
              @change="changeCatalogProvider"
            >
              <option v-for="provider in catalogProviders" :key="provider.id" :value="provider.id" :disabled="!provider.configured || !provider.available">
                {{ provider.name }} · {{ matchDataProviderStatus(provider) }}
              </option>
            </select>
            <div
              v-if="selectedMatchDataProvider"
              class="provider-state"
              :class="{ ready: selectedMatchDataProviderReady, unavailable: !selectedMatchDataProviderReady }"
            >
              <strong>{{ matchDataProviderStatus(selectedMatchDataProvider) }}</strong>
              <span v-if="selectedMatchDataProviderReady">Used by this project after you bind a match.</span>
              <span v-else>{{ selectedMatchDataProviderReason }}</span>
            </div>
            <small v-for="provider in unavailableCatalogProviders" :key="`unavailable-${provider.id}`" class="provider-unavailable-note">
              {{ provider.name }}: {{ providerUnavailableReason(provider) }}
            </small>
          </div>
          <div class="date-search match-search"><input v-model="catalogQuery" type="search" placeholder="Spain vs Belgium" :disabled="!selectedMatchDataProviderReady" @keyup.enter="loadCatalogSearch" /><button class="button primary" :disabled="!selectedMatchDataProviderReady || catalogLoading" @click="loadCatalogSearch">Search match</button></div>
          <div class="date-search"><input v-model="catalogDate" type="date" :disabled="!selectedMatchDataProviderReady" /><button class="button primary" :disabled="!selectedMatchDataProviderReady || catalogLoading" @click="loadCatalog">Load fixtures</button></div>
          <div v-if="catalogLoading" class="catalog-loading"><i /><span>Contacting football catalog…</span></div>
          <div v-else-if="catalogError" class="catalog-provider-error" role="alert"><strong>Provider request failed</strong><span>{{ catalogError }}</span></div>
          <div v-else class="event-list">
            <button v-for="event in catalogEvents" :key="event.id" :disabled="Boolean(bundleLoading) || reconstructionMutationLocked" @click="bindMatch(event)">
              <div class="event-meta"><span>{{ event.time?.slice(0, 5) || 'FT' }}</span><small>{{ event.league }}</small></div>
              <div class="event-teams"><strong>{{ event.home.name }}</strong><strong>{{ event.away.name }}</strong></div>
              <div class="event-score"><strong>{{ event.home_score ?? '–' }}</strong><strong>{{ event.away_score ?? '–' }}</strong></div>
              <span class="bind-arrow">{{ bundleLoading === event.id ? '…' : '→' }}</span>
            </button>
            <div v-if="!catalogEvents.length" class="empty-catalog"><span>◌</span><p>No matches loaded yet.</p><small>Search by teams or choose a date. Free-source coverage varies.</small></div>
          </div>
          <div class="source-note"><span>SERVER-SIDE PROVIDER</span><p>Responses are normalized and cached by Replay Studio. Provider credentials are never sent to the browser.</p></div>
        </aside>
      </div>
    </transition>

    <VideoIngestDrawer :open="videoIngestOpen" @close="videoIngestOpen = false" @ready="openProcessedVideo" />

    <div v-if="error && scene" class="toast" @click="error = null"><strong>Something went wrong</strong><span>{{ error }}</span><button>×</button></div>
  </main>
</template>
