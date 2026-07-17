export type Keyframe = {
  id?: string
  t: number
  x: number
  y?: number
  z: number
  confidence: number
  observed?: boolean
  state?: 'observed' | 'inferred' | 'occluded'
  detectionConfidence?: number | null
  trajectoryConfidence?: number | null
  confidenceKind?: string | null
  positionSource?: string | null
  heightSource?: string | null
  provenance?: {
    source: string
    method?: string
    [key: string]: unknown
  }
  presenceState?: 'observed' | 'inferred-before-first' | 'inferred-gap' | 'inferred-after-last'
  support?: number
  metricSupport?: number
  projection?: {
    source: ProjectionSource
    calibrationFrameIndex?: number | null
    uncertaintyMetres?: number | null
  }
  projectionSource?: ProjectionSource
  positionUncertaintyMetres?: number | null
}

export type PlayerActionType =
  | 'idle'
  | 'walk'
  | 'run'
  | 'sprint'
  | 'turn'
  | 'jump'
  | 'fall'
  | 'get-up'
  | 'first-touch'
  | 'drive'
  | 'pass'
  | 'cross'
  | 'shot'
  | 'header'
  | 'throw-in'
  | 'clearance'
  | 'tackle'
  | 'slide-tackle'
  | 'block'
  | 'interception'
  | 'feint'

export type PlayerActionKeypointKind =
  | 'wind-up'
  | 'contact'
  | 'release'
  | 'apex'
  | 'impact'
  | 'recovery'

export type PlayerActionKeypoint = {
  kind: PlayerActionKeypointKind
  time: number
}

export type PlayerActionStatus = 'suggested' | 'confirmed' | 'rejected'
export type PlayerActionSource = 'automatic' | 'manual'

export type PlayerAction = {
  id: string
  canonicalPersonId: string
  type: PlayerActionType
  startTime: number
  endTime: number
  keypoints: PlayerActionKeypoint[]
  confidence: number
  status: PlayerActionStatus
  source: PlayerActionSource
  evidence?: {
    observationIds?: string[]
    ballTrajectoryFingerprint?: string | null
    model?: string | null
    reasons?: string[]
    artifactUri?: string | null
    artifactHash?: string | null
  }
  createdAt?: string
  updatedAt?: string
}

export type ReconstructionModel =
  | 'yolo26n.pt'
  | 'yolo26s.pt'
  | 'yolo26m.pt'
  | 'yolo26l.pt'
  | 'yolo26x.pt'

export type BallDetectionBackend =
  | 'generic-ultralytics'
  | 'dedicated-ultralytics'
  | 'wasb-service'

export type BallTrajectoryMode = 'automatic' | 'manual'

export type MetricProjectionStatus = 'accepted' | 'rejected' | 'unprojected'
export type ObservationPositionSource = 'observation' | 'track-inferred'

export type CanonicalPersonRole = 'player' | 'goalkeeper' | 'referee' | 'other'
export type CanonicalPersonStatus = 'resolved' | 'provisional' | 'excluded'
export type CanonicalIdentityEvidenceKind =
  | 'manual'
  | 'reid'
  | 'jersey-ocr'
  | 'team'
  | 'role'
  | 'trajectory'
  | 'multi-pass'
  | 'multi-angle-identity'
  | 'roster-prior'

/**
 * Human-readable evidence published by the identity resolver. Model embeddings
 * stay in backend artifacts; the editor only receives compact provenance.
 */
export type CanonicalIdentityEvidence = {
  id: string
  kind: CanonicalIdentityEvidenceKind
  label: string
  value?: string | number | null
  confidence?: number | null
  supportCount?: number
  sampleCount?: number
  source?: string | null
  model?: string | null
  frameIndices?: number[]
  manual?: boolean
  /** Cross-view provenance published when an independently aligned replay supports this identity. */
  sourceSceneId?: string | null
  sourceCanonicalPersonId?: string | null
  signals?: string[]
  alignmentConfidence?: number | null
  alignmentMethod?: string | null
  observationCount?: number
}

export type CanonicalMultiAngleIdentityEvidence = CanonicalIdentityEvidence & {
  kind: 'multi-angle-identity'
  sourceSceneId: string
  sourceCanonicalPersonId: string
  signals: string[]
  observations: TrackObservation[]
  sourceTrackletIds: string[]
}

/** A hypothesis is not a binding until CanonicalPerson.externalPlayerId is set. */
export type CanonicalRosterCandidate = {
  externalPlayerId: string
  rank?: number | null
  /** Legacy candidate probability. New roster fusion publishes `score`. */
  confidence?: number | null
  score?: number | null
  identitySignalScore?: number | null
  name?: string | null
  number?: string | null
  position?: string | null
  teamId?: string | null
  reasons?: string[]
  conflicts?: string[]
  eligible?: boolean
  proposalStatus?: 'selected' | 'alternative' | 'ambiguous' | 'rejected' | string
  requiresManualConfirmation?: boolean
  evidence?: CanonicalIdentityEvidence[]
}

export type CanonicalIdentityConflict = {
  id: string
  code: string
  message: string
  severity: 'review' | 'blocking'
  relatedCanonicalPersonIds?: string[]
  relatedTrackletIds?: string[]
}

export type CanonicalPerson = {
  canonicalPersonId: string
  displayName: string
  identityStatus: CanonicalPersonStatus
  identityConfidence: number | null
  identitySource: string | null
  teamId: string | null
  role: CanonicalPersonRole | null
  jerseyNumber: string | null
  /** Accepted automatic or manual roster binding. Candidate IDs live elsewhere. */
  externalPlayerId: string | null
  memberTrackletIds: string[]
  annotationIds?: string[]
  observationCount?: number
  observations?: TrackObservation[]
  evidence: CanonicalIdentityEvidence[]
  multiAngleEvidence?: CanonicalMultiAngleIdentityEvidence[]
  sourcePassIds?: string[]
  rosterCandidates: CanonicalRosterCandidate[]
  conflicts: CanonicalIdentityConflict[]
  provenance?: 'automatic' | 'manual' | 'mixed'
}

export type CanonicalIdentityDiagnostics = {
  sourceTrackletCount: number
  canonicalPersonCount: number
  resolvedPersonCount: number
  provisionalPersonCount: number
  excludedPersonCount: number
  conflictPersonCount: number
  manualDecisionCount: number
  estimatedIdSwitchCount?: number | null
  duplicateOverlapSeconds?: number | null
  jerseyReadableCoverage?: number | null
  associationConfidenceP10?: number | null
  associationConfidenceP50?: number | null
  acceptedAssociationConfidenceP10?: number | null
  reviewAssociationConfidenceP50?: number | null
  strongReidBidirectionalEdgeCount?: number
  reidUsableObservationCount?: number
  reidSelectedIndependentSampleCount?: number
  reidCropCoverage?: number
  jerseyReliablePersonCount?: number
  jerseyProvisionalPersonCount?: number
  jerseyConflictPersonCount?: number
  rosterCandidateCount?: number
  groundTruthAvailable?: boolean
}

export type TrackObservation = {
  id?: string
  observationId?: string
  frameIndex: number
  sourceFrameIndex?: number
  sceneTime: number
  sourceTime?: number
  bbox: { x: number; y: number; width: number; height: number }
  pitch?: { x: number; z: number }
  rawPitch?: { x: number; z: number } | null
  confidence: number
  annotationId?: string | null
  annotationIds?: string[]
  metricStatus?: MetricProjectionStatus | null
  metricReason?: string | null
  positionSource?: ObservationPositionSource | null
  canonicalPersonId?: string | null
}

export type Track = {
  id: string
  label: string
  teamId: string
  color: string
  number: number
  role?: CanonicalPersonRole
  canonicalPersonId?: string | null
  source?: 'automatic' | 'manual-anchor'
  annotationIds?: string[]
  identityCorrection?: {
    status: 'merged'
    targetId: string
    annotationIds: string[]
    mergedTrackIds: string[]
  }
  externalPlayerId: string | null
  presence?: {
    policy: 'continuous-latent'
    coverage: number
    observationCount: number
    inferredKeyframeCount: number
    observedStart?: number
    observedEnd?: number
    observedSpanRatio?: number
    sampleCadenceSeconds?: number
  }
  observations?: TrackObservation[]
  keyframes: Keyframe[]
}

export type FrameAnnotationKind =
  | 'home-player'
  | 'away-player'
  | 'home-goalkeeper'
  | 'away-goalkeeper'
  | 'referee'
  | 'other'
  | 'ignore'

export type FrameIdentityAction = 'confirm' | 'exclude' | 'merge' | 'split'
export type FrameIdentityScope = 'observation' | 'range' | 'identity'
export type FrameIdentityPreviewState = 'uncorrected' | 'confirmed' | 'excluded' | 'merged' | 'split'

export type FrameAnnotation = {
  id: string
  sceneTime: number
  sourceTime: number
  frameIndex: number
  bbox: { x: number; y: number; width: number; height: number }
  kind: FrameAnnotationKind
  label: string | null
  externalPlayerId: string | null
  action?: FrameIdentityAction
  scope?: FrameIdentityScope
  mergeTargetId?: string | null
  sourceTrackId?: string | null
  canonicalPersonId?: string | null
  targetObservationId?: string | null
  targetObservation?: TrackObservation | null
  rangeStart?: number | null
  rangeEnd?: number | null
  splitCanonicalPersonId?: string | null
  affectedPreview?: {
    canonicalPersonId: string
    splitCanonicalPersonId: string
    rangeStart: number
    rangeEnd: number
    affectedObservationCount: number
    remainingObservationCount: number
  } | null
  previewState?: Exclude<FrameIdentityPreviewState, 'uncorrected'>
  correctionKind?: 'canonical-roster-binding-v1' | string
  rosterBindingState?: 'bound' | 'unbound'
  updatedAt: string
}

export type FrameAnalysis = {
  sceneId: string
  requestedTime: number
  sceneTime: number
  ballSceneTime?: number
  ballFrameIndex?: number
  sourceTime: number
  frameIndex: number
  frameWidth: number
  frameHeight: number
  model: string
  ballBackend?: BallDetectionBackend
  projectionMode: string
  calibrationStatus: 'ready' | 'review' | 'approximate' | 'fallback' | 'rejected'
  matchedTracks: number
  people: Array<{
    id: string
    confidence: number
    bbox: { x: number; y: number; width: number; height: number }
    pitch: { x: number; z: number }
    jerseyColor: string
    annotationId: string | null
    annotationIds?: string[]
    annotationLabel: string | null
    kind: FrameAnnotationKind | null
    source: 'automatic' | 'manual' | 'tracked-observation'
    matchedTrackId: string | null
    matchedTrackLabel: string | null
    canonicalPersonId?: string | null
    identityStatus?: CanonicalPersonStatus | null
    identityConfidence?: number | null
    identitySource?: string | null
    displayName?: string | null
    jerseyNumber?: string | null
    teamId: string | null
    matchDistance: number | null
    observationId?: string | null
    matchSource?: 'persisted-observation' | 'manual-identity' | 'legacy-observed-frame' | null
    matchIou?: number | null
    matchConfidence?: number | null
    matchMargin?: number | null
    metricStatus: MetricProjectionStatus | null
    metricReason: string | null
    positionSource: ObservationPositionSource | null
    correctionAction: FrameIdentityAction | null
    correctionScope: FrameIdentityScope | null
    mergeTargetId: string | null
    sourceTrackId: string | null
    targetObservationId?: string | null
    rangeStart?: number | null
    rangeEnd?: number | null
    splitCanonicalPersonId?: string | null
    affectedPreview?: FrameAnnotation['affectedPreview']
    previewState: FrameIdentityPreviewState
  }>
  annotations: FrameAnnotation[]
  correctionSummary: { confirmed: number; excluded: number; merged: number; split?: number }
  ballCandidates: Array<{
    id: string
    confidence: number
    image: { x: number; y: number }
    pitch: { x: number; z: number }
    primary: boolean
    backend?: string
  }>
  warnings: string[]
  reconstruction?: {
    status: 'queued' | 'processing' | 'ready' | 'failed'
    model?: string
    runId?: string
    runRevision?: number
    inputFingerprint?: string
  }
}

export type ModelComparisonRun = {
  model: string
  frameCount: number
  totalDetections: number
  meanDetectionsPerFrame: number
  minimumDetectionsInFrame: number
  maximumDetectionsInFrame: number
  inPitchDetections: number
  outsidePitchDetections: number
  wouldClampDetections: number
  lowConfidenceDetections: number
  rawTrackCount: number
  stableTrackCount: number
  acceptedTrackCount: number
  boundaryRiskTrackCount: number
  inferenceSeconds: number
  meanInferenceMilliseconds: number
}

export type ModelComparisonReport = {
  sceneId: string
  completedAt: string
  frameCount: number
  settings: {
    imageSize: number
    confidence: number
    device: string
    outsidePitchMarginMetres: number
    metricProjection: boolean
  }
  baseline: ModelComparisonRun
  candidate: ModelComparisonRun
  comparison: {
    sharedDetections: number
    baselineOnlyDetections: number
    candidateOnlyDetections: number
    baselineOnlyInPitchDetections: number
    candidateOnlyInPitchDetections: number
    inPitchObservationGain: number
    outsidePitchDetectionDelta: number
    stableTrackDelta: number
    acceptedTrackDelta: number
    verdict: 'candidate' | 'baseline' | 'review'
    rationale: string[]
  }
  warnings: string[]
}

export type PitchCalibrationPreset =
  | 'penalty-area-left'
  | 'goal-area-left'
  | 'center-circle'
  | 'goal-area-right'
  | 'penalty-area-right'

export type PitchCalibrationAnchor = {
  id: string
  label: string
  image: { x: number; y: number }
  pitch: { x: number; z: number }
}

export type PitchCalibrationAlignmentMetrics = {
  precision: number
  recall: number
  f1: number
  residualP50: number
  residualP95: number
  modelSampleCount?: number
  observedSampleCount?: number
  tolerancePixels?: number
}

export type PitchCalibrationBackendDiagnostics = {
  inputWidth?: number
  inputHeight?: number
  elapsedSeconds?: number
  budgetExhausted?: boolean
  quadCandidateLimit?: number
  candidatePoolLimit?: number
  quadCandidatesGenerated?: number
  quadCandidatesEvaluated?: number
  terminationReason?: 'candidate-limit' | 'candidate-pool-limit' | 'deadline' | 'completed' | string
  candidateLimitReached?: boolean
  candidatePoolLimitReached?: boolean
  candidateEvaluationLimitReached?: boolean
  deadlineExceeded?: boolean
}

export type CalibrationEvidenceLine = {
  id?: string | number | null
  name?: string | null
  label?: string | null
  family?: string | null
  start?: { x: number; y: number } | null
  end?: { x: number; y: number } | null
  points?: Array<{ x: number; y: number }>
  confidence?: number | null
  groundPlane?: boolean | null
  residualP50?: number | null
  residualP95?: number | null
  residualStatus?: 'scored' | 'not-scored' | 'not-scored-3d' | string | null
  accepted?: boolean | null
  inlier?: boolean | null
}

export type PitchCalibrationDraft = {
  sceneId: string
  sceneTime: number
  frameIndex: number
  frameWidth: number
  frameHeight: number
  source: 'reconstruction' | 'frame-evidence' | 'saved' | 'approximate-seed' | 'manual-seed' | 'manual'
  status?: 'accepted' | 'review' | 'rejected' | 'missing'
  method?: string | null
  backend?: string | null
  confidenceKind?: string | null
  visiblePitchSide?: 'left' | 'right' | 'unknown' | null
  preset: PitchCalibrationPreset
  confidence: number
  alignmentError: number | null
  alignmentMetrics?: PitchCalibrationAlignmentMetrics | null
  horizon?: {
    start: { x: number; y: number }
    end: { x: number; y: number }
  } | null
  quality: 'good' | 'review' | 'poor'
  anchors: PitchCalibrationAnchor[]
  markings: Array<{
    id: string
    kind: 'line' | 'curve'
    points: Array<{ x: number; y: number }>
  }>
  imageToPitch: number[][]
  keypoints?: CalibrationEvidencePoint[]
  detectedKeypoints?: CalibrationEvidencePoint[]
  rawLines?: CalibrationEvidenceLine[]
  detectedKeypointCount?: number | null
  inlierCount?: number | null
  inlierRatio?: number | null
  keypointCount?: number | null
  reprojectionP95?: number | null
  rejectionReasons?: string[]
  evidence?: {
    backendDiagnostics?: PitchCalibrationBackendDiagnostics | null
    visiblePitchSide?: 'left' | 'right' | 'unknown' | null
    rawLines?: CalibrationEvidenceLine[]
  } | null
  warnings: string[]
}

export type Team = {
  id: string
  name: string
  color: string
  externalTeamId: string | null
}

export type EventBinding = {
  sceneTime: number
  externalEventId: string
  label: string
  type: string
}

export type VideoSegment = {
  id: string
  label: string
  start: number
  end: number
  duration: number
  score: number
  recommended?: boolean
  sceneId?: string
  layout?: {
    group: number
    variant: string
    label: string
    role: 'original' | 'replay' | 'continuation'
    confidence: number
    motionCost?: number | null
  }
}

export type SegmentLayout = {
  status: 'proposed' | 'edited' | 'confirmed'
  method: 'scoreboard-change+motion-dtw' | 'shot-order-fallback' | 'empty'
  confidence: number
  scoreChangeTimes: number[]
  groups: Array<{
    id: string
    index: number
    label: string
    segmentIds: string[]
    replayCount: number
  }>
  warnings: string[]
}

export type MultiPassSummary = {
  id: string
  status: 'queued' | 'processing' | 'ready' | 'failed'
  parentSceneId: string
  selectedSegmentIds: string[]
  referenceSceneId?: string | null
  currentPass: number
  passes: Array<{
    sceneId: string
    segmentId: string
    label: string
    sourceStart: number
    sourceEnd: number
    status: 'ready' | 'failed'
    quality: number
    trackCount: number
    ballSamples: number
    calibrationStatus: 'ready' | 'approximate' | 'fallback' | 'rejected'
    calibrationConfidence?: number | null
    qualityVerdict?: 'pass' | 'review' | 'reject'
    relation?: 'reference' | 'replay-overlap' | 'continuation-before' | 'continuation-after' | 'independent'
    alignment?: {
      relation: 'reference' | 'replay-overlap' | 'continuation-before' | 'continuation-after' | 'independent'
      method: 'identity' | 'motion-dtw' | 'source-continuity' | 'phase-normalized'
      confidence: number
      motionCost: number
      overlap: boolean
      anchors: Array<{ referenceTime: number; passTime: number }>
    }
    error?: string | null
  }>
  consensus?: {
    passesAnalyzed: number
    metricPasses: number
    ballPasses: number
    trackPasses: number
    overlappingPasses?: number
    evidenceScore: number
  } | null
  ballSupport?: {
    referenceSamples: number
    supportedSamples: number
    visualPasses: number
    metricPasses: number
    spatialErrors: number[]
  }
  warnings: string[]
}

export type ReconstructionPhase = {
  id: string
  label: string
  status: 'completed' | 'current' | 'pending'
}

export type ReconstructionProgress = {
  phase: string
  phaseIndex: number
  phaseCount: number
  label: string
  detail: string
  completed: number
  total: number
  phasePercent: number
  overallPercent: number
  elapsedSeconds: number
  etaSeconds: number | null
  updatedAt: string
  phases: ReconstructionPhase[]
}

export type ProcessingStatus = 'queued' | 'processing' | 'completed' | 'failed'
export type QualityVerdict = 'pass' | 'review' | 'reject' | 'pending' | 'unknown'
export type QualityGateStatus = 'pass' | 'review' | 'fail' | 'not-available'
export type CalibrationFrameStatus = 'pass' | 'review' | 'reject' | 'missing'
export type ProjectionSource =
  | 'direct'
  | 'temporal-forward'
  | 'temporal-backward'
  | 'temporal-bidirectional'
  | 'temporal-interpolation'
  | 'dense-bracket-interpolated'
  | 'manual-propagated'
  | 'screen-approximate'
  | 'propagated'
  | 'presence-inferred'
  | 'manual'
  | 'manual-pitch-coordinate'
  | 'approximate'
  | 'screen'
  | 'none'

export type CameraMotionStatus =
  | 'first-frame'
  | 'reference'
  | 'estimated'
  | 'unreliable'
  | 'cut'
  | 'unestimated'
  | 'missing'

export type CameraMotionEvidence = {
  status: CameraMotionStatus
  model?: string | null
  confidence?: number | null
  currentToPrevious?: number[][] | null
  currentToReference?: number[][] | null
  metrics?: {
    trackedCount?: number | null
    inlierCount?: number | null
    inlierRatio?: number | null
    residualP50Px?: number | null
    residualP95Px?: number | null
    forwardBackwardP95Px?: number | null
    coverageRatio?: number | null
    sceneChangeScore?: number | null
  } | null
  rejectionReasons?: string[]
}

export type CalibrationHypothesisEvidence = {
  id: string
  rank: number
  selected: boolean
  origin: 'direct' | 'direct-rejected' | 'temporal-direct' | 'temporal-forward' | 'temporal-backward'
  eligibility?: 'rejected-observation' | null
  score: number
  scoreKind?: string | null
  visiblePitchSide?: 'left' | 'right' | 'unknown' | null
  anchorFrameIndices: number[]
  anchorSampleIndices?: number[]
  motionEdgeIndices?: number[]
  temporalDistanceSeconds?: number | null
  motionConfidence?: number | null
  uncertaintyP95Metres?: number | null
  disagreementMetres?: number | null
  imageToPitch?: number[][] | null
  rejectionReasons?: string[]
}

export type TemporalCalibrationEvidence = {
  direction: 'forward' | 'backward' | 'bidirectional'
  anchorFrameIndices: number[]
  anchorSampleIndices?: number[]
  anchorSceneTimes?: number[]
  motionEdgeIndices?: number[]
  temporalDistanceSeconds?: number | null
  motionConfidence?: number | null
}

export type CalibrationUncertaintyEvidence = {
  kind?: string | null
  p50Metres?: number | null
  p95Metres?: number | null
  temporalDistanceSeconds?: number | null
  motionConfidence?: number | null
  source?: string | null
  reasons?: string[]
}

export type ReconstructionQualityGate = {
  id: string
  label: string
  status: QualityGateStatus
  value?: number | string | null
  unit?: string | null
  threshold?: string | null
  detail?: string | null
}

export type CalibrationEvidencePoint = {
  id?: string | null
  label?: string | null
  image: { x: number; y: number }
  projected?: { x: number; y: number } | null
  projectedImage?: { x: number; y: number } | null
  residualPx?: number | null
  residualVector?: { dx: number; dy: number; magnitude: number } | null
  inlier?: boolean | null
}

export type CalibrationEvidenceMarking = {
  id: string
  kind: 'line' | 'curve'
  label?: string | null
  points: Array<{ x: number; y: number }>
  accepted?: boolean | null
}

export type CalibrationFrameQuality = {
  frameIndex: number
  sourceFrameIndex?: number | null
  sceneTime: number
  sourceTime?: number | null
  status: CalibrationFrameStatus
  projectionSource: ProjectionSource
  calibrationId?: string | null
  model?: string | null
  confidence?: number | null
  frameWidth?: number | null
  frameHeight?: number | null
  keypointCount?: number | null
  inlierCount?: number | null
  inlierRatio?: number | null
  reprojectionErrorP50Px?: number | null
  reprojectionErrorP95Px?: number | null
  temporalJitterMetres?: number | null
  visibleSide?: 'left' | 'right' | 'unknown' | null
  imageToPitch?: number[][] | null
  keypoints?: CalibrationEvidencePoint[]
  markings?: CalibrationEvidenceMarking[]
  reasons?: string[]
  warnings?: string[]
}

export type ReconstructionQualityReport = {
  verdict: QualityVerdict
  score?: number | null
  reasons?: string[]
  gates: ReconstructionQualityGate[]
  calibration: {
    sampledFrameCount: number
    directFrameCount: number
    propagatedFrameCount?: number
    rejectedFrameCount: number
    missingFrameCount?: number
    directCoverage: number
    metricCoverage?: number | null
    fallbackFrameCount: number
    reprojectionErrorP50Px?: number | null
    reprojectionErrorP95Px?: number | null
    temporalJitterP95Metres?: number | null
    sideFlipCount?: number
    frames: CalibrationFrameQuality[]
  }
}

export type CalibrationFrameEvidence = {
  sourceFrameIndex: number
  sampleIndex: number
  sceneTime: number
  sourceTime: number
  status: 'accepted' | 'rejected' | 'missing'
  observationStatus?: 'direct-accepted' | 'direct-rejected' | 'manual-propagated' | 'missing'
  solutionStatus?:
    | 'direct-accepted'
    | 'temporal-accepted'
    | 'temporal-rejected'
    | 'manual-accepted'
    | 'ambiguous'
    | 'unresolved'
  source: string
  projectionSource: ProjectionSource
  backend?: string | null
  confidence?: number | null
  imageToPitch?: number[][] | null
  keypointCount?: number | null
  detectedKeypointCount?: number | null
  inlierCount?: number | null
  inlierRatio?: number | null
  reprojectionError?: number | null
  reprojectionP95?: number | null
  visiblePitchSide?: 'left' | 'right' | 'unknown' | null
  rejectionReasons?: string[]
  personSupport?: number | { supported: number; total: number; ratio: number } | null
  cameraMotion?: CameraMotionEvidence | null
  temporal?: TemporalCalibrationEvidence | null
  uncertainty?: CalibrationUncertaintyEvidence | null
  positionUncertaintyMetres?: number | null
  selectedHypothesisId?: string | null
  ambiguityMargin?: number | null
  hypotheses?: CalibrationHypothesisEvidence[]
  frameWidth?: number | null
  frameHeight?: number | null
  keypoints?: CalibrationEvidencePoint[]
  rawLines?: CalibrationEvidenceLine[]
  markings?: CalibrationEvidenceMarking[]
}

export type CalibrationEvidence = {
  schemaVersion: 1
  summary: {
    sampledFrameCount: number
    acceptedFrameCount: number
    rejectedFrameCount: number
    missingFrameCount: number
    directCoverage: number
    usableCoverage: number
    directFrameCount?: number
    temporalRecoveredFrameCount?: number
    temporalAmbiguousFrameCount?: number
    temporalUncertaintyP95Metres?: number | null
    cameraMotionReliability?: number | null
    cameraMotionEstimatedEdgeCount?: number
    cameraMotionUnreliableEdgeCount?: number
    cameraMotionCutCount?: number
    maxGapSeconds: number | null
    sideAgreement: number | null
    reprojectionP50?: number | null
    reprojectionP95?: number | null
    alignmentF1P10?: number | null
    visiblePitchSide?: 'left' | 'right' | null
  }
  frameEvidence: CalibrationFrameEvidence[]
}

export type ReconstructionQuality = {
  verdict: QualityVerdict
  summary?: string | string[] | Record<string, unknown> | null
  metrics?: Record<string, number | string | boolean | null | {
    value: number | string | boolean | null
    unit?: string | null
    source?: string | null
    [key: string]: unknown
  }>
  gates?: Array<ReconstructionQualityGate | Record<string, unknown>> | Record<string, Record<string, unknown>>
  limitations?: Array<string | { code: string; message: string }>
  identityValidation?: {
    groundTruthAvailable: boolean
    status: 'evaluated' | 'unavailable' | 'invalid'
    sampleCount?: number
    idf1: number | null
    idPrecision?: number | null
    idRecall?: number | null
    idSwitchCount?: number | null
    fragmentCount?: number | null
    duplicateAssignmentFrameCount?: number | null
    duplicateOverlapSeconds?: number | null
    duplicateOverlapTimebase?: 'seconds' | 'frame-index+explicit-fps' | 'frame-index-without-fps' | 'mixed'
    identityAssignmentFrameRate?: number | null
    hota?: number | null
    gsHota?: number | null
    reason?: string
  }
}

export type PitchOrientation = {
  visiblePitchSide?: 'left' | 'right' | 'unknown'
  visiblePitchSideSource?: 'manual' | 'manual-calibration' | 'calibration' | 'unknown'
  visiblePitchSideAgreement?: number | null
  attackingGoal: 'left' | 'right' | 'unknown'
  attackingGoalSource?: 'manual' | 'unknown'
  source: 'manual' | 'calibration' | 'unknown'
  updatedAt: string
}

export type ExternalLineupEntry = {
  id: string
  player_id: string
  player_name: string
  team_id?: string | null
  team_name?: string | null
  side: 'home' | 'away' | 'unknown'
  position?: string | null
  number?: string | null
  role: 'starter' | 'substitute' | 'unknown'
  order: number
  /** Provider formation for this team, for example `4-3-3`. */
  formation?: string | null
  /** Provider-relative formation cell, for example `2:4`. */
  grid?: string | null
}

export type ExternalSubstitution = {
  id: string
  minute?: number | null
  team_id?: string | null
  team_name?: string | null
  player_out_id?: string | null
  player_out_name?: string | null
  player_in_id?: string | null
  player_in_name?: string | null
  label: string
}

export type ExternalRosterQuality = {
  status: 'automatic-ready' | 'partial' | 'unavailable'
  playerCount: number
  homePlayerCount: number
  awayPlayerCount: number
  automaticIdentityEligible: boolean
  manualIdentityEligible: boolean
  reasons: string[]
}

export type MatchBindingScope = 'project' | 'scene'

/** Stable server-side match-data adapter identifier. Keys never reach the web app. */
export type MatchDataProviderId = 'api-football' | 'thesportsdb' | (string & {})

export type MatchDataProvider = {
  id: MatchDataProviderId
  name: string
  configured: boolean
  available: boolean
  reason?: string | null
  capabilities?: string[]
}

export type MatchDataProviderCatalog = {
  providers: MatchDataProvider[]
  defaultProvider: MatchDataProviderId
  /** Configured product preference before availability fallback is applied. */
  preferredProvider?: MatchDataProviderId
}

/** Versioned offline match snapshot; schema v2 is authoritative for identity UI. */
export type PersistedMatchBinding = {
  schemaVersion?: number
  /**
   * Match settings normally belong to the root video project. Segment and
   * multi-angle scenes receive the same effective snapshot from the API.
   */
  scope?: MatchBindingScope
  /** Root video scene which owns a project-scoped binding. */
  projectSceneId?: string | null
  /** True when this effective snapshot was inherited by an internal scene. */
  inherited?: boolean
  source: string
  eventId: string
  fetchedAt: string | null
  event?: ExternalEvent
  teams?: { home: ExternalTeam; away: ExternalTeam }
  players?: ExternalPlayer[]
  lineup?: ExternalLineupEntry[]
  timeline?: TimelineEvent[]
  substitutions?: ExternalSubstitution[]
  rosterQuality?: ExternalRosterQuality
  warnings?: string[]
  provenance?: Record<string, unknown>
}

export type ManualMatchImportRequest = {
  /** ManualMatchEvent intentionally has no provider-owned team/provider fields. */
  event: Omit<ExternalEvent, 'home' | 'away' | 'provider'>
  teams: { home: ExternalTeam; away: ExternalTeam }
  players: ExternalPlayer[]
  lineup?: ExternalLineupEntry[]
  timeline?: TimelineEvent[]
  substitutions?: ExternalSubstitution[]
  provenance?: {
    label?: string | null
    reference?: string | null
    capturedAt?: string | null
    notes?: string | null
  } | null
}

export type SceneDocument = {
  id: string
  title: string
  version: number
  /** Full-document CAS token; absent only on legacy/offline fixtures. */
  revision?: number
  duration: number
  payload: {
    pitch: { length: number; width: number }
    matchBinding: PersistedMatchBinding | null
    videoAsset?: {
      id: string
      filename: string
      mediaUrl: string
      posterUrl: string
      fps: number
      analysisFps?: number
      frameCount: number
      processingState: string
      sourceStart?: number
      sourceEnd?: number
      parentSceneId?: string
      selectedSegmentId?: string
      primarySceneId?: string
      segments?: VideoSegment[]
      segmentLayout?: SegmentLayout
      multiPass?: MultiPassSummary
      reconstruction?: {
        runId?: string
        runRevision?: number
        inputFingerprint?: string
        trackObservationSchemaVersion?: number
        status: 'queued' | 'processing' | 'ready' | 'failed'
        processingStatus?: ProcessingStatus
        qualityVerdict?: QualityVerdict
        qualityReport?: ReconstructionQualityReport
        quality?: ReconstructionQuality
        calibration?: CalibrationEvidence
        calibrationFrames?: CalibrationFrameEvidence[]
        progress?: ReconstructionProgress
        model?: ReconstructionModel
        ballBackend?: BallDetectionBackend
        ballDetection?: {
          schemaVersion: number
          status: 'ready' | 'degraded' | 'failed'
          requestedBackend: BallDetectionBackend
          runtimeModelVersions?: string[]
          frameCount: number
          candidateCount: number
          framesWithCandidates: number
          fallbackFrameCount?: number
          failedFrameCount?: number
          observedFrameCount: number
          inferredFrameCount: number
          occludedFrameCount: number
          observedCoverage?: number | null
          publishedCoverage?: number | null
          backendCounts?: Record<string, number>
          frameSource?: {
            source?: string
            frameRate?: number
            frameCount?: number
            cacheKey?: string
            cacheHit?: boolean
            detectionCacheHit?: boolean
            detectionCacheStored?: boolean
            detectionCacheKey?: string
            detectionCacheWriteError?: string
          }
          tracking?: Record<string, unknown>
        }
        frameCount?: number
        trackCount?: number
        ballSamples?: number
        coordinateSpace?: string
        cameraMotionCompensated?: boolean
        inputRange?: {
          sourceStart: number
          sourceEnd: number
          firstFrameTime: number
          lastFrameTime: number
        }
        diagnostics?: {
          meanPersonDetections: number
          framesWithBall: number
          rawTrackCount: number
          stableTrackCount: number
          acceptedTrackCount: number
          rawProjectedObservationCount?: number
          discardedProjectedObservationCount?: number
          splitTrajectoryCount?: number
          preFilterMaximumSpeedMetresPerSecond?: number | null
          identityObservationCoverage?: number
          metricObservationCoverage?: number
          identity?: CanonicalIdentityDiagnostics
        }
        previousResult?: {
          completedAt?: string | null
          trackCount: number
          ballSamples: number
          calibrationStatus?: string | null
        }
        modelComparison?: ModelComparisonReport
        frameAnnotations?: FrameAnnotation[]
        multiPassEvidence?: MultiPassSummary['consensus']
        multiPassBallSupport?: MultiPassSummary['ballSupport']
        pitchCalibration?: {
          status: 'ready' | 'review' | 'approximate' | 'fallback' | 'rejected'
          method: string
          confidence?: number
          supportedLines?: number
          matchedCurves?: number
          meanLineScore?: number
          rectangle?: string
          pitchSide?: 'left' | 'right' | null
          imageToPitch?: number[][]
          preset?: PitchCalibrationPreset
          sceneTime?: number
          frameIndex?: number
          alignmentError?: number | null
          anchors?: PitchCalibrationAnchor[]
          reason?: string
        }
        pitchOrientation?: PitchOrientation
        error?: string | null
        warnings?: string[]
      }
    }
    teams: Team[]
    canonicalPeople?: CanonicalPerson[]
    identityReviewDecisions?: {
      rosterRejections?: Array<{
        id: string
        schema: 'roster-candidate-rejection-v1' | string
        canonicalPersonId: string
        externalPlayerId: string
        anchorObservationId?: string | null
        createdAt?: string
      }>
    }
    tracks: Track[]
    /** Compact reviewed semantics; dense pose artifacts stay outside scene JSON. */
    playerActions?: PlayerAction[]
    ball: {
      /** Which trajectory is currently rendered by the editor and player. */
      mode?: BallTrajectoryMode
      keyframes: Keyframe[]
      /** Last detector result, retained while a human-authored path is active. */
      automaticKeyframes?: Keyframe[]
      /** Human-authored control points. Positions between them are interpolated. */
      manualKeyframes?: Keyframe[]
      diagnostics?: {
        algorithm?: string
        status?: string
        frameCount?: number
        observedFrameCount?: number
        inferredFrameCount?: number
        occludedFrameCount?: number
        observedCoverage?: number
        publishedCoverage?: number
        pathCostMargin?: number | null
        worldProjectionStatus?: string
        gaps?: {
          longestGapSeconds?: number | null
          [key: string]: unknown
        }
        path?: Array<Record<string, unknown>>
        [key: string]: unknown
      }
    }
    eventBindings: EventBinding[]
    cameraCuts: Array<{ t: number; preset: string }>
  }
}

export type ExternalTeam = {
  id: string
  name: string
  badge?: string | null
}

export type ExternalEvent = {
  id: string
  provider?: MatchDataProviderId | null
  name: string
  date?: string | null
  time?: string | null
  status?: string | null
  league?: string | null
  season?: string | null
  home: ExternalTeam
  away: ExternalTeam
  home_score?: number | null
  away_score?: number | null
  thumbnail?: string | null
}

export type ExternalPlayer = {
  id: string
  name: string
  team_id?: string | null
  team_name?: string | null
  position?: string | null
  number?: string | null
  thumbnail?: string | null
  lineup_role?: 'starter' | 'substitute' | 'unknown'
  lineup_order?: number | null
}

export type TimelineEvent = {
  id: string
  minute?: number | null
  type: string
  label: string
  player_id?: string | null
  player_name?: string | null
  team_id?: string | null
  team_name?: string | null
  secondary_player_id?: string | null
  secondary_player_name?: string | null
  detail?: string | null
}

export type EventBundle = {
  /** Provider id (`api-football`, `thesportsdb`, …) or a user-owned import. */
  source: MatchDataProviderId | 'manual'
  event: ExternalEvent
  players: ExternalPlayer[]
  lineup: ExternalLineupEntry[]
  timeline: TimelineEvent[]
  substitutions: ExternalSubstitution[]
  roster_quality?: {
    status: 'automatic-ready' | 'partial' | 'unavailable'
    player_count: number
    home_player_count: number
    away_player_count: number
    automatic_identity_eligible: boolean
    manual_identity_eligible: boolean
    reasons: string[]
  } | null
  fetched_at: string
  warnings: string[]
}

export type IdentityReviewCropDiagnostic = {
  status?: string | null
  usable?: boolean | null
  rejectionReasons?: string[]
  number?: string | null
  confidence?: number | null
  [key: string]: unknown
}

export type IdentityReviewRepresentativeObservation = {
  observationId: string
  frameIndex: number
  sourceFrameIndex?: number | null
  sceneTime: number
  sourceTime?: number | null
  bbox: TrackObservation['bbox']
  confidence?: number | null
  reviewQuality?: number | null
  cropUrl?: string | null
  reid?: IdentityReviewCropDiagnostic | null
  jerseyOcr?: IdentityReviewCropDiagnostic | null
}

export type IdentityReviewResolutionState = 'conflict' | 'suggested' | 'anonymous' | 'bound' | 'excluded'

export type IdentityReviewItem = {
  canonicalPersonId: string
  displayName: string
  identityStatus: CanonicalPersonStatus
  identityConfidence?: number | null
  identitySource?: string | null
  teamId?: string | null
  role?: CanonicalPersonRole | null
  jerseyNumber?: string | null
  candidateNumber?: string | null
  externalPlayerId?: string | null
  renderTrackId?: string | null
  observationCount: number
  resolutionState: IdentityReviewResolutionState
  priority: number
  representativeObservations: IdentityReviewRepresentativeObservation[]
  evidence: CanonicalIdentityEvidence[]
  rosterCandidates: CanonicalRosterCandidate[]
  conflicts: CanonicalIdentityConflict[]
}

export type IdentityReviewWorkerHealth = {
  configured?: boolean | null
  status: string
  backend?: string | null
  modelVersion?: string | null
  providerVersion?: string | null
  detail?: string | null
  [key: string]: unknown
}

export type IdentityReviewResponse = {
  sceneId: string
  revision: number
  matchBinding: {
    scope?: MatchBindingScope
    projectSceneId?: string | null
    inherited?: boolean
    source?: string | null
    eventId?: string | null
    fetchedAt?: string | null
    roster: {
      status: 'ready' | 'incomplete' | 'review' | 'unavailable'
      playerCount: number
      complete: boolean
      automaticIdentityEligible: boolean
      manualIdentityEligible: boolean
      reasons: string[]
      warnings: string[]
    }
  }
  workers: {
    identity?: IdentityReviewWorkerHealth
    /** Compatibility with a future API spelling. */
    reid?: IdentityReviewWorkerHealth
    jerseyOcr?: IdentityReviewWorkerHealth
  }
  summary: {
    canonicalPersonCount: number
    boundCount: number
    suggestedCount: number
    conflictCount: number
    anonymousCount: number
    excludedCount: number
  }
  items: IdentityReviewItem[]
}

export type SceneMatchBindingResponse = {
  scene: SceneDocument
  bundle: EventBundle
}

export type SceneSummary = {
  id: string
  title: string
  duration: number
  kind: 'video' | 'segment' | 'multi-pass' | 'demo'
  parent_scene_id?: string | null
  updated_at?: string | null
}

export type VideoAsset = {
  id: string
  filename: string
  original_name: string
  content_type: string
  status: 'queued' | 'processing' | 'ready' | 'failed'
  stage: string
  progress: number
  duration?: number | null
  width?: number | null
  height?: number | null
  fps?: number | null
  frame_count: number
  scene_id?: string | null
  media_url?: string | null
  poster_url?: string | null
  error?: string | null
  created_at?: string | null
}
