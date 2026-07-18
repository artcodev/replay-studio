import type { PlayerAction } from './playerActions'
import type { FrameAnnotation, ModelComparisonReport } from './analysis'
import type { CalibrationEvidence, PitchCalibrationAnchor, PitchCalibrationPreset, PitchOrientation } from './calibration'
import type { CanonicalIdentityDiagnostics, CanonicalPerson } from './identity'
import type { MultiPassSummary, SegmentLayout, VideoSegment } from './media'
import type { BallDetectionBackend, BallTrajectoryMode, ProcessingStatus, QualityVerdict, ReconstructionArtifactManifest, ReconstructionModel, ReconstructionProgress, ReconstructionQuality, ReconstructionQualityReport } from './reconstruction'
import type { Keyframe, Track } from './tracking'

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

export type SceneDocument = {
  id: string
  title: string
  version: number
  /** Full-document CAS token required by every mutation. */
  revision: number
  duration: number
  payload: {
    pitch: { length: number; width: number }
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
        matchSnapshotRef?: {
          id: string
          contentHash: string
          schemaVersion: number
        }
        currentInputFingerprint?: string
        inputState?: 'current' | 'stale' | 'unknown'
        inputStateReason?: 'reconstruction-input-changed' | string
        trackObservationSchemaVersion?: number
        status: 'queued' | 'processing' | 'ready' | 'cancelled' | 'failed'
        processingStatus?: ProcessingStatus
        qualityVerdict?: QualityVerdict
        qualityReport?: ReconstructionQualityReport
        quality?: ReconstructionQuality
        artifactManifest?: ReconstructionArtifactManifest
        calibration?: CalibrationEvidence
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
          frames?: Array<Record<string, unknown>>
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
      keyframeCount?: number
      automaticKeyframeCount?: number
      manualKeyframeCount?: number
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

export type SceneVideoAsset = NonNullable<SceneDocument['payload']['videoAsset']>

export type SceneSummary = {
  id: string
  title: string
  duration: number
  kind: 'video' | 'segment' | 'multi-pass' | 'demo'
  parent_scene_id?: string | null
  updated_at?: string | null
}
