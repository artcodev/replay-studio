import type { CalibrationFrameEvidence } from '../types/calibration'
import type { ContactPointProfile } from '../types/reconstruction'
import type { Track, TrackObservation } from '../types/tracking'

export const PLAYER_SPEED_QA_METRES_PER_SECOND = 14
export const IDENTITY_SPLIT_METRES_PER_SECOND = 25

export type PitchPoint = { x: number; z: number }
export type ImagePoint = { x: number; y: number }
export type ProjectionDebugSeverity = 'ok' | 'speed-violation' | 'identity-split' | 'unprojected'

export type ProjectionDebugSample = {
  observation: TrackObservation
  calibration: CalibrationFrameEvidence | null
  contactPoint: ImagePoint
  storedPitch: PitchPoint | null
  matrixPitch: PitchPoint | null
  effectivePitch: PitchPoint | null
  projectionMethod: 'frame-homography' | 'stored-observation' | 'unavailable'
  previous: ProjectionDebugSample | null
  elapsedSeconds: number | null
  imageDeltaPixels: number | null
  cameraCompensatedImageDeltaPixels: number | null
  pitchDeltaMetres: number | null
  imageMotionPitchDeltaMetres: number | null
  calibrationMotionPitchDeltaMetres: number | null
  calibrationContinuityDeltaMetres: number | null
  speedMetresPerSecond: number | null
  storedMatrixDeltaMetres: number | null
  severity: ProjectionDebugSeverity
}

export type ProjectionPopulationEdge = {
  comparedTrackCount: number
  speedViolationCount: number
  identitySplitCount: number
}

function distance(left: { x: number; z: number }, right: { x: number; z: number }) {
  return Math.hypot(right.x - left.x, right.z - left.z)
}

function imageDistance(left: ImagePoint, right: ImagePoint) {
  return Math.hypot(right.x - left.x, right.y - left.y)
}

function finiteHomography(matrix: number[][] | null | undefined): number[][] | null {
  if (
    !matrix
    || matrix.length !== 3
    || matrix.some((row) => !Array.isArray(row) || row.length !== 3)
  ) return null
  const normalized = matrix.map((row) => row.map(Number))
  return normalized.some((row) => row.some((value) => !Number.isFinite(value)))
    ? null
    : normalized
}

function multiplyHomographies(
  left: number[][] | null | undefined,
  right: number[][] | null | undefined,
): number[][] | null {
  const a = finiteHomography(left)
  const b = finiteHomography(right)
  if (!a || !b) return null
  return a.map((row) => b[0].map((_, column) => (
    row.reduce((sum, value, index) => sum + value * b[index][column], 0)
  )))
}

function transformImagePoint(
  matrix: number[][] | null | undefined,
  point: ImagePoint,
): ImagePoint | null {
  const value = finiteHomography(matrix)
  if (!value) return null
  const denominator = value[2][0] * point.x + value[2][1] * point.y + value[2][2]
  if (Math.abs(denominator) < 1e-8) return null
  const x = (value[0][0] * point.x + value[0][1] * point.y + value[0][2]) / denominator
  const y = (value[1][0] * point.x + value[1][1] * point.y + value[1][2]) / denominator
  return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null
}

export function bboxBottomCentre(observation: TrackObservation): ImagePoint {
  return {
    x: observation.bbox.x + observation.bbox.width / 2,
    y: observation.bbox.y + observation.bbox.height,
  }
}

export function storedObservationPitch(observation: TrackObservation): PitchPoint | null {
  const value = observation.rawPitch ?? observation.pitch
  return value ? { x: Number(value.x), z: Number(value.z) } : null
}

export function projectImagePoint(
  matrix: number[][] | null | undefined,
  point: ImagePoint,
  pitch: { length: number; width: number },
): PitchPoint | null {
  const projected = transformImagePoint(matrix, point)
  if (!projected) return null
  const { x, y: z } = projected
  const halfLength = pitch.length / 2
  const halfWidth = pitch.width / 2
  if (x < -halfLength - 4 || x > halfLength + 4 || z < -halfWidth - 4 || z > halfWidth + 4) {
    return null
  }
  return {
    x: Math.max(-halfLength, Math.min(halfLength, x)),
    z: Math.max(-halfWidth, Math.min(halfWidth, z)),
  }
}

function severityForSpeed(speed: number | null, projected: boolean): ProjectionDebugSeverity {
  if (!projected) return 'unprojected'
  if (speed != null && speed > IDENTITY_SPLIT_METRES_PER_SECOND) return 'identity-split'
  if (speed != null && speed > PLAYER_SPEED_QA_METRES_PER_SECOND) return 'speed-violation'
  return 'ok'
}

/**
 * Replays the exact bbox-bottom → stored frame homography path used by
 * reconstruction. Pose-feet runs fall back to their persisted observation
 * because the pose contact pixel is intentionally not reconstructed from a box.
 */
export function buildTrackProjectionDebugSamples(
  observations: readonly TrackObservation[] | null | undefined,
  calibrationFrames: readonly CalibrationFrameEvidence[],
  pitch: { length: number; width: number },
  contactPointProfile: ContactPointProfile = 'bbox-bottom',
): ProjectionDebugSample[] {
  const calibrationByFrame = new Map(
    calibrationFrames.map((frame) => [frame.sourceFrameIndex, frame]),
  )
  const ordered = [...(observations ?? [])]
    .filter((observation) => observation.bbox && Number.isFinite(observation.sceneTime))
    .sort((left, right) => left.sceneTime - right.sceneTime || left.frameIndex - right.frameIndex)
  const result: ProjectionDebugSample[] = []
  for (const observation of ordered) {
    const calibration = calibrationByFrame.get(
      observation.sourceFrameIndex ?? observation.frameIndex,
    ) ?? null
    const contactPoint = bboxBottomCentre(observation)
    const storedPitch = storedObservationPitch(observation)
    const matrixPitch = contactPointProfile === 'bbox-bottom'
      ? projectImagePoint(calibration?.imageToPitch, contactPoint, pitch)
      : null
    const effectivePitch = matrixPitch ?? storedPitch
    const previous = [...result].reverse().find((sample) => sample.effectivePitch) ?? null
    const elapsedSeconds = previous
      ? observation.sceneTime - previous.observation.sceneTime
      : null
    const pitchDeltaMetres = previous?.effectivePitch && effectivePitch
      ? distance(previous.effectivePitch, effectivePitch)
      : null
    // Freeze the previous frame matrix to isolate contact-point movement, then
    // change only the matrix at the current contact point. A synchronized
    // matrix contribution across players is camera jitter, not player motion.
    const previousMatrixAtCurrentContact = previous?.calibration?.imageToPitch
      ? projectImagePoint(previous.calibration.imageToPitch, contactPoint, pitch)
      : null
    const imageMotionPitchDeltaMetres = previous?.matrixPitch && previousMatrixAtCurrentContact
      ? distance(previous.matrixPitch, previousMatrixAtCurrentContact)
      : null
    const calibrationMotionPitchDeltaMetres = previousMatrixAtCurrentContact && matrixPitch
      ? distance(previousMatrixAtCurrentContact, matrixPitch)
      : null
    const adjacentCalibration = Boolean(
      previous?.calibration
      && calibration
      && calibration.sampleIndex === previous.calibration.sampleIndex + 1,
    )
    const currentToPrevious = adjacentCalibration
      && calibration?.cameraMotion?.status === 'estimated'
      ? calibration.cameraMotion.currentToPrevious
      : null
    const currentContactInPreviousImage = currentToPrevious
      ? transformImagePoint(currentToPrevious, contactPoint)
      : null
    const cameraCompensatedImageDeltaPixels = previous && currentContactInPreviousImage
      ? imageDistance(previous.contactPoint, currentContactInPreviousImage)
      : null
    const motionExpectedMatrix = previous?.calibration?.imageToPitch && currentToPrevious
      ? multiplyHomographies(previous.calibration.imageToPitch, currentToPrevious)
      : null
    const motionExpectedPitch = motionExpectedMatrix
      ? projectImagePoint(motionExpectedMatrix, contactPoint, pitch)
      : null
    const calibrationContinuityDeltaMetres = motionExpectedPitch && matrixPitch
      ? distance(motionExpectedPitch, matrixPitch)
      : null
    const speedMetresPerSecond = elapsedSeconds != null
      && elapsedSeconds > 1e-6
      && pitchDeltaMetres != null
      ? pitchDeltaMetres / elapsedSeconds
      : null
    const storedMatrixDeltaMetres = storedPitch && matrixPitch
      ? distance(storedPitch, matrixPitch)
      : null
    result.push({
      observation,
      calibration,
      contactPoint,
      storedPitch,
      matrixPitch,
      effectivePitch,
      projectionMethod: matrixPitch
        ? 'frame-homography'
        : storedPitch
          ? 'stored-observation'
          : 'unavailable',
      previous,
      elapsedSeconds,
      imageDeltaPixels: previous ? imageDistance(previous.contactPoint, contactPoint) : null,
      cameraCompensatedImageDeltaPixels,
      pitchDeltaMetres,
      imageMotionPitchDeltaMetres,
      calibrationMotionPitchDeltaMetres,
      calibrationContinuityDeltaMetres,
      speedMetresPerSecond,
      storedMatrixDeltaMetres,
      severity: severityForSpeed(speedMetresPerSecond, Boolean(effectivePitch)),
    })
  }
  return result
}

export function nearestProjectionDebugSample(
  samples: readonly ProjectionDebugSample[],
  currentTime: number,
): { sample: ProjectionDebugSample; index: number; timeDistance: number; active: boolean } | null {
  if (!samples.length) return null
  let index = 0
  for (let candidate = 1; candidate < samples.length; candidate += 1) {
    if (
      Math.abs(samples[candidate].observation.sceneTime - currentTime)
      < Math.abs(samples[index].observation.sceneTime - currentTime)
    ) index = candidate
  }
  const timeDistance = Math.abs(samples[index].observation.sceneTime - currentTime)
  return { sample: samples[index], index, timeDistance, active: timeDistance <= 0.08 }
}

export function projectionPopulationEdge(
  tracks: readonly Track[],
  calibrationFrames: readonly CalibrationFrameEvidence[],
  pitch: { length: number; width: number },
  contactPointProfile: ContactPointProfile,
  fromFrame: number | null | undefined,
  toFrame: number,
): ProjectionPopulationEdge {
  if (fromFrame == null) {
    return { comparedTrackCount: 0, speedViolationCount: 0, identitySplitCount: 0 }
  }
  let comparedTrackCount = 0
  let speedViolationCount = 0
  let identitySplitCount = 0
  for (const track of tracks) {
    const samples = buildTrackProjectionDebugSamples(
      track.observations,
      calibrationFrames,
      pitch,
      contactPointProfile,
    )
    const current = samples.find((sample) => sample.observation.frameIndex === toFrame)
    if (!current || current.previous?.observation.frameIndex !== fromFrame) continue
    comparedTrackCount += 1
    if ((current.speedMetresPerSecond ?? 0) > PLAYER_SPEED_QA_METRES_PER_SECOND) {
      speedViolationCount += 1
    }
    if ((current.speedMetresPerSecond ?? 0) > IDENTITY_SPLIT_METRES_PER_SECOND) {
      identitySplitCount += 1
    }
  }
  return { comparedTrackCount, speedViolationCount, identitySplitCount }
}
