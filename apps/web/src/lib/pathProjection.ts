import type { CalibrationFrameEvidence } from '../types/calibration'
import type { PathTrackingSegment } from './pathTracking'
import {
  invertHomography,
  projectHomographyPoint,
  type Matrix3,
} from './pitchProjection'

export type PathProjectionMode = 'exact' | 'interpolated' | 'nearest'

export type PathProjectionContext = {
  mode: PathProjectionMode
  width: number
  height: number
  alpha: number
  timeOffsetSeconds: number
  lowerSceneTime: number
  upperSceneTime: number
  interpolationIntervalSeconds: number
  pitchToImage: Matrix3
  uncertaintyMetres: number | null
}

export type ProjectedPathPoint = { x: number; y: number }
export type ProjectedPathEdge = { start: ProjectedPathPoint; end: ProjectedPathPoint }

export const PATH_PROJECTION_EXACT_TOLERANCE_SECONDS = 0.001
export const PATH_PROJECTION_MAX_INTERPOLATION_GAP_SECONDS = 0.8
export const PATH_PROJECTION_MAX_NEAREST_AGE_SECONDS = 0.25

type UsableFrame = {
  frame: CalibrationFrameEvidence
  width: number
  height: number
  imageToPitch: Matrix3
  pitchToImage: Matrix3
}

function finitePositive(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}

function finiteMatrix3(value: number[][]): Matrix3 | null {
  if (
    value.length !== 3
    || value.some((row) => row.length !== 3 || row.some((item) => !Number.isFinite(item)))
  ) return null
  return value as Matrix3
}

/** Mirror the server's consistently-scaled, bounded interpolation contract. */
export function normalisePathProjectionHomography(
  matrix: number[][],
  width: number,
  height: number,
): Matrix3 | null {
  const value = finiteMatrix3(matrix)
  if (!value || !finitePositive(width) || !finitePositive(height)) return null
  const scale = value[2][2]
  if (!Number.isFinite(scale) || Math.abs(scale) < 1e-10) return null
  const normalized = value.map((row) => row.map((item) => item / scale)) as Matrix3
  const inverse = invertHomography(normalized)
  if (!inverse) return null

  // A cheap Frobenius condition estimate catches the same pathological class
  // as the server SVD gate without shipping a numerical linear-algebra bundle.
  const matrixNorm = Math.hypot(...normalized.flat())
  const inverseNorm = Math.hypot(...inverse.flat())
  if (!Number.isFinite(matrixNorm * inverseNorm) || matrixNorm * inverseNorm > 1e12) return null

  const denominatorRowNorm = Math.hypot(...normalized[2])
  for (const yFraction of [0.58, 0.78, 0.96]) {
    for (const xFraction of [0.08, 0.5, 0.92]) {
      const probe: [number, number, number] = [width * xFraction, height * yFraction, 1]
      const denominator = normalized[2][0] * probe[0]
        + normalized[2][1] * probe[1]
        + normalized[2][2]
      const denominatorScale = denominatorRowNorm * Math.hypot(...probe)
      if (Math.abs(denominator) <= 1e-9 * Math.max(1, denominatorScale)) return null
      if (!projectHomographyPoint([probe[0], probe[1]], normalized)) return null
    }
  }
  return normalized
}

export function interpolatePathProjectionHomography(
  lower: number[][],
  upper: number[][],
  alpha: number,
  width: number,
  height: number,
): Matrix3 | null {
  if (!(alpha > 0 && alpha < 1)) return null
  const normalizedLower = normalisePathProjectionHomography(lower, width, height)
  const normalizedUpper = normalisePathProjectionHomography(upper, width, height)
  if (!normalizedLower || !normalizedUpper) return null
  const candidate = normalizedLower.map((row, rowIndex) => row.map(
    (value, columnIndex) => (
      value * (1 - alpha) + normalizedUpper[rowIndex][columnIndex] * alpha
    ),
  ))
  return normalisePathProjectionHomography(candidate, width, height)
}

function frameUncertainty(frame: CalibrationFrameEvidence) {
  const uncertainty = frame.positionUncertaintyMetres ?? frame.uncertainty?.p95Metres
  return typeof uncertainty === 'number' && Number.isFinite(uncertainty)
    ? uncertainty
    : null
}

function usableFrame(frame: CalibrationFrameEvidence): UsableFrame | null {
  if (
    frame.status !== 'accepted'
    || !frame.imageToPitch
    || !finitePositive(frame.frameWidth)
    || !finitePositive(frame.frameHeight)
  ) return null
  const imageToPitch = normalisePathProjectionHomography(
    frame.imageToPitch,
    frame.frameWidth,
    frame.frameHeight,
  )
  if (!imageToPitch) return null
  const pitchToImage = invertHomography(imageToPitch)
  if (!pitchToImage) return null
  return {
    frame,
    width: frame.frameWidth,
    height: frame.frameHeight,
    imageToPitch,
    pitchToImage,
  }
}

function exactContext(frame: UsableFrame, currentTime: number): PathProjectionContext {
  const timeOffsetSeconds = Math.abs(frame.frame.sceneTime - currentTime)
  const baseUncertainty = frameUncertainty(frame.frame)
  return {
    mode: timeOffsetSeconds <= PATH_PROJECTION_EXACT_TOLERANCE_SECONDS
      ? 'exact'
      : 'nearest',
    width: frame.width,
    height: frame.height,
    alpha: 0,
    timeOffsetSeconds,
    lowerSceneTime: frame.frame.sceneTime,
    upperSceneTime: frame.frame.sceneTime,
    interpolationIntervalSeconds: 0,
    pitchToImage: frame.pitchToImage,
    uncertaintyMetres: baseUncertainty === null
      ? null
      : Math.min(12, baseUncertainty + timeOffsetSeconds * 2),
  }
}

/**
 * Resolve the camera for the video frame without carrying a homography across
 * an unobserved interval. Interpolation is allowed only across two adjacent,
 * QA-accepted samples joined by a reliable camera-motion edge.
 */
export function resolvePathProjectionContext(
  frames: CalibrationFrameEvidence[],
  currentTime: number,
  options: {
    maxInterpolationGapSeconds?: number
    maxNearestAgeSeconds?: number
  } = {},
): PathProjectionContext | null {
  if (!Number.isFinite(currentTime)) return null
  const ordered = frames
    .filter((frame) => Number.isFinite(frame.sceneTime))
    .sort((left, right) => left.sceneTime - right.sceneTime)
  if (!ordered.length) return null

  const exact = ordered.filter((frame) => (
    Math.abs(frame.sceneTime - currentTime) <= PATH_PROJECTION_EXACT_TOLERANCE_SECONDS
  ))
  if (exact.length) {
    const best = exact
      .map(usableFrame)
      .filter((frame): frame is UsableFrame => Boolean(frame))
      .sort((left, right) => (right.frame.confidence ?? 0) - (left.frame.confidence ?? 0))[0]
    // An explicit rejected/missing sample is a real calibration gap. Do not
    // conceal it with a nearby camera from another timestamp.
    return best ? exactContext(best, currentTime) : null
  }

  const upperIndex = ordered.findIndex((frame) => frame.sceneTime > currentTime)
  const maxNearestAge = options.maxNearestAgeSeconds ?? PATH_PROJECTION_MAX_NEAREST_AGE_SECONDS
  if (upperIndex <= 0) {
    const nearest = usableFrame(ordered[upperIndex < 0 ? ordered.length - 1 : 0])
    if (!nearest || Math.abs(nearest.frame.sceneTime - currentTime) > maxNearestAge) return null
    return exactContext(nearest, currentTime)
  }

  const lower = ordered[upperIndex - 1]
  const upper = ordered[upperIndex]
  const lowerUsable = usableFrame(lower)
  const upperUsable = usableFrame(upper)
  const interval = upper.sceneTime - lower.sceneTime
  const maxInterpolationGap = options.maxInterpolationGapSeconds
    ?? PATH_PROJECTION_MAX_INTERPOLATION_GAP_SECONDS
  if (
    !lowerUsable
    || !upperUsable
    || !Number.isFinite(interval)
    || interval <= 0
    || interval > maxInterpolationGap
    || lowerUsable.width !== upperUsable.width
    || lowerUsable.height !== upperUsable.height
    || upper.cameraMotion?.status !== 'estimated'
  ) return null

  const alpha = (currentTime - lower.sceneTime) / interval
  const imageToPitch = interpolatePathProjectionHomography(
    lowerUsable.imageToPitch,
    upperUsable.imageToPitch,
    alpha,
    lowerUsable.width,
    lowerUsable.height,
  )
  if (!imageToPitch) return null
  const pitchToImage = invertHomography(imageToPitch)
  if (!pitchToImage) return null
  const lowerUncertainty = frameUncertainty(lower)
  const upperUncertainty = frameUncertainty(upper)
  const motionConfidence = Math.max(0, Math.min(1, upper.cameraMotion.confidence ?? 0))
  const midpointWeight = 4 * alpha * (1 - alpha)
  const interpolationPenalty = 0.15
    + 0.6 * midpointWeight * interval / Math.max(1e-6, maxInterpolationGap)
    + 0.75 * (1 - motionConfidence)
  const uncertaintyMetres = lowerUncertainty === null || upperUncertainty === null
    ? null
    : Math.min(
      12,
      lowerUncertainty * (1 - alpha) + upperUncertainty * alpha + interpolationPenalty,
    )

  return {
    mode: 'interpolated',
    width: lowerUsable.width,
    height: lowerUsable.height,
    alpha,
    timeOffsetSeconds: Math.min(currentTime - lower.sceneTime, upper.sceneTime - currentTime),
    lowerSceneTime: lower.sceneTime,
    upperSceneTime: upper.sceneTime,
    interpolationIntervalSeconds: interval,
    pitchToImage,
    uncertaintyMetres,
  }
}

/** Project one ground-plane world point through the current video camera. */
export function projectPitchPointInContext(
  context: PathProjectionContext,
  point: { x: number; z: number },
): ProjectedPathPoint | null {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.z)) return null
  return projectHomographyPoint([point.x, point.z], context.pitchToImage)
}

/** Liang–Barsky clipping keeps crossing trail edges instead of dropping them. */
export function clipPathEdgeToFrame(
  start: ProjectedPathPoint,
  end: ProjectedPathPoint,
  width: number,
  height: number,
): ProjectedPathEdge | null {
  if (
    !finitePositive(width)
    || !finitePositive(height)
    || ![start.x, start.y, end.x, end.y].every(Number.isFinite)
  ) return null
  const dx = end.x - start.x
  const dy = end.y - start.y
  const p = [-dx, dx, -dy, dy]
  const q = [start.x, width - start.x, start.y, height - start.y]
  let minimum = 0
  let maximum = 1
  for (let index = 0; index < 4; index += 1) {
    if (Math.abs(p[index]) < 1e-12) {
      if (q[index] < 0) return null
      continue
    }
    const ratio = q[index] / p[index]
    if (p[index] < 0) minimum = Math.max(minimum, ratio)
    else maximum = Math.min(maximum, ratio)
    if (minimum > maximum) return null
  }
  return {
    start: { x: start.x + minimum * dx, y: start.y + minimum * dy },
    end: { x: start.x + maximum * dx, y: start.y + maximum * dy },
  }
}

export function projectPitchEdgeInContext(
  context: PathProjectionContext,
  start: { x: number; z: number },
  end: { x: number; z: number },
): ProjectedPathEdge | null {
  const projectedStart = projectPitchPointInContext(context, start)
  const projectedEnd = projectPitchPointInContext(context, end)
  if (!projectedStart || !projectedEnd) return null
  return clipPathEdgeToFrame(
    projectedStart,
    projectedEnd,
    context.width,
    context.height,
  )
}

export function pathHasVisibleProjection(
  context: PathProjectionContext,
  segments: PathTrackingSegment[],
) {
  return segments.some((segment) => segment.points.slice(1).some((point, index) => (
    projectPitchEdgeInContext(context, segment.points[index], point) !== null
  )))
}
