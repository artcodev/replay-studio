export type FrameDetectionPoint = {
  x: number
  y: number
}

export type FrameDetectionBox = {
  x: number
  y: number
  width: number
  height: number
}

export type FrameDetectionHitTarget = {
  id: string
  bbox: FrameDetectionBox
  confidence?: number
}

export type FrameDetectionHitOptions = {
  /** Minimum width and height of the clickable area, expressed in frame units. */
  minimumTargetSize?: number
}

type RankedDetection<T> = {
  detection: T
  actualContainment: boolean
  area: number
  centerDistanceSquared: number
  confidence: number
}

function finiteOr(value: number, fallback = 0) {
  return Number.isFinite(value) ? value : fallback
}

function normalizeBox(box: FrameDetectionBox) {
  const sourceX = finiteOr(box.x)
  const sourceY = finiteOr(box.y)
  const sourceWidth = finiteOr(box.width)
  const sourceHeight = finiteOr(box.height)
  const x = sourceWidth < 0 ? sourceX + sourceWidth : sourceX
  const y = sourceHeight < 0 ? sourceY + sourceHeight : sourceY
  const width = Math.abs(sourceWidth)
  const height = Math.abs(sourceHeight)
  return { x, y, width, height }
}

function containsPoint(
  box: { x: number; y: number; width: number; height: number },
  point: FrameDetectionPoint,
) {
  return point.x >= box.x
    && point.x <= box.x + box.width
    && point.y >= box.y
    && point.y <= box.y + box.height
}

/**
 * Returns every detection whose real or minimum-size hit area contains the
 * point. The ordering is deterministic and favours what the user most likely
 * intended when distant players or duplicate detections overlap.
 */
export function orderedFrameDetectionHits<T extends FrameDetectionHitTarget>(
  detections: readonly T[],
  point: FrameDetectionPoint,
  options: FrameDetectionHitOptions = {},
): T[] {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return []

  const minimumTargetSize = Math.max(0, finiteOr(options.minimumTargetSize ?? 0))
  const ranked: Array<RankedDetection<T>> = []

  for (const detection of detections) {
    const box = normalizeBox(detection.bbox)
    const centerX = box.x + box.width / 2
    const centerY = box.y + box.height / 2
    const targetWidth = Math.max(box.width, minimumTargetSize)
    const targetHeight = Math.max(box.height, minimumTargetSize)
    const hitBox = {
      x: centerX - targetWidth / 2,
      y: centerY - targetHeight / 2,
      width: targetWidth,
      height: targetHeight,
    }
    if (!containsPoint(hitBox, point)) continue

    ranked.push({
      detection,
      actualContainment: containsPoint(box, point),
      area: box.width * box.height,
      centerDistanceSquared: (point.x - centerX) ** 2 + (point.y - centerY) ** 2,
      confidence: finiteOr(detection.confidence ?? 0),
    })
  }

  ranked.sort((left, right) => {
    if (left.actualContainment !== right.actualContainment) {
      return left.actualContainment ? -1 : 1
    }
    if (left.area !== right.area) return left.area - right.area
    if (left.centerDistanceSquared !== right.centerDistanceSquared) {
      return left.centerDistanceSquared - right.centerDistanceSquared
    }
    if (left.confidence !== right.confidence) return right.confidence - left.confidence
    return left.detection.id.localeCompare(right.detection.id)
  })

  return ranked.map(({ detection }) => detection)
}

/**
 * Selects the first ranked hit, or the next hit when the previously selected
 * candidate is still under the pointer. This makes overlapping boxes
 * reachable through repeated clicks without changing their ranking.
 */
export function selectFrameDetectionHit<T extends FrameDetectionHitTarget>(
  detections: readonly T[],
  point: FrameDetectionPoint,
  options: FrameDetectionHitOptions & { previousCandidateId?: string | null } = {},
): T | null {
  const candidates = orderedFrameDetectionHits(detections, point, options)
  if (!candidates.length) return null

  const previousIndex = options.previousCandidateId
    ? candidates.findIndex((candidate) => candidate.id === options.previousCandidateId)
    : -1
  return candidates[previousIndex < 0 ? 0 : (previousIndex + 1) % candidates.length]
}
