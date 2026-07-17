export type VideoReviewTransform = {
  scale: number
  x: number
  y: number
}

export const VIDEO_REVIEW_MIN_SCALE = 1
export const VIDEO_REVIEW_MAX_SCALE = 4

export type MediaViewportRect = {
  left: number
  top: number
  width: number
  height: number
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value))
}

export function clampVideoReviewTransform(
  transform: VideoReviewTransform,
  width: number,
  height: number,
): VideoReviewTransform {
  const scale = clamp(transform.scale, VIDEO_REVIEW_MIN_SCALE, VIDEO_REVIEW_MAX_SCALE)
  if (scale === VIDEO_REVIEW_MIN_SCALE || width <= 0 || height <= 0) {
    return { scale, x: 0, y: 0 }
  }
  const maximumX = width * (scale - 1) / 2
  const maximumY = height * (scale - 1) / 2
  return {
    scale,
    x: clamp(transform.x, -maximumX, maximumX),
    y: clamp(transform.y, -maximumY, maximumY),
  }
}

export function zoomVideoReviewTransform(
  transform: VideoReviewTransform,
  nextScale: number,
  focalX: number,
  focalY: number,
  width: number,
  height: number,
) {
  const scale = clamp(nextScale, VIDEO_REVIEW_MIN_SCALE, VIDEO_REVIEW_MAX_SCALE)
  if (scale === transform.scale) return clampVideoReviewTransform(transform, width, height)
  if (scale === VIDEO_REVIEW_MIN_SCALE) return { scale, x: 0, y: 0 }
  const ratio = scale / transform.scale
  return clampVideoReviewTransform({
    scale,
    x: focalX - (focalX - transform.x) * ratio,
    y: focalY - (focalY - transform.y) * ratio,
  }, width, height)
}

export function panVideoReviewTransform(
  transform: VideoReviewTransform,
  deltaX: number,
  deltaY: number,
  width: number,
  height: number,
) {
  return clampVideoReviewTransform({
    ...transform,
    x: transform.x + deltaX,
    y: transform.y + deltaY,
  }, width, height)
}

export function clientPointToContainedMedia(
  clientX: number,
  clientY: number,
  rect: MediaViewportRect,
  mediaWidth: number,
  mediaHeight: number,
) {
  if (rect.width <= 0 || rect.height <= 0 || mediaWidth <= 0 || mediaHeight <= 0) return null
  const scale = Math.min(rect.width / mediaWidth, rect.height / mediaHeight)
  const offsetX = (rect.width - mediaWidth * scale) / 2
  const offsetY = (rect.height - mediaHeight * scale) / 2
  return {
    x: Math.max(0, Math.min(mediaWidth, (clientX - rect.left - offsetX) / scale)),
    y: Math.max(0, Math.min(mediaHeight, (clientY - rect.top - offsetY) / scale)),
  }
}
