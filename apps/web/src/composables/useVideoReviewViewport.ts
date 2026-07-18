import { computed, onScopeDispose, ref, watch } from 'vue'
import {
  clampVideoReviewTransform,
  panVideoReviewTransform,
  VIDEO_REVIEW_MIN_SCALE,
  zoomVideoReviewTransform,
  type VideoReviewTransform,
} from '../lib/videoReviewTransform'

type VideoReviewViewportOptions = {
  isZoomBlocked?: () => boolean
  isPanBlocked?: () => boolean
}

/** Owns the zoom/pan interaction state for the source-video viewport. */
export function useVideoReviewViewport(options: VideoReviewViewportOptions = {}) {
  const viewport = ref<HTMLDivElement | null>(null)
  const transform = ref<VideoReviewTransform>({ scale: 1, x: 0, y: 0 })
  const panDrag = ref<{
    pointerId: number
    clientX: number
    clientY: number
    transform: VideoReviewTransform
  } | null>(null)
  let resizeObserver: ResizeObserver | null = null

  const style = computed(() => ({
    transform: `translate3d(${transform.value.x}px, ${transform.value.y}px, 0) scale(${transform.value.scale})`,
  }))
  const zoomPercent = computed(() => Math.round(transform.value.scale * 100))
  const panning = computed(() => panDrag.value !== null)

  function viewportSize() {
    return {
      width: viewport.value?.clientWidth ?? 0,
      height: viewport.value?.clientHeight ?? 0,
    }
  }

  function commit(nextTransform: VideoReviewTransform) {
    const { width, height } = viewportSize()
    transform.value = clampVideoReviewTransform(nextTransform, width, height)
  }

  function reset() {
    panDrag.value = null
    transform.value = { scale: VIDEO_REVIEW_MIN_SCALE, x: 0, y: 0 }
  }

  function setZoom(nextScale: number, clientX?: number, clientY?: number) {
    const element = viewport.value
    if (!element) return
    const rect = element.getBoundingClientRect()
    const focalX = clientX === undefined ? 0 : clientX - rect.left - rect.width / 2
    const focalY = clientY === undefined ? 0 : clientY - rect.top - rect.height / 2
    transform.value = zoomVideoReviewTransform(
      transform.value,
      nextScale,
      focalX,
      focalY,
      rect.width,
      rect.height,
    )
  }

  function adjustZoom(delta: number) {
    setZoom(transform.value.scale + delta)
  }

  function onWheel(event: WheelEvent) {
    if (options.isZoomBlocked?.()) return
    setZoom(
      transform.value.scale * Math.exp(-event.deltaY * 0.0015),
      event.clientX,
      event.clientY,
    )
  }

  function startPan(event: PointerEvent) {
    if (
      transform.value.scale <= VIDEO_REVIEW_MIN_SCALE
      || options.isPanBlocked?.()
      || event.button !== 0
    ) return
    const target = event.target
    if (
      target instanceof Element
      && target.closest('button, input, select, .calibration-anchor, .frame-person-box, .frame-ignore-box')
    ) return
    panDrag.value = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      transform: { ...transform.value },
    }
    try {
      viewport.value?.setPointerCapture(event.pointerId)
    } catch {
      // Pointer capture is optional for synthetic accessibility input.
    }
    event.preventDefault()
  }

  function updatePan(event: PointerEvent) {
    const drag = panDrag.value
    if (!drag || !viewport.value || drag.pointerId !== event.pointerId) return
    const { width, height } = viewportSize()
    transform.value = panVideoReviewTransform(
      drag.transform,
      event.clientX - drag.clientX,
      event.clientY - drag.clientY,
      width,
      height,
    )
    event.preventDefault()
  }

  function finishPan(event: PointerEvent) {
    const drag = panDrag.value
    if (!drag || drag.pointerId !== event.pointerId) return
    panDrag.value = null
    try {
      if (viewport.value?.hasPointerCapture(event.pointerId)) {
        viewport.value.releasePointerCapture(event.pointerId)
      }
    } catch {
      // The pointer may already have been released by the browser.
    }
  }

  function onKeydown(event: KeyboardEvent) {
    if (event.target !== event.currentTarget) return
    if (event.key === '+' || event.key === '=') {
      event.preventDefault()
      adjustZoom(0.25)
      return
    }
    if (event.key === '-') {
      event.preventDefault()
      adjustZoom(-0.25)
      return
    }
    if (event.key === '0' || event.key === 'Home') {
      event.preventDefault()
      reset()
      return
    }
    const amount = event.shiftKey ? 64 : 24
    const movement = {
      ArrowLeft: [-amount, 0],
      ArrowRight: [amount, 0],
      ArrowUp: [0, -amount],
      ArrowDown: [0, amount],
    }[event.key]
    if (!movement || transform.value.scale <= VIDEO_REVIEW_MIN_SCALE) return
    event.preventDefault()
    const { width, height } = viewportSize()
    transform.value = panVideoReviewTransform(
      transform.value,
      movement[0],
      movement[1],
      width,
      height,
    )
  }

  watch(viewport, (element) => {
    resizeObserver?.disconnect()
    resizeObserver = null
    if (!element || typeof ResizeObserver === 'undefined') return
    resizeObserver = new ResizeObserver(() => commit(transform.value))
    resizeObserver.observe(element)
  })

  onScopeDispose(() => resizeObserver?.disconnect())

  return {
    viewport,
    transform,
    style,
    zoomPercent,
    panning,
    reset,
    setZoom,
    adjustZoom,
    onWheel,
    startPan,
    updatePan,
    finishPan,
    onKeydown,
  }
}
