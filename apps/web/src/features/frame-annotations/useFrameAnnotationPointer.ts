import { ref, type Ref } from 'vue'
import { selectFrameDetectionHit } from '../../lib/frameDetectionHitTest'
import { clientPointToContainedMedia } from '../../lib/videoReviewTransform'
import type { FrameAnalysis } from '../../types/analysis'
import {
  newManualFrameAnnotationDraft,
  type FrameAnnotationDraft,
} from './frameAnnotationDraft'

type FrameAnnotationPointerOptions = {
  analysis: Readonly<Ref<FrameAnalysis | null>>
  mode: Readonly<Ref<boolean>>
  draft: Ref<FrameAnnotationDraft | null>
  saveState: Ref<string>
  selectPerson: (person: FrameAnalysis['people'][number]) => void
}

/** Owns only pointer-to-source coordinate mapping, hit cycling and box drawing. */
export function useFrameAnnotationPointer(options: FrameAnnotationPointerOptions) {
  const overlay = ref<SVGSVGElement | null>(null)
  const drag = ref<{ x: number; y: number; pointerId: number } | null>(null)
  let hitCycle: { frameIndex: number; x: number; y: number; personId: string } | null = null
  let suppressNextOverlayClick = false

  function imagePoint(event: MouseEvent | PointerEvent) {
    const source = options.analysis.value
    if (!overlay.value || !source) return null
    return clientPointToContainedMedia(
      event.clientX,
      event.clientY,
      overlay.value.getBoundingClientRect(),
      source.frameWidth,
      source.frameHeight,
    )
  }

  function selectAtPoint(event: MouseEvent) {
    if (suppressNextOverlayClick || drag.value) {
      suppressNextOverlayClick = false
      return
    }
    const source = options.analysis.value
    const point = imagePoint(event)
    if (!source || !overlay.value || !point) return
    const rect = overlay.value.getBoundingClientRect()
    const renderedScale = Math.min(
      rect.width / Math.max(1, source.frameWidth),
      rect.height / Math.max(1, source.frameHeight),
    )
    const minimumTargetSize = 24 / Math.max(0.001, renderedScale)
    const sameHitCluster = hitCycle?.frameIndex === source.frameIndex
      && (hitCycle.x - point.x) ** 2 + (hitCycle.y - point.y) ** 2 <= (minimumTargetSize / 2) ** 2
    const person = selectFrameDetectionHit(source.people, point, {
      minimumTargetSize,
      previousCandidateId: sameHitCluster ? hitCycle?.personId : null,
    })
    if (!person) {
      hitCycle = null
      return
    }
    hitCycle = { frameIndex: source.frameIndex, x: point.x, y: point.y, personId: person.id }
    options.selectPerson(person)
  }

  function startDrag(event: PointerEvent) {
    if (!options.mode.value || event.button !== 0 || event.target !== event.currentTarget) return
    const point = imagePoint(event)
    if (!point) return
    drag.value = { ...point, pointerId: event.pointerId }
    options.draft.value = newManualFrameAnnotationDraft(point)
    try {
      overlay.value?.setPointerCapture(event.pointerId)
    } catch {
      // Pointer capture is optional for synthetic input.
    }
  }

  function updateDrag(event: PointerEvent) {
    const start = drag.value
    const currentDraft = options.draft.value
    const point = imagePoint(event)
    if (!start || !currentDraft || !point || start.pointerId !== event.pointerId) return
    currentDraft.bbox = {
      x: Math.min(start.x, point.x),
      y: Math.min(start.y, point.y),
      width: Math.max(4, Math.abs(point.x - start.x)),
      height: Math.max(4, Math.abs(point.y - start.y)),
    }
  }

  function finishDrag(event: PointerEvent) {
    const start = drag.value
    const point = imagePoint(event)
    if (!start || !point || start.pointerId !== event.pointerId) return
    suppressNextOverlayClick = event.type === 'pointerup'
    if (suppressNextOverlayClick) {
      window.setTimeout(() => { suppressNextOverlayClick = false }, 0)
    }
    updateDrag(event)
    drag.value = null
    try {
      if (overlay.value?.hasPointerCapture(event.pointerId)) overlay.value.releasePointerCapture(event.pointerId)
    } catch {
      // The browser may release capture before pointerup reaches the overlay.
    }
    if (Math.abs(point.x - start.x) < 6 || Math.abs(point.y - start.y) < 10) {
      options.draft.value = null
      options.saveState.value = 'Draw a larger box around the full person'
    } else options.saveState.value = 'New manual person box ready'
  }

  function clear() {
    drag.value = null
    hitCycle = null
    suppressNextOverlayClick = false
  }

  return { overlay, drag, selectAtPoint, startDrag, updateDrag, finishDrag, clear }
}
