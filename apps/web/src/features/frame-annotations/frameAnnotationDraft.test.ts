import { describe, expect, it } from 'vitest'
import {
  frameAnnotationWrite,
  newManualFrameAnnotationDraft,
  normalizeFrameAnnotationAction,
} from './frameAnnotationDraft'

describe('frame annotation drafts', () => {
  it('normalizes split semantics without mutating the editor draft', () => {
    const draft = { ...newManualFrameAnnotationDraft({ x: 2, y: 4 }), action: 'split' as const }
    const result = normalizeFrameAnnotationAction(draft, 3, 12)

    expect(result).not.toBe(draft)
    expect(result).toMatchObject({ scope: 'range', rangeStart: 3, rangeEnd: 12 })
    expect(draft.rangeStart).toBeNull()
  })

  it('publishes only fields meaningful to an exclusion command', () => {
    const draft = {
      ...newManualFrameAnnotationDraft({ x: 2, y: 4 }),
      action: 'exclude' as const,
      label: 'phantom',
      scope: 'observation' as const,
      mergeTargetId: 'person-2',
      rangeStart: 1,
      rangeEnd: 2,
    }
    expect(frameAnnotationWrite(draft, 5)).toMatchObject({
      sceneTime: 5,
      kind: 'ignore',
      label: null,
      mergeTargetId: null,
      rangeStart: null,
      rangeEnd: null,
    })
  })
})
