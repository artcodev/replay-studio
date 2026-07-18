import { describe, expect, it } from 'vitest'
import { bestFramePersonForCanonicalIdentity, validFrameMatchedTrackId } from './frameAnalysisSelection'

describe('frame analysis selection', () => {
  it('rejects stale matched track IDs', () => {
    expect(validFrameMatchedTrackId({ matchedTrackId: 'gone' } as never, [{ id: 'active' }] as never)).toBeNull()
  })

  it('prefers persisted observations before detector distance and confidence', () => {
    const source = {
      people: [
        { id: 'manual', canonicalPersonId: 'p1', matchSource: 'manual-identity', matchDistance: 0.01, confidence: 0.99 },
        { id: 'persisted', canonicalPersonId: 'p1', matchSource: 'persisted-observation', matchDistance: 0.5, confidence: 0.5 },
      ],
    }
    expect(bestFramePersonForCanonicalIdentity(source as never, 'p1')?.id).toBe('persisted')
  })
})
