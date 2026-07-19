import { describe, expect, it } from 'vitest'
import type { SceneDocument } from '../../../types/scene'
import { preserveSceneActorSelection } from './preserveSceneActorSelection'

function scene(): SceneDocument {
  return {
    payload: {
      tracks: [
        { id: 'track-a', canonicalPersonId: 'person-a' },
        { id: 'track-unbound', canonicalPersonId: null },
      ],
      canonicalPeople: [{ canonicalPersonId: 'person-a' }],
    },
  } as SceneDocument
}

describe('preserveSceneActorSelection', () => {
  it('never creates a selection merely because the Scene has tracks', () => {
    expect(preserveSceneActorSelection(scene(), {
      trackId: null,
      canonicalPersonId: null,
    })).toEqual({ trackId: null, canonicalPersonId: null })
  })

  it('preserves an existing canonical or unbound-track selection only while it remains valid', () => {
    expect(preserveSceneActorSelection(scene(), {
      trackId: 'old-render-track',
      canonicalPersonId: 'person-a',
    })).toEqual({ trackId: 'track-a', canonicalPersonId: 'person-a' })
    expect(preserveSceneActorSelection(scene(), {
      trackId: 'track-unbound',
      canonicalPersonId: null,
    })).toEqual({ trackId: 'track-unbound', canonicalPersonId: null })
    expect(preserveSceneActorSelection(scene(), {
      trackId: 'missing-track',
      canonicalPersonId: 'missing-person',
    })).toEqual({ trackId: null, canonicalPersonId: null })
  })
})
