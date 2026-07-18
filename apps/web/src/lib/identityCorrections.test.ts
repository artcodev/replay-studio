import { describe, expect, it } from 'vitest'
import type { FrameAnnotation } from '../types/analysis'
import type { Track } from '../types/tracking'
import {
  annotationIdentityAction,
  buildIdentityMergeTargets,
  confirmedRosterBindingsConflict,
  dedicatedRosterBindingStateForOwner,
  dedicatedRosterMergeCompatible,
  identitySplitObservationCounts,
  identitySplitRangeIsValid,
  hasActiveDedicatedUnbindForOwner,
  semanticAnnotationForEdit,
  semanticAnnotationIdForEdit,
  wouldCreateIdentityMergeCycle,
} from './identityCorrections'

function annotation(id: string, overrides: Partial<FrameAnnotation> = {}): FrameAnnotation {
  return {
    id,
    sceneTime: 0,
    sourceTime: 0,
    frameIndex: 1,
    bbox: { x: 10, y: 10, width: 20, height: 40 },
    kind: 'home-player',
    label: id,
    externalPlayerId: null,
    updatedAt: '2026-01-01T00:00:00Z',
    ...overrides,
  }
}

function track(id: string): Track {
  return {
    id,
    label: id,
    teamId: 'home',
    color: '#fff',
    number: 1,
    externalPlayerId: null,
    keyframes: [],
  }
}

describe('identity corrections', () => {
  it('blocks merges between two different confirmed roster players', () => {
    expect(confirmedRosterBindingsConflict('player-8', 'player-10')).toBe(true)
    expect(confirmedRosterBindingsConflict('player-8', 'player-8')).toBe(false)
    expect(confirmedRosterBindingsConflict('player-8', null)).toBe(false)
    expect(confirmedRosterBindingsConflict(null, 'player-10')).toBe(false)
  })

  it('maps the ignore semantic to the exclude identity action', () => {
    expect(annotationIdentityAction(annotation('ignored', { kind: 'ignore' }))).toBe('exclude')
    expect(annotationIdentityAction(annotation('confirmed'))).toBe('confirm')
    expect(annotationIdentityAction(annotation('split', { action: 'split', scope: 'range' }))).toBe('split')
  })

  it('creates a separate semantic annotation instead of editing a dedicated roster correction', () => {
    const dedicated = annotation('roster-binding', {
      correctionKind: 'canonical-roster-binding-v1',
      canonicalPersonId: 'canonical-a',
      rosterBindingState: 'bound',
      externalPlayerId: 'player-8',
      label: 'Pedri',
    })
    const generic = annotation('semantic-label', {
      canonicalPersonId: 'canonical-a',
      label: 'Player A',
    })

    expect(semanticAnnotationIdForEdit({ annotationId: dedicated.id }, [dedicated])).toBeNull()
    expect(semanticAnnotationIdForEdit({
      annotationId: dedicated.id,
      annotationIds: [dedicated.id, generic.id],
    }, [dedicated, generic])).toBe(generic.id)
    expect(semanticAnnotationForEdit({
      // Aggregated person fields may now say "Pedri", but the editable
      // semantic source remains the older generic annotation.
      annotationId: dedicated.id,
      annotationIds: [dedicated.id, generic.id],
    }, [dedicated, generic])).toBe(generic)
    expect(semanticAnnotationForEdit({
      annotationId: dedicated.id,
      annotationIds: [dedicated.id, generic.id],
    }, [dedicated, generic])?.label).toBe('Player A')

    const primaryGeneric = annotation('primary-semantic', { label: 'Player A' })
    const olderGeneric = annotation('older-semantic', { label: 'Old label' })
    expect(semanticAnnotationForEdit({
      annotationId: primaryGeneric.id,
      annotationIds: [olderGeneric.id, dedicated.id, primaryGeneric.id],
    }, [olderGeneric, dedicated, primaryGeneric])).toBe(primaryGeneric)
  })

  it('rejects merge targets with incompatible dedicated Bind and Unbind decisions', () => {
    const annotations = [
      annotation('bound-a', {
        correctionKind: 'canonical-roster-binding-v1',
        canonicalPersonId: 'canonical-a',
        rosterBindingState: 'bound',
        externalPlayerId: 'player-8',
      }),
      annotation('unbound-b', {
        correctionKind: 'canonical-roster-binding-v1',
        canonicalPersonId: 'canonical-b',
        rosterBindingState: 'unbound',
        externalPlayerId: null,
      }),
      annotation('bound-c', {
        correctionKind: 'canonical-roster-binding-v1',
        canonicalPersonId: 'canonical-c',
        rosterBindingState: 'bound',
        externalPlayerId: 'player-8',
      }),
    ]

    expect(dedicatedRosterMergeCompatible(
      annotations,
      ['canonical-a'],
      ['canonical-b'],
    )).toBe(false)
    expect(dedicatedRosterMergeCompatible(
      annotations,
      ['canonical-a'],
      ['canonical-c'],
    )).toBe(true)
    expect(dedicatedRosterMergeCompatible(
      annotations,
      ['canonical-a'],
      ['canonical-without-decision'],
    )).toBe(true)
    expect(dedicatedRosterBindingStateForOwner(annotations, ['canonical-a'])).toBe('bound')
    expect(dedicatedRosterBindingStateForOwner(annotations, ['canonical-b'])).toBe('unbound')
    expect(dedicatedRosterBindingStateForOwner(annotations, ['canonical-without-decision'])).toBeNull()
    expect(hasActiveDedicatedUnbindForOwner(annotations, ['canonical-a'])).toBe(false)
    expect(hasActiveDedicatedUnbindForOwner(annotations, ['canonical-b'])).toBe(true)
    const malformedWithoutState = [
      annotation('missing-state', {
        correctionKind: 'canonical-roster-binding-v1',
        canonicalPersonId: 'canonical-b',
        externalPlayerId: null,
      }),
    ]
    expect(hasActiveDedicatedUnbindForOwner(
      malformedWithoutState,
      ['canonical-b'],
    )).toBe(false)
    expect(dedicatedRosterBindingStateForOwner(
      malformedWithoutState,
      ['canonical-b'],
    )).toBe('conflict')
    expect(hasActiveDedicatedUnbindForOwner(
      [...annotations, annotation('second-unbound-b', {
        correctionKind: 'canonical-roster-binding-v1',
        canonicalPersonId: 'canonical-b',
        rosterBindingState: 'unbound',
      })],
      ['canonical-b'],
    )).toBe(false)
  })

  it('resolves a durable roster correction by its moved observation anchor after split', () => {
    const annotations = [
      annotation('bound-moved-to-child', {
        correctionKind: 'canonical-roster-binding-v1',
        canonicalPersonId: 'canonical-root',
        targetObservationId: 'observation-child-anchor',
        rosterBindingState: 'bound',
        externalPlayerId: 'player-8',
      }),
      annotation('unbound-stays-root', {
        correctionKind: 'canonical-roster-binding-v1',
        canonicalPersonId: 'canonical-root',
        targetObservationId: 'observation-root-anchor',
        rosterBindingState: 'unbound',
        externalPlayerId: null,
      }),
    ]
    const observation = (observationId: string, frameIndex: number) => ({
      observationId,
      frameIndex,
      sceneTime: frameIndex / 10,
      bbox: { x: 0, y: 0, width: 20, height: 40 },
      confidence: 0.9,
    })
    const ownership = {
      canonicalPeople: [
        {
          canonicalPersonId: 'canonical-root',
          observations: [observation('observation-root-anchor', 1)],
        },
        {
          canonicalPersonId: 'canonical-child',
          observations: [observation('observation-child-anchor', 2)],
        },
      ],
      tracks: [],
    }

    expect(dedicatedRosterMergeCompatible(
      annotations,
      ['canonical-child'],
      ['canonical-root'],
      ownership,
    )).toBe(false)
    // The stale persisted canonicalPersonId on the bound correction must not
    // make the root look internally conflicted after its anchor moved.
    expect(dedicatedRosterMergeCompatible(
      annotations,
      ['canonical-root'],
      ['canonical-without-decision'],
      ownership,
    )).toBe(true)
    expect(dedicatedRosterBindingStateForOwner(
      annotations,
      ['canonical-child'],
      ownership,
    )).toBe('bound')
    expect(dedicatedRosterBindingStateForOwner(
      annotations,
      ['canonical-root'],
      ownership,
    )).toBe('unbound')
    expect(hasActiveDedicatedUnbindForOwner(
      annotations,
      ['canonical-root'],
      ownership,
    )).toBe(true)

    const linkedCorrection = annotation('bound-linked-to-child-track', {
      correctionKind: 'canonical-roster-binding-v1',
      canonicalPersonId: 'canonical-root',
      targetObservationId: 'remapped-anchor-not-in-published-observations',
      rosterBindingState: 'bound',
      externalPlayerId: 'player-10',
    })
    expect(dedicatedRosterMergeCompatible(
      [linkedCorrection, annotations[1]],
      ['canonical-child'],
      ['canonical-root'],
      {
        canonicalPeople: ownership.canonicalPeople.map((person) => ({
          ...person,
          annotationIds: person.canonicalPersonId === 'canonical-child'
            ? [linkedCorrection.id]
            : [],
        })),
        tracks: [],
      },
    )).toBe(false)
    expect(dedicatedRosterMergeCompatible(
      [linkedCorrection, annotations[1]],
      ['canonical-child'],
      ['canonical-root'],
      {
        ...ownership,
        tracks: [{
          id: 'track-child',
          canonicalPersonId: 'canonical-child',
          annotationIds: [linkedCorrection.id],
        }],
      },
    )).toBe(false)

    const geometricCorrection = annotation('unbound-geometric-child', {
      correctionKind: 'canonical-roster-binding-v1',
      canonicalPersonId: 'canonical-root',
      targetObservationId: 'missing-observation-id',
      targetObservation: observation('original-anchor-id', 2),
      rosterBindingState: 'unbound',
      externalPlayerId: null,
    })
    expect(dedicatedRosterBindingStateForOwner(
      [geometricCorrection],
      ['canonical-child'],
      ownership,
    )).toBe('unbound')
  })

  it('validates an exclusive-end split range around its immutable target', () => {
    const base = {
      duration: 4,
      canonicalPersonId: 'canonical-a',
      targetObservationId: 'observation-a',
      rangeStart: 1,
      rangeEnd: 3,
      targetTime: 2,
    }
    expect(identitySplitRangeIsValid(base)).toBe(true)
    expect(identitySplitRangeIsValid({ ...base, targetTime: 3 })).toBe(false)
    expect(identitySplitRangeIsValid({ ...base, rangeEnd: 1 })).toBe(false)
    expect(identitySplitRangeIsValid({ ...base, targetObservationId: null })).toBe(false)
  })

  it('previews affected and remaining observations using [start, end)', () => {
    const observations = [0, 1, 2, 3].map((sceneTime, frameIndex) => ({
      frameIndex,
      sceneTime,
      bbox: { x: 0, y: 0, width: 10, height: 20 },
      confidence: 1,
    }))
    expect(identitySplitObservationCounts(observations, 1, 3)).toEqual({
      affected: 2,
      remaining: 2,
    })
    expect(identitySplitObservationCounts([], 1, 3, {
      affectedObservationCount: 5,
      remainingObservationCount: 7,
    })).toEqual({ affected: 5, remaining: 7 })
  })

  it('detects a proposed annotation merge cycle', () => {
    const annotations = [
      annotation('person-a'),
      annotation('person-b', { action: 'merge', mergeTargetId: 'person-a' }),
    ]

    expect(wouldCreateIdentityMergeCycle(annotations, 'person-a', 'person-b')).toBe(true)
    expect(wouldCreateIdentityMergeCycle(annotations, 'person-a', 'track-1')).toBe(false)
  })

  it('offers valid tracks and people but excludes self, phantoms, and cyclic targets', () => {
    const annotations = [
      annotation('person-a'),
      annotation('person-b', { action: 'merge', mergeTargetId: 'person-a' }),
      annotation('phantom', { kind: 'ignore', action: 'exclude' }),
      annotation('split-boundary', { action: 'split', scope: 'range' }),
      annotation('dedicated-roster', { correctionKind: 'canonical-roster-binding-v1' }),
      annotation('person-c'),
    ]

    expect(buildIdentityMergeTargets(
      [track('track-current'), track('track-target')],
      annotations,
      'person-a',
      'track-current',
    )).toEqual([
      { id: 'track-target', label: 'track-target', type: 'track' },
      { id: 'person-c', label: 'person-c', type: 'annotation' },
    ])
  })

  it('omits tracks excluded by another whole-identity correction', () => {
    const annotations = [
      annotation('person-current'),
      annotation('excluded-identity', {
        action: 'exclude',
        scope: 'identity',
        sourceTrackId: 'track-excluded',
      }),
      annotation('excluded-observation', {
        action: 'exclude',
        scope: 'observation',
        sourceTrackId: 'track-observation-only',
      }),
    ]

    const targets = buildIdentityMergeTargets(
      [
        track('track-current'),
        track('track-excluded'),
        track('track-observation-only'),
        track('track-target'),
      ],
      annotations,
      'person-current',
      'track-current',
    )

    expect(targets.filter((target) => target.type === 'track')).toEqual([
      { id: 'track-observation-only', label: 'track-observation-only', type: 'track' },
      { id: 'track-target', label: 'track-target', type: 'track' },
    ])
  })
})
