import { describe, expect, it } from 'vitest'
import type { SceneDocument } from '../types'
import {
  identityValidationSummary,
  matchBindingNeedsRefresh,
  mergeFrameReconstructionMetadata,
  persistedEventBundle,
  projectMatchBindingContext,
  reconstructionLocksMutations,
  resolvedProjectMatchTeams,
  resolvedRosterPlayers,
} from './reconstructionUi'

function scene(): SceneDocument {
  return {
    id: 'scene-1',
    title: 'Moment',
    version: 4,
    duration: 3,
    payload: {
      pitch: { length: 105, width: 68 },
      matchBinding: {
        schemaVersion: 2,
        source: 'thesportsdb',
        eventId: 'event-1',
        fetchedAt: '2026-07-17T00:00:00Z',
        event: {
          id: 'event-1',
          name: 'Home v Away',
          home: { id: 'home', name: 'Home' },
          away: { id: 'away', name: 'Away' },
        },
        players: [{ id: 'saved-8', name: 'Saved Eight', number: '8' }],
        lineup: [],
        timeline: [{ id: 'goal-1', type: 'goal', label: 'Goal' }],
        substitutions: [],
        rosterQuality: {
          status: 'automatic-ready',
          playerCount: 22,
          homePlayerCount: 11,
          awayPlayerCount: 11,
          automaticIdentityEligible: true,
          manualIdentityEligible: true,
          reasons: [],
        },
      },
      videoAsset: {
        id: 'video-1',
        filename: 'clip.mp4',
        mediaUrl: '/clip.mp4',
        posterUrl: '/clip.jpg',
        fps: 25,
        frameCount: 75,
        processingState: 'ready',
        reconstruction: {
          status: 'ready',
          processingStatus: 'completed',
          qualityVerdict: 'pass',
          quality: { verdict: 'pass' },
          progress: {
            phase: 'finalizing',
            phaseIndex: 6,
            phaseCount: 6,
            label: 'Complete',
            detail: 'Previous run complete',
            completed: 1,
            total: 1,
            phasePercent: 100,
            overallPercent: 100,
            elapsedSeconds: 12,
            etaSeconds: 0,
            updatedAt: '2026-07-17T00:00:00Z',
            phases: [],
          },
          runId: 'run-old',
          runRevision: 4,
          model: 'yolo26m.pt',
        },
      },
      teams: [],
      tracks: [],
      canonicalPeople: [],
      ball: { mode: 'automatic', keyframes: [] },
      eventBindings: [],
      cameraCuts: [],
    },
  }
}

describe('reconstruction editor state', () => {
  it('locks mutations throughout optimistic, queued, and processing reconstruction', () => {
    expect(reconstructionLocksMutations('ready')).toBe(false)
    expect(reconstructionLocksMutations('failed')).toBe(false)
    expect(reconstructionLocksMutations('queued')).toBe(true)
    expect(reconstructionLocksMutations('processing')).toBe(true)
    expect(reconstructionLocksMutations('ready', true)).toBe(true)
  })

  it('merges queued correction metadata without dropping the current reconstruction', () => {
    const original = scene()
    const updated = mergeFrameReconstructionMetadata(original, {
      reconstruction: {
        status: 'queued',
        runId: 'run-new',
        runRevision: 5,
        inputFingerprint: 'fingerprint-new',
      },
    })

    expect(updated).not.toBe(original)
    expect(updated.payload.videoAsset?.reconstruction).toMatchObject({
      status: 'queued',
      processingStatus: 'queued',
      qualityVerdict: 'pending',
      runId: 'run-new',
      runRevision: 5,
      inputFingerprint: 'fingerprint-new',
      model: 'yolo26m.pt',
    })
    expect(updated.payload.videoAsset?.reconstruction?.quality).toBeUndefined()
    expect(updated.payload.videoAsset?.reconstruction?.progress).toBeUndefined()
    expect(original.payload.videoAsset?.reconstruction?.status).toBe('ready')
  })

  it('uses only the roster persisted with the scene', () => {
    const savedScene = scene()
    expect(resolvedRosterPlayers(savedScene).map((player) => player.id)).toEqual(['saved-8'])
  })

  it('treats a child scene project snapshot as inherited match data', () => {
    const shot = scene()
    shot.id = 'shot-1'
    shot.payload.videoAsset!.parentSceneId = 'project-1'
    shot.payload.teams = [
      { id: 'placeholder-home', name: 'Home', color: '#112233', externalTeamId: null },
      { id: 'placeholder-away', name: 'Away', color: '#445566', externalTeamId: null },
    ]
    Object.assign(shot.payload.matchBinding!, {
      scope: 'project',
      projectSceneId: 'project-1',
      inherited: true,
      teams: {
        home: { id: 'spain', name: 'Spain' },
        away: { id: 'belgium', name: 'Belgium' },
      },
    })

    expect(projectMatchBindingContext(shot)).toMatchObject({
      projectSceneId: 'project-1',
      projectScoped: true,
      inherited: true,
      label: 'Inherited from video project · TheSportsDB · #event-1',
    })
    expect(resolvedProjectMatchTeams(shot)).toEqual({
      home: {
        id: 'spain',
        name: 'Spain',
        color: '#112233',
        externalTeamId: 'spain',
      },
      away: {
        id: 'belgium',
        name: 'Belgium',
        color: '#445566',
        externalTeamId: 'belgium',
      },
    })
    expect(resolvedRosterPlayers(shot).map((player) => player.id)).toEqual(['saved-8'])
    expect(persistedEventBundle(shot)?.event.name).toBe('Home v Away')
  })

  it('describes an unbound child as a project setting instead of scene-local data', () => {
    const shot = scene()
    shot.id = 'shot-1'
    shot.payload.videoAsset!.parentSceneId = 'project-1'
    shot.payload.matchBinding = null

    expect(projectMatchBindingContext(shot)).toMatchObject({
      binding: null,
      projectSceneId: 'project-1',
      projectScoped: true,
      inherited: false,
      label: 'Project has no match data',
    })
  })

  it('constructs event UI from a v2 snapshot and never guesses from legacy metadata', () => {
    const savedScene = scene()
    expect(persistedEventBundle(savedScene)).toMatchObject({
      source: 'thesportsdb',
      event: { id: 'event-1', name: 'Home v Away' },
      players: [{ id: 'saved-8' }],
      timeline: [{ id: 'goal-1', label: 'Goal' }],
      roster_quality: { automatic_identity_eligible: true },
    })

    savedScene.payload.matchBinding = {
      source: 'thesportsdb',
      eventId: 'event-1',
      fetchedAt: null,
    }
    expect(persistedEventBundle(savedScene)).toBeNull()
    expect(matchBindingNeedsRefresh(savedScene)).toBe(true)
  })

  it('preserves an API-Football snapshot source for refresh and attribution', () => {
    const savedScene = scene()
    Object.assign(savedScene.payload.matchBinding!, {
      source: 'api-football',
      scope: 'project',
      projectSceneId: savedScene.id,
    })

    expect(persistedEventBundle(savedScene)?.source).toBe('api-football')
    expect(projectMatchBindingContext(savedScene).label).toBe(
      'Project match data · API-Football · #event-1',
    )
  })

  it('offers provider refresh for partial snapshots but not manual imports', () => {
    const partial = scene()
    partial.payload.matchBinding!.rosterQuality!.status = 'partial'
    partial.payload.matchBinding!.rosterQuality!.automaticIdentityEligible = false
    expect(matchBindingNeedsRefresh(partial)).toBe(true)

    partial.payload.matchBinding!.source = 'manual'
    expect(matchBindingNeedsRefresh(partial)).toBe(false)
    expect(matchBindingNeedsRefresh(null)).toBe(false)
  })

  it('does not present unavailable or invalid validation as zero percent', () => {
    expect(identityValidationSummary({
      groundTruthAvailable: false,
      status: 'unavailable',
      idf1: null,
      reason: 'no labelled rows',
    })).toBe('Identity accuracy · ground truth unavailable · no labelled rows')
    expect(identityValidationSummary({
      groundTruthAvailable: true,
      status: 'invalid',
      idf1: null,
      reason: 'conflicting duplicate labels',
    })).toBe('Identity validation invalid · conflicting duplicate labels')
    expect(identityValidationSummary({
      groundTruthAvailable: true,
      status: 'evaluated',
      idf1: 0.873,
      idSwitchCount: 2,
    })).toBe('Labelled IDF1 87% · 2 ID switches')
    expect(identityValidationSummary({
      groundTruthAvailable: true,
      status: 'evaluated',
      idf1: 0.5,
    })).toBe('Labelled IDF1 50% · ID switches unavailable')
    expect(identityValidationSummary(undefined)).toBe('Identity validation unavailable')
  })
})
