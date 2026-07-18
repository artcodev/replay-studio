import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import type { CanonicalMatch } from '../types/match'
import ProjectMatchTab from './ProjectMatchTab.vue'

function canonicalMatch(overrides: Partial<CanonicalMatch> = {}): CanonicalMatch {
  return {
    id: 'match-1',
    revision: 3,
    name: 'Spain vs Belgium',
    competition: 'World Cup',
    season: '2026',
    kickoffAt: '2026-07-10T18:00:00Z',
    status: 'finished',
    score: { home: 2, away: 1 },
    homeTeam: { id: 'team-home', name: 'Spain', shortName: 'ESP' },
    awayTeam: { id: 'team-away', name: 'Belgium', shortName: 'BEL' },
    roster: [
      { id: 'player-home-10', teamId: 'team-home', name: 'Home Forward', number: '10', position: 'Forward', role: 'starter' },
      { id: 'player-home-1', teamId: 'team-home', name: 'Home Keeper', number: '1', position: 'Goalkeeper', role: 'starter', goalkeeper: true },
      { id: 'player-away-8', teamId: 'team-away', name: 'Away Midfielder', number: '8', position: 'Midfielder', role: 'substitute' },
    ],
    events: [
      { id: 'event-2', kind: 'yellow-card', minute: 52, teamId: 'team-away', playerId: 'player-away-8', label: 'Booking' },
      { id: 'event-1', kind: 'goal', minute: 12, teamId: 'team-home', playerId: 'player-home-10', label: 'Opening goal' },
    ],
    substitutions: [{
      id: 'sub-1',
      teamId: 'team-away',
      minute: 61,
      playerInId: 'player-away-8',
      label: 'Away Midfielder enters',
    }],
    sync: {
      state: 'synced',
      syncedAt: '2026-07-17T10:00:00Z',
      stale: true,
      warnings: ['Lineup is missing one shirt number.'],
    },
    ...overrides,
    snapshotId: overrides.snapshotId ?? 'snapshot-1',
    snapshotHash: overrides.snapshotHash ?? 'sha256:one',
  }
}

describe('ProjectMatchTab', () => {
  it('renders canonical teams, roster, ordered events, substitutions and sync QA', async () => {
    const html = await renderToString(createSSRApp(ProjectMatchTab, {
      match: canonicalMatch(),
    }))

    expect(html).toContain('Canonical project data')
    expect(html).toContain('Spain vs Belgium')
    expect(html).toContain('ESP')
    expect(html).toContain('BEL')
    expect(html).toContain('Spain 2, Belgium 1')
    expect(html).toContain('3 players')
    expect(html.indexOf('#1 · Home Keeper')).toBeLessThan(html.indexOf('#10 · Home Forward'))
    expect(html.indexOf('Opening goal')).toBeLessThan(html.indexOf('Booking'))
    expect(html).toContain('Goal · Opening goal')
    expect(html).toContain('Away Midfielder enters')
    expect(html).toContain('Lineup is missing one shirt number.')
    expect(html).toContain('Stale')
    expect(html).toContain('>Refresh match</button>')
    expect(html).not.toContain('api-football')
    expect(html).not.toContain('thesportsdb')
  })

  it('makes manual ownership explicit and removes automatic refresh', async () => {
    const match = canonicalMatch({
      sync: { state: 'manual', syncedAt: null, stale: false, warnings: [] },
    })
    const html = await renderToString(createSSRApp(ProjectMatchTab, { match }))

    expect(html).toContain('Manual match data')
    expect(html).toContain('changes only when you import a replacement file')
    expect(html).toContain('>Replace JSON</button>')
    expect(html).not.toContain('>Refresh match</button>')
  })

  it('does not call a saved automatic snapshot never synced when its timestamp is unavailable', async () => {
    const match = canonicalMatch({
      sync: { state: 'synced', syncedAt: null, stale: false, warnings: [] },
    })
    const html = await renderToString(createSSRApp(ProjectMatchTab, { match }))

    expect(html).toContain('Sync time unavailable')
    expect(html).not.toContain('Never synced')
  })

  it('offers a provider-neutral manual path for a project without a match', async () => {
    const html = await renderToString(createSSRApp(ProjectMatchTab))

    expect(html).toContain('No match assigned')
    expect(html).toContain('Import match JSON')
    expect(html).toContain('Automatic synchronization can be configured later')
  })

  it('declares only refresh and import actions', () => {
    const events = (ProjectMatchTab as unknown as { emits: string[] }).emits
    expect(events).toEqual(expect.arrayContaining(['refresh', 'import']))
    expect(events).toHaveLength(2)
  })
})
