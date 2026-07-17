import { describe, expect, it } from 'vitest'
import { parseManualMatchImport } from './matchImport'

const valid = {
  event: { id: 'match-1', name: 'Home v Away' },
  teams: {
    home: { id: 'home', name: 'Home' },
    away: { id: 'away', name: 'Away' },
  },
  players: [{ id: 'player-1', name: 'Player One', team_id: 'home' }],
  provenance: { label: 'Official sheet' },
}

describe('manual match JSON import', () => {
  it('parses a valid match-scoped roster without rewriting it', () => {
    expect(parseManualMatchImport(JSON.stringify(valid))).toEqual(valid)
    expect(parseManualMatchImport(`\uFEFF${JSON.stringify(valid)}`)).toEqual(valid)
  })

  it.each([
    ['', 'empty'],
    ['{', 'not valid JSON'],
    ['[]', 'root must be an object'],
    [JSON.stringify({ ...valid, event: {} }), 'event.id and event.name'],
    [JSON.stringify({ ...valid, teams: {} }), 'named home and away teams'],
    [JSON.stringify({ ...valid, players: [] }), 'at least one player'],
  ])('reports an actionable envelope error for %#', (source, message) => {
    expect(() => parseManualMatchImport(source)).toThrow(message)
  })
})
