import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import type { MatchCandidate } from '../types/match'
import ProjectMatchSearch from './ProjectMatchSearch.vue'

const candidate: MatchCandidate = {
  id: 'candidate-1',
  name: 'Spain vs Belgium',
  date: '2026-07-17',
  time: '20:00:00',
  competition: 'World Cup',
  homeTeam: { name: 'Spain' },
  awayTeam: { name: 'Belgium' },
  score: { home: 2, away: 1 },
}

describe('ProjectMatchSearch', () => {
  it('renders provider-neutral candidates and both discovery modes', async () => {
    const html = await renderToString(createSSRApp(ProjectMatchSearch, {
      query: 'Spain vs Belgium',
      date: '2026-07-17',
      candidates: [candidate],
    }))

    expect(html).toContain('Find or change match')
    expect(html).toContain('Search match by teams')
    expect(html).toContain('Match date')
    expect(html).toContain('Spain')
    expect(html).toContain('Belgium')
    expect(html).toContain('World Cup')
    expect(html.toLowerCase()).not.toContain('api-football')
    expect(html.toLowerCase()).not.toContain('thesportsdb')
  })

  it('shows errors without rendering stale candidates', async () => {
    const html = await renderToString(createSSRApp(ProjectMatchSearch, {
      error: 'Catalog unavailable',
      candidates: [candidate],
    }))

    expect(html).toContain('Catalog unavailable')
    expect(html).not.toContain('World Cup')
  })
})
