import { describe, expect, it } from 'vitest'
import {
  LEGACY_MATCH_DATA_PROVIDERS,
  matchDataProviderLabel,
  matchDataProviderStatus,
  resolveMatchDataProvider,
} from './matchDataProviders'

describe('match-data provider UI', () => {
  it('preserves the provider owned by the current video project', () => {
    expect(resolveMatchDataProvider({
      defaultProvider: 'api-football',
      providers: [
        { id: 'api-football', name: 'API-Football', configured: true, available: true },
        { id: 'thesportsdb', name: 'TheSportsDB', configured: true, available: true },
      ],
    }, 'thesportsdb')).toBe('thesportsdb')
  })

  it('uses the server fallback when API-Football is not configured', () => {
    expect(resolveMatchDataProvider({
      defaultProvider: 'api-football',
      providers: [
        { id: 'api-football', name: 'API-Football', configured: false, available: false },
        { id: 'thesportsdb', name: 'TheSportsDB', configured: true, available: true },
      ],
    })).toBe('thesportsdb')
    expect(resolveMatchDataProvider(LEGACY_MATCH_DATA_PROVIDERS)).toBe('thesportsdb')
  })

  it('renders provider state without credentials', () => {
    expect(matchDataProviderLabel('api-football')).toBe('API-Football')
    expect(matchDataProviderStatus({
      id: 'api-football',
      name: 'API-Football',
      configured: false,
      available: false,
    })).toBe('Not configured')
  })
})
