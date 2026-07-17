import type {
  MatchDataProvider,
  MatchDataProviderCatalog,
  MatchDataProviderId,
} from '../types'

export const API_FOOTBALL_PROVIDER_ID: MatchDataProviderId = 'api-football'
export const THESPORTSDB_PROVIDER_ID: MatchDataProviderId = 'thesportsdb'

/**
 * Compatibility state for an API server which predates provider discovery.
 * Its catalog endpoints are TheSportsDB-only, so presenting API-Football as
 * active in that case would bind the wrong provider silently.
 */
export const LEGACY_MATCH_DATA_PROVIDERS: MatchDataProviderCatalog = {
  defaultProvider: THESPORTSDB_PROVIDER_ID,
  providers: [
    {
      id: API_FOOTBALL_PROVIDER_ID,
      name: 'API-Football',
      configured: false,
      available: false,
      reason: 'This API server does not expose the API-Football adapter yet.',
      capabilities: [],
    },
    {
      id: THESPORTSDB_PROVIDER_ID,
      name: 'TheSportsDB',
      configured: true,
      available: true,
      reason: null,
      capabilities: ['fixtures', 'lineups', 'events'],
    },
  ],
}

const PROVIDER_LABELS: Record<string, string> = {
  [API_FOOTBALL_PROVIDER_ID]: 'API-Football',
  [THESPORTSDB_PROVIDER_ID]: 'TheSportsDB',
  manual: 'Manual roster',
}

export function matchDataProviderLabel(providerId: string | null | undefined): string {
  if (!providerId) return 'Match data'
  return PROVIDER_LABELS[providerId] ?? providerId
}

export function matchDataProviderStatus(provider: MatchDataProvider): string {
  if (!provider.configured) return 'Not configured'
  if (!provider.available) return 'Unavailable'
  return 'Ready'
}

/** Resolve a project-owned provider first, then the server's safe default. */
export function resolveMatchDataProvider(
  catalog: MatchDataProviderCatalog,
  projectSource?: string | null,
): MatchDataProviderId {
  const ids = new Set(catalog.providers.map((provider) => provider.id))
  if (projectSource && projectSource !== 'manual' && ids.has(projectSource)) {
    return projectSource
  }
  const defaultProvider = catalog.providers.find(
    (provider) => provider.id === catalog.defaultProvider,
  )
  if (defaultProvider?.configured && defaultProvider.available) return defaultProvider.id
  return catalog.providers.find((provider) => provider.configured && provider.available)?.id
    ?? catalog.providers[0]?.id
    ?? API_FOOTBALL_PROVIDER_ID
}
