import type {
  EventBundle,
  ExternalPlayer,
  FrameAnalysis,
  PersistedMatchBinding,
  ReconstructionQuality,
  SceneDocument,
  Team,
} from '../types'
import { matchDataProviderLabel } from './matchDataProviders'

export function reconstructionLocksMutations(
  status: 'queued' | 'processing' | 'ready' | 'failed' | undefined,
  optimisticRunning = false,
): boolean {
  return optimisticRunning || status === 'queued' || status === 'processing'
}

/**
 * Frame-correction endpoints queue a reconstruction and return the new run
 * metadata with the refreshed frame payload. Merge that metadata immediately
 * so the editor cannot save an older whole-scene document over the queued run.
 */
export function mergeFrameReconstructionMetadata(
  scene: SceneDocument,
  analysis: Pick<FrameAnalysis, 'reconstruction'>,
): SceneDocument {
  const metadata = analysis.reconstruction
  const videoAsset = scene.payload.videoAsset
  if (!metadata || !videoAsset) return scene

  return {
    ...scene,
    payload: {
      ...scene.payload,
      videoAsset: {
        ...videoAsset,
        reconstruction: {
          ...videoAsset.reconstruction,
          status: metadata.status,
          processingStatus: metadata.status === 'ready' ? 'completed' : metadata.status,
          ...(metadata.runId === undefined ? {} : { runId: metadata.runId }),
          ...(metadata.runRevision === undefined ? {} : { runRevision: metadata.runRevision }),
          ...(metadata.inputFingerprint === undefined ? {} : { inputFingerprint: metadata.inputFingerprint }),
          ...(metadata.status === 'queued' || metadata.status === 'processing'
            ? {
                qualityVerdict: 'pending' as const,
                quality: undefined,
                qualityReport: undefined,
                progress: undefined,
                error: null,
              }
            : {}),
        },
      },
    },
  }
}

export type ProjectMatchBindingContext = {
  binding: PersistedMatchBinding | null
  projectSceneId: string | null
  projectScoped: boolean
  inherited: boolean
  label: string
}

/**
 * Describe the effective binding returned for the current scene. The backend
 * publishes the complete project snapshot on child scenes; the UI must not
 * mistake an inherited snapshot for an unbound shot.
 */
export function projectMatchBindingContext(
  scene: SceneDocument | null,
): ProjectMatchBindingContext {
  const binding = scene?.payload.matchBinding ?? null
  const projectSceneId = binding?.projectSceneId
    ?? scene?.payload.videoAsset?.multiPass?.parentSceneId
    ?? scene?.payload.videoAsset?.parentSceneId
    ?? scene?.id
    ?? null
  const projectScoped = binding?.scope === 'project'
    || binding?.inherited === true
    || Boolean(binding?.projectSceneId)
  const inherited = binding?.inherited === true
    || (projectScoped && Boolean(scene?.id && projectSceneId && scene.id !== projectSceneId))

  if (!binding) {
    return {
      binding,
      projectSceneId,
      projectScoped: true,
      inherited: false,
      label: 'Project has no match data',
    }
  }

  const source = `${matchDataProviderLabel(binding.source)} · #${binding.eventId}`
  return {
    binding,
    projectSceneId,
    projectScoped,
    inherited,
    label: inherited
      ? `Inherited from video project · ${source}`
      : projectScoped
        ? `Project match data · ${source}`
        : source,
  }
}

/** Prefer the effective match snapshot over placeholder teams on child scenes. */
export function resolvedProjectMatchTeams(
  scene: SceneDocument | null,
): { home: Team; away: Team } {
  const fallbackHome = scene?.payload.teams[0] ?? {
    id: 'home',
    name: 'Home',
    color: '#46d7c2',
    externalTeamId: null,
  }
  const fallbackAway = scene?.payload.teams[1] ?? {
    id: 'away',
    name: 'Away',
    color: '#f2c94c',
    externalTeamId: null,
  }
  const binding = scene?.payload.matchBinding
  const home = binding?.teams?.home ?? binding?.event?.home
  const away = binding?.teams?.away ?? binding?.event?.away
  return {
    home: {
      ...fallbackHome,
      ...(home ? {
        id: home.id || fallbackHome.id,
        name: home.name || fallbackHome.name,
        externalTeamId: home.id || fallbackHome.externalTeamId,
      } : {}),
    },
    away: {
      ...fallbackAway,
      ...(away ? {
        id: away.id || fallbackAway.id,
        name: away.name || fallbackAway.name,
        externalTeamId: away.id || fallbackAway.externalTeamId,
      } : {}),
    },
  }
}

/** The effective persisted v2 snapshot is the only roster authority for identity work. */
export function resolvedRosterPlayers(
  scene: SceneDocument | null,
): ExternalPlayer[] {
  return scene?.payload.matchBinding?.players ?? []
}

/** Rehydrate event UI from the same offline snapshot used by reconstruction. */
export function persistedEventBundle(scene: SceneDocument | null): EventBundle | null {
  const binding = scene?.payload.matchBinding
  if (binding?.schemaVersion !== 2 || !binding.event) return null
  const quality = binding.rosterQuality
  return {
    source: binding.source,
    event: binding.event,
    players: binding.players ?? [],
    lineup: binding.lineup ?? [],
    timeline: binding.timeline ?? [],
    substitutions: binding.substitutions ?? [],
    roster_quality: quality
      ? {
          status: quality.status,
          player_count: quality.playerCount,
          home_player_count: quality.homePlayerCount,
          away_player_count: quality.awayPlayerCount,
          automatic_identity_eligible: quality.automaticIdentityEligible,
          manual_identity_eligible: quality.manualIdentityEligible,
          reasons: [...quality.reasons],
        }
      : null,
    fetched_at: binding.fetchedAt ?? '',
    warnings: binding.warnings ?? [],
  }
}

export function matchBindingNeedsRefresh(scene: SceneDocument | null): boolean {
  const binding = scene?.payload.matchBinding
  if (!binding?.eventId || binding.source === 'manual') return false
  if (binding.schemaVersion !== 2 || !binding.event) return true
  const quality = binding.rosterQuality
  return !quality || quality.status !== 'automatic-ready' || !quality.automaticIdentityEligible
}

export function identityValidationSummary(
  validation: ReconstructionQuality['identityValidation'] | null | undefined,
): string {
  if (!validation) return 'Identity validation unavailable'
  if (validation?.status === 'invalid') {
    return `Identity validation invalid${validation.reason ? ` · ${validation.reason}` : ''}`
  }
  if (validation.status === 'unavailable' || !validation.groundTruthAvailable) {
    return `Identity accuracy · ground truth unavailable${validation.reason ? ` · ${validation.reason}` : ''}`
  }
  if (validation.idf1 === null || validation.idf1 === undefined) {
    return 'Identity validation invalid · evaluated result has no IDF1 value'
  }
  const switches = validation.idSwitchCount === null || validation.idSwitchCount === undefined
    ? 'ID switches unavailable'
    : `${validation.idSwitchCount} ID switches`
  return `Labelled IDF1 ${Math.round(validation.idf1 * 100)}% · ${switches}`
}
