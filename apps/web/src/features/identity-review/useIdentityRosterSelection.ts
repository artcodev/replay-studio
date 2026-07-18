import { computed, ref, watch, type Ref } from 'vue'
import { manualRosterBindingDecision } from '../../lib/identityReview'
import type { CanonicalPerson } from '../../types/identity'
import type { ExternalPlayer } from '../../types/match'
import {
  identityRosterPlayerLabel,
  orderedIdentityRosterPlayers,
} from './identityReviewPresentation'

type IdentityRosterSelectionOptions = {
  identity: Readonly<Ref<CanonicalPerson | null>>
  players: Readonly<Ref<ExternalPlayer[]>>
}

/** Owns manual roster picker state and fail-closed duplicate-ID validation. */
export function useIdentityRosterSelection(options: IdentityRosterSelectionOptions) {
  const selection = ref(options.identity.value?.externalPlayerId ?? '')
  const idCounts = computed(() => {
    const counts = new Map<string, number>()
    for (const player of options.players.value) {
      counts.set(player.id, (counts.get(player.id) ?? 0) + 1)
    }
    return counts
  })
  const orderedPlayers = computed(() => orderedIdentityRosterPlayers(options.players.value))
  const selectionCount = computed(() => idCounts.value.get(selection.value) ?? 0)
  const decision = computed(() => {
    const identity = options.identity.value
    return identity
      ? manualRosterBindingDecision(
          identity.canonicalPersonId,
          identity.externalPlayerId,
          selection.value,
          options.players.value,
        )
      : null
  })
  const currentBindingMissing = computed(() => Boolean(
    options.identity.value?.externalPlayerId
    && !idCounts.value.has(options.identity.value.externalPlayerId),
  ))
  const currentBindingLabel = computed(() => {
    const externalPlayerId = options.identity.value?.externalPlayerId
    if (!externalPlayerId) return ''
    return identityRosterPlayerLabel(
      options.players.value.find((player) => player.id === externalPlayerId)
      ?? { id: externalPlayerId, name: externalPlayerId },
    )
  })

  watch(
    () => options.identity.value?.canonicalPersonId,
    () => { selection.value = options.identity.value?.externalPlayerId ?? '' },
  )
  watch(
    () => options.identity.value?.externalPlayerId,
    (externalPlayerId) => { selection.value = externalPlayerId ?? '' },
  )

  return {
    selection,
    idCounts,
    orderedPlayers,
    selectionCount,
    decision,
    canBind: computed(() => Boolean(decision.value)),
    currentBindingMissing,
    currentBindingLabel,
  }
}
