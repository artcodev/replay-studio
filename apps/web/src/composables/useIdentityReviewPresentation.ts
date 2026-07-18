import { computed, type ComputedRef, type Ref, type ShallowRef } from 'vue'
import {
  identityReviewItemObservations,
  identityReviewWorkerStates,
} from '../lib/identityReview'
import type { CanonicalPerson } from '../types/identity'
import type { SceneDocument } from '../types/scene'
import type { IdentityReviewResponse } from '../types/identityReview'

type IdentityReviewPresentationOptions = {
  scene: ShallowRef<SceneDocument | null>
  selectedPerson: ComputedRef<CanonicalPerson | null>
  snapshot: Ref<IdentityReviewResponse | null>
  hasDedicatedUnbind: (canonicalPersonId: string) => boolean
}

/** Builds the evidence-enriched identity view without owning review mutations. */
export function useIdentityReviewPresentation(options: IdentityReviewPresentationOptions) {
  const item = computed(() => {
    const identity = options.selectedPerson.value
    const review = options.snapshot.value
    if (!identity || !review || review.sceneId !== options.scene.value?.id) return null
    if (review.revision !== options.scene.value?.revision) return null
    return review.items.find(
      (candidate) => candidate.canonicalPersonId === identity.canonicalPersonId,
    ) ?? null
  })

  const person = computed<CanonicalPerson | null>(() => {
    const identity = options.selectedPerson.value
    if (!identity) return null
    const reviewItem = item.value
    const rejectedIds = new Set(
      (options.scene.value?.payload.identityReviewDecisions?.rosterRejections ?? [])
        .filter((decision) => decision.canonicalPersonId === identity.canonicalPersonId)
        .map((decision) => decision.externalPlayerId),
    )
    return {
      ...identity,
      ...(reviewItem
        ? {
            displayName: reviewItem.displayName,
            identityStatus: reviewItem.identityStatus,
            identityConfidence: reviewItem.identityConfidence ?? null,
            identitySource: reviewItem.identitySource ?? null,
            teamId: reviewItem.teamId ?? null,
            role: reviewItem.role ?? null,
            jerseyNumber: reviewItem.jerseyNumber ?? null,
            externalPlayerId: reviewItem.externalPlayerId ?? null,
            observationCount: reviewItem.observationCount,
            evidence: reviewItem.evidence,
            rosterCandidates: reviewItem.rosterCandidates,
            conflicts: reviewItem.conflicts,
          }
        : {}),
      rosterCandidates: (reviewItem?.rosterCandidates ?? identity.rosterCandidates).filter(
        (candidate) => !rejectedIds.has(candidate.externalPlayerId),
      ),
    }
  })

  const observations = computed(() => identityReviewItemObservations(item.value))
  const workers = computed(() => identityReviewWorkerStates(options.snapshot.value))
  const dedicatedUnbindActive = computed(() => (
    options.selectedPerson.value
      ? options.hasDedicatedUnbind(options.selectedPerson.value.canonicalPersonId)
      : false
  ))

  return { item, person, observations, workers, dedicatedUnbindActive }
}
