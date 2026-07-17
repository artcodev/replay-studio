<script setup lang="ts">
import { computed, ref, useId, watch } from 'vue'
import {
  cannotLinkReviewDecision,
  canonicalReviewObservations,
  identityReviewWorkerStatusLabel,
  inspectIdentityObservationDecision,
  linkReviewDecision,
  manualRosterBindingDecision,
  rosterReviewDecision,
  topIdentityReviewObservations,
  type IdentityReviewCandidateDecision,
  type IdentityReviewCannotLinkDecision,
  type IdentityReviewInspectFrame,
  type IdentityReviewLinkCandidate,
  type IdentityReviewObservation,
  type IdentityReviewWorkerState,
} from '../lib/identityReview'
import {
  canonicalPersonDisplayName,
  canonicalPersonSourceTrackletCount,
  canonicalPersonStatusLabel,
  identityConfidenceLabel,
  identityEvidenceKindLabel,
  topRosterCandidates,
} from '../lib/identityResolution'
import type { CanonicalIdentityEvidence, CanonicalPerson, ExternalPlayer } from '../types'

const props = withDefaults(defineProps<{
  identity?: CanonicalPerson | null
  rosterPlayers?: ExternalPlayer[]
  observations?: IdentityReviewObservation[] | null
  workerStates?: IdentityReviewWorkerState[] | null
  linkCandidates?: IdentityReviewLinkCandidate[] | null
  dedicatedUnbindActive?: boolean
  disabled?: boolean
  maxObservations?: number
}>(), {
  identity: null,
  rosterPlayers: () => [],
  observations: null,
  workerStates: null,
  linkCandidates: null,
  dedicatedUnbindActive: false,
  disabled: false,
  maxObservations: 8,
})

const emit = defineEmits<{
  'bind-candidate': [payload: { canonicalPersonId: string; externalPlayerId: string }]
  'reject-candidate': [payload: IdentityReviewCandidateDecision]
  'cannot-link': [payload: IdentityReviewCannotLinkDecision]
  'inspect-frame': [payload: IdentityReviewInspectFrame]
  'unbind-roster': [payload: { canonicalPersonId: string }]
  'clear-roster-binding': [payload: { canonicalPersonId: string }]
}>()

const uid = useId()
const titleId = `${uid}-identity-review-title`
const evidenceId = `${uid}-identity-review-evidence`
const observationsId = `${uid}-identity-review-observations`
const workersId = `${uid}-identity-review-workers`
const candidatesId = `${uid}-identity-review-candidates`
const linksId = `${uid}-identity-review-links`
const conflictsId = `${uid}-identity-review-conflicts`

const failedPreviewIds = ref(new Set<string>())
const displayName = computed(() => props.identity
  ? canonicalPersonDisplayName(props.identity, props.rosterPlayers)
  : 'No person selected')
const observationsWereSupplied = computed(() => props.observations !== null)
const reviewObservations = computed(() => {
  if (!props.identity) return []
  const source = props.observations ?? canonicalReviewObservations(props.identity)
  return topIdentityReviewObservations(source, props.maxObservations)
})
const candidates = computed(() => props.identity
  ? topRosterCandidates(props.identity, props.rosterPlayers, Math.max(1, props.identity.rosterCandidates.length))
  : [])
const sourceTrackletCount = computed(() => props.identity
  ? canonicalPersonSourceTrackletCount(props.identity)
  : 0)
const rosterSelection = ref(props.identity?.externalPlayerId ?? '')
const rosterIdCounts = computed(() => {
  const counts = new Map<string, number>()
  for (const player of props.rosterPlayers) {
    counts.set(player.id, (counts.get(player.id) ?? 0) + 1)
  }
  return counts
})
const sortedRosterPlayers = computed(() => [...props.rosterPlayers].sort((left, right) => (
  String(left.team_name || left.team_id || '').localeCompare(String(right.team_name || right.team_id || ''))
  || rosterNumberSortValue(left.number) - rosterNumberSortValue(right.number)
  || left.name.localeCompare(right.name)
  || left.id.localeCompare(right.id)
)))
const rosterSelectionCount = computed(() => rosterIdCounts.value.get(rosterSelection.value) ?? 0)
const manualRosterBinding = computed(() => props.identity
  ? manualRosterBindingDecision(
      props.identity.canonicalPersonId,
      props.identity.externalPlayerId,
      rosterSelection.value,
      props.rosterPlayers,
    )
  : null)
const canBindRosterSelection = computed(() => Boolean(manualRosterBinding.value))
const currentBindingMissingFromRoster = computed(() => Boolean(
  props.identity?.externalPlayerId
  && !rosterIdCounts.value.has(props.identity.externalPlayerId),
))
const currentRosterBindingLabel = computed(() => {
  const externalPlayerId = props.identity?.externalPlayerId
  if (!externalPlayerId) return ''
  return rosterPlayerLabel(
    props.rosterPlayers.find((player) => player.id === externalPlayerId)
    ?? { id: externalPlayerId, name: externalPlayerId },
  )
})

watch(
  () => props.identity?.canonicalPersonId,
  () => {
    failedPreviewIds.value = new Set()
    rosterSelection.value = props.identity?.externalPlayerId ?? ''
  },
)

watch(
  () => props.identity?.externalPlayerId,
  (externalPlayerId) => { rosterSelection.value = externalPlayerId ?? '' },
)

function finitePercent(value: number | null | undefined): string {
  return identityConfidenceLabel(value)
}

function rosterNumberSortValue(value: string | null | undefined): number {
  if (!value?.trim()) return Number.POSITIVE_INFINITY
  const number = Number(value)
  return Number.isFinite(number) ? number : Number.POSITIVE_INFINITY
}

function formatTime(seconds: number): string {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
  const minutes = Math.floor(safe / 60)
  const remainder = safe - minutes * 60
  return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(3).padStart(6, '0')}`
}

function evidenceDetail(evidence: CanonicalIdentityEvidence): string {
  const parts: string[] = []
  if (evidence.value !== null && evidence.value !== undefined && String(evidence.value).trim()) {
    parts.push(String(evidence.value))
  }
  if (evidence.confidence !== null && evidence.confidence !== undefined) {
    parts.push(finitePercent(evidence.confidence))
  }
  if (evidence.supportCount !== undefined) {
    parts.push(evidence.sampleCount !== undefined
      ? `${evidence.supportCount}/${evidence.sampleCount} samples`
      : `${evidence.supportCount} samples`)
  }
  return parts.join(' · ') || 'Recorded evidence'
}

function previewSource(observation: IdentityReviewObservation): string | null {
  if (failedPreviewIds.value.has(observation.id)) return null
  return observation.cropUrl?.trim() || observation.previewUrl?.trim() || null
}

function markPreviewFailed(observation: IdentityReviewObservation) {
  failedPreviewIds.value = new Set(failedPreviewIds.value).add(observation.id)
}

function boxStyle(observation: IdentityReviewObservation): Record<string, string> | null {
  const box = observation.bbox
  const width = Number(observation.frameWidth)
  const height = Number(observation.frameHeight)
  if (
    !box
    || !Number.isFinite(width)
    || !Number.isFinite(height)
    || width <= 0
    || height <= 0
  ) return null
  const clamp = (value: number) => Math.min(100, Math.max(0, value))
  return {
    left: `${clamp(box.x / width * 100)}%`,
    top: `${clamp(box.y / height * 100)}%`,
    width: `${clamp(box.width / width * 100)}%`,
    height: `${clamp(box.height / height * 100)}%`,
  }
}

function inspectFrame(observation: IdentityReviewObservation) {
  if (!props.identity || props.disabled) return
  emit('inspect-frame', inspectIdentityObservationDecision(
    props.identity.canonicalPersonId,
    observation,
  ))
}

function bindCandidate(externalPlayerId: string) {
  if (!props.identity || props.disabled || props.identity.externalPlayerId === externalPlayerId) return
  emit('bind-candidate', {
    canonicalPersonId: props.identity.canonicalPersonId,
    externalPlayerId,
  })
}

function bindRosterSelection() {
  const decision = manualRosterBinding.value
  if (!decision || props.disabled) return
  emit('bind-candidate', decision)
}

function rosterPlayerLabel(player: ExternalPlayer): string {
  const parts = [
    player.number ? `#${player.number}` : null,
    player.name,
    player.team_name || player.team_id || null,
  ].filter(Boolean)
  return parts.join(' · ')
}

function unbindRoster() {
  if (!props.identity || props.disabled || !props.identity.externalPlayerId) return
  emit('unbind-roster', { canonicalPersonId: props.identity.canonicalPersonId })
}

function clearRosterBinding() {
  if (!props.identity || props.disabled || !props.dedicatedUnbindActive) return
  emit('clear-roster-binding', { canonicalPersonId: props.identity.canonicalPersonId })
}

function rejectRosterCandidate(externalPlayerId: string) {
  if (!props.identity || props.disabled) return
  emit('reject-candidate', rosterReviewDecision(
    props.identity.canonicalPersonId,
    externalPlayerId,
  ))
}

function rejectLinkCandidate(candidate: IdentityReviewLinkCandidate) {
  if (!props.identity || props.disabled || candidate.status === 'rejected') return
  emit('reject-candidate', linkReviewDecision(
    props.identity.canonicalPersonId,
    candidate,
  ))
}

function cannotLink(candidate: IdentityReviewLinkCandidate) {
  if (!props.identity || props.disabled) return
  emit('cannot-link', cannotLinkReviewDecision(
    props.identity.canonicalPersonId,
    candidate,
  ))
}
</script>

<template>
  <section class="identity-review" :aria-labelledby="titleId">
    <header class="review-heading">
      <div>
        <span>Identity review</span>
        <h3 :id="titleId">{{ displayName }}</h3>
        <small v-if="identity">{{ identity.canonicalPersonId }}</small>
        <small v-else>Select a canonical person in the video or 3D scene to inspect identity evidence.</small>
      </div>
      <b v-if="identity" :class="identity.identityStatus">
        {{ canonicalPersonStatusLabel(identity.identityStatus) }}
      </b>
    </header>

    <template v-if="identity">
      <div v-if="identity.externalPlayerId || dedicatedUnbindActive" class="binding-state-actions">
        <button
          v-if="identity.externalPlayerId"
          type="button"
          class="negative"
          :disabled="disabled"
          :aria-label="`Unbind roster player from ${displayName}`"
          @click="unbindRoster"
        >UNBIND ROSTER PLAYER</button>
        <button
          v-if="dedicatedUnbindActive"
          type="button"
          :disabled="disabled"
          :aria-label="`Clear manual roster Unbind for ${displayName}`"
          @click="clearRosterBinding"
        >CLEAR MANUAL UNBIND</button>
        <small v-if="dedicatedUnbindActive">The explicit Unbind decision blocks roster proposals until it is cleared.</small>
      </div>

      <div class="review-summary" aria-label="Selected canonical person evidence summary">
        <div><span>Confidence</span><strong>{{ finitePercent(identity.identityConfidence) }}</strong></div>
        <div><span>Observations</span><strong>{{ identity.observationCount ?? identity.observations?.length ?? '—' }}</strong></div>
        <div><span>Tracklets</span><strong>{{ sourceTrackletCount }}</strong></div>
        <div><span>Evidence</span><strong>{{ identity.evidence.length }}</strong></div>
        <div><span>Team / role</span><strong>{{ identity.teamId || '—' }} · {{ identity.role || '—' }}</strong></div>
        <div><span>Jersey</span><strong>{{ identity.jerseyNumber ? `#${identity.jerseyNumber}` : '—' }}</strong></div>
      </div>

      <section class="review-section" :aria-labelledby="observationsId">
        <div class="section-heading">
          <div>
            <h4 :id="observationsId">Best observations</h4>
            <small>Quality-ranked evidence; a box-only tile is not an image crop.</small>
          </div>
          <span>{{ reviewObservations.length }}</span>
        </div>
        <div v-if="reviewObservations.length" class="observation-filmstrip">
          <article v-for="observation in reviewObservations" :key="observation.id" class="observation-card">
            <div class="observation-preview">
              <img
                v-if="previewSource(observation)"
                :src="previewSource(observation) || undefined"
                :alt="`Identity observation at ${formatTime(observation.sceneTime)}`"
                @error="markPreviewFailed(observation)"
              />
              <div v-else class="box-only-preview" role="img" :aria-label="`Detector box only at frame ${observation.frameIndex}; no image preview available`">
                <i aria-hidden="true" />
                <span>{{ failedPreviewIds.has(observation.id) ? 'Preview failed' : 'Box only' }}</span>
              </div>
              <i v-if="previewSource(observation) && boxStyle(observation) && !observation.cropUrl" class="bbox-overlay" :style="boxStyle(observation) || undefined" aria-hidden="true" />
            </div>
            <div class="observation-meta">
              <strong>{{ formatTime(observation.sceneTime) }}</strong>
              <small>Frame {{ observation.frameIndex }} · {{ finitePercent(observation.quality ?? observation.confidence) }}</small>
              <small v-if="observation.bbox">
                Box {{ Math.round(observation.bbox.x) }}, {{ Math.round(observation.bbox.y) }} · {{ Math.round(observation.bbox.width) }}×{{ Math.round(observation.bbox.height) }} px
              </small>
              <small v-if="observation.rejectionReasons?.length" class="observation-rejection">
                {{ observation.rejectionReasons.join(' · ') }}
              </small>
            </div>
            <button
              type="button"
              :disabled="disabled"
              :aria-label="`Inspect frame ${observation.frameIndex} for ${displayName}`"
              @click="inspectFrame(observation)"
            >
              INSPECT FRAME
            </button>
          </article>
        </div>
        <p v-else-if="observationsWereSupplied" class="empty-state" role="status">
          No reviewable crop, frame preview, or valid detector box was supplied.
        </p>
        <p v-else class="empty-state" role="status">
          No observation previews are available for this canonical person. Rebuild identity evidence or inspect a detected person directly in Video Review.
        </p>
      </section>

      <section class="review-section" :aria-labelledby="evidenceId">
        <div class="section-heading">
          <h4 :id="evidenceId">Published evidence</h4>
          <span>{{ identity.evidence.length }}</span>
        </div>
        <ul v-if="identity.evidence.length" class="evidence-list">
          <li v-for="evidence in identity.evidence" :key="evidence.id">
            <i :class="{ manual: evidence.manual || evidence.kind === 'manual' }" aria-hidden="true" />
            <span>
              <strong>{{ identityEvidenceKindLabel(evidence.kind) }} · {{ evidence.label }}</strong>
              <small>{{ evidenceDetail(evidence) }}</small>
              <em v-if="evidence.source || evidence.model">{{ evidence.source || 'model' }}{{ evidence.model ? ` · ${evidence.model}` : '' }}</em>
            </span>
          </li>
        </ul>
        <p v-else class="empty-state" role="status">No identity evidence has been published. The person remains anonymous.</p>
      </section>

      <section class="review-section" :aria-labelledby="workersId">
        <div class="section-heading">
          <div>
            <h4 :id="workersId">Identity workers</h4>
            <small>Readiness and rejection counts from the reconstruction run.</small>
          </div>
          <span>{{ workerStates?.length ?? '—' }}</span>
        </div>
        <ul v-if="workerStates?.length" class="worker-list">
          <li v-for="worker in workerStates" :key="worker.id" :class="`status-${worker.status}`">
            <div>
              <strong>{{ worker.label }}</strong>
              <small>{{ worker.backend || 'Backend unavailable' }}{{ worker.modelVersion ? ` · ${worker.modelVersion}` : '' }}</small>
            </div>
            <b>{{ identityReviewWorkerStatusLabel(worker.status) }}</b>
            <p v-if="worker.requestedCount != null || worker.usableCount != null || worker.rejectedCount != null">
              Requested {{ worker.requestedCount ?? '—' }} · usable {{ worker.usableCount ?? '—' }} · rejected {{ worker.rejectedCount ?? '—' }}
            </p>
            <p v-if="worker.rejectionReasons?.length" class="worker-rejections">{{ worker.rejectionReasons.join(' · ') }}</p>
            <p v-if="worker.detail">{{ worker.detail }}</p>
          </li>
        </ul>
        <p v-else-if="workerStates" class="empty-state" role="status">No ReID or jersey OCR workers are configured for this review.</p>
        <p v-else class="empty-state" role="status">Worker readiness was not supplied. Availability cannot be inferred from missing evidence.</p>
      </section>

      <section class="review-section" :aria-labelledby="candidatesId">
        <div class="section-heading manual-roster-heading">
          <div>
            <h4 :id="`${candidatesId}-manual`">Manual roster binding</h4>
            <small>Choose any saved roster player when automatic identity evidence abstains.</small>
          </div>
          <span>{{ rosterPlayers.length }}</span>
        </div>
        <div v-if="rosterPlayers.length" class="manual-roster-picker" :aria-labelledby="`${candidatesId}-manual`">
          <select v-model="rosterSelection" :disabled="disabled" aria-label="Manual roster player">
            <option value="">Choose saved roster player</option>
            <option
              v-if="currentBindingMissingFromRoster && identity.externalPlayerId"
              :value="identity.externalPlayerId"
              disabled
            >Bound ID missing from roster · {{ identity.externalPlayerId }}</option>
            <option
              v-for="(player, playerIndex) in sortedRosterPlayers"
              :key="`${player.id}-${player.team_id || ''}-${player.name}-${playerIndex}`"
              :value="player.id"
              :disabled="(rosterIdCounts.get(player.id) ?? 0) !== 1"
            >{{ rosterPlayerLabel(player) }}{{ (rosterIdCounts.get(player.id) ?? 0) !== 1 ? ' · duplicate ID' : '' }}</option>
          </select>
          <button
            type="button"
            :disabled="disabled || !canBindRosterSelection"
            aria-label="Bind selected roster player"
            @click="bindRosterSelection"
          >BIND SELECTED</button>
          <small v-if="rosterSelection && rosterSelectionCount > 1" class="manual-roster-warning" role="alert">
            This external player ID occurs {{ rosterSelectionCount }} times. Resolve the duplicate roster rows before binding.
          </small>
          <small v-else-if="identity.externalPlayerId">
            Current binding · {{ currentRosterBindingLabel }}
          </small>
        </div>
        <p v-else class="empty-state" role="status">No persisted roster is available for manual binding.</p>

        <div class="section-heading">
          <div>
            <h4 :id="candidatesId">Roster candidates</h4>
            <small>Ranked suggestions remain hypotheses until explicitly bound.</small>
          </div>
          <span>{{ candidates.length }}</span>
        </div>
        <ol v-if="candidates.length" class="candidate-list">
          <li v-for="(candidate, index) in candidates" :key="candidate.externalPlayerId">
            <span class="rank">{{ index + 1 }}</span>
            <span class="candidate-copy">
              <strong>{{ candidate.number ? `#${candidate.number} · ` : '' }}{{ candidate.name }}</strong>
              <small>{{ finitePercent(candidate.confidence) }}{{ candidate.position ? ` · ${candidate.position}` : '' }}</small>
              <em v-if="candidate.reasons?.length">{{ candidate.reasons.join(' · ') }}</em>
              <em v-if="candidate.conflicts?.length" class="candidate-conflicts">{{ candidate.conflicts.join(' · ') }}</em>
            </span>
            <span class="candidate-actions">
              <button
                type="button"
                :disabled="disabled || candidate.eligible === false || identity.externalPlayerId === candidate.externalPlayerId"
                :aria-label="identity.externalPlayerId === candidate.externalPlayerId ? `${candidate.name} is already bound` : `Bind ${candidate.name} to ${displayName}`"
                @click="bindCandidate(candidate.externalPlayerId)"
              >{{ identity.externalPlayerId === candidate.externalPlayerId ? 'BOUND' : 'BIND' }}</button>
              <button
                type="button"
                class="negative"
                :disabled="disabled || identity.externalPlayerId === candidate.externalPlayerId"
                :aria-label="`Reject ${candidate.name} as a candidate for ${displayName}`"
                @click="rejectRosterCandidate(candidate.externalPlayerId)"
              >REJECT</button>
            </span>
          </li>
        </ol>
        <p v-else-if="!rosterPlayers.length" class="empty-state" role="status">No persisted match roster is available. A player name cannot be proposed safely.</p>
        <p v-else class="empty-state" role="status">The resolver produced no roster candidates for this person.</p>
      </section>

      <section v-if="linkCandidates !== null" class="review-section" :aria-labelledby="linksId">
        <div class="section-heading">
          <div>
            <h4 :id="linksId">Identity-link review</h4>
            <small>Possible cross-gap links and their rejection reasons.</small>
          </div>
          <span>{{ linkCandidates?.length ?? '—' }}</span>
        </div>
        <ul v-if="linkCandidates?.length" class="link-list">
          <li v-for="candidate in linkCandidates" :key="candidate.id" :class="candidate.status">
            <span class="candidate-copy">
              <strong>{{ candidate.targetLabel || candidate.targetCanonicalPersonId }}</strong>
              <small>{{ candidate.status === 'rejected' ? 'Rejected by resolver' : 'Needs review' }}{{ candidate.confidence != null ? ` · ${finitePercent(candidate.confidence)}` : '' }}{{ candidate.source ? ` · ${candidate.source}` : '' }}</small>
              <em v-if="candidate.reasons?.length">{{ candidate.reasons.join(' · ') }}</em>
              <em v-else>No rejection or support reasons were supplied.</em>
            </span>
            <span class="candidate-actions">
              <button
                type="button"
                class="negative"
                :disabled="disabled || candidate.status === 'rejected'"
                :aria-label="`Reject identity link to ${candidate.targetLabel || candidate.targetCanonicalPersonId}`"
                @click="rejectLinkCandidate(candidate)"
              >REJECT LINK</button>
              <button
                type="button"
                class="negative"
                :disabled="disabled"
                :aria-label="`Mark ${displayName} and ${candidate.targetLabel || candidate.targetCanonicalPersonId} as cannot-link`"
                @click="cannotLink(candidate)"
              >CANNOT LINK</button>
            </span>
          </li>
        </ul>
        <p v-else class="empty-state" role="status">There are no pending or rejected identity-link hypotheses for this person.</p>
      </section>

      <section class="review-section" :aria-labelledby="conflictsId">
        <div class="section-heading">
          <h4 :id="conflictsId">Conflicts</h4>
          <span>{{ identity.conflicts.length }}</span>
        </div>
        <ul v-if="identity.conflicts.length" class="conflict-list">
          <li v-for="conflict in identity.conflicts" :key="conflict.id" :class="conflict.severity">
            <strong>{{ conflict.code.replaceAll('-', ' ') }}</strong>
            <small>{{ conflict.message }}</small>
          </li>
        </ul>
        <p v-else class="empty-state" role="status">No published identity conflicts.</p>
      </section>
    </template>
  </section>
</template>

<style scoped>
.identity-review { display: flex; flex-direction: column; gap: 16px; min-width: 0; color: #dce3de; }
.review-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; padding-bottom: 14px; border-bottom: 1px solid var(--line); }
.review-heading > div { min-width: 0; }
.review-heading span, .review-summary span, .section-heading > span { color: #76817b; font: 500 9px 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
.review-heading h3 { margin: 4px 0 3px; color: #eef2ee; font-size: 17px; line-height: 1.2; overflow-wrap: anywhere; }
.review-heading small { color: #758079; font: 500 9px 'DM Mono', monospace; overflow-wrap: anywhere; }
.review-heading > b { flex: 0 0 auto; padding: 4px 6px; border: 1px solid var(--line-strong); border-radius: 3px; color: #89938d; font: 600 9px 'DM Mono', monospace; }
.review-heading > b.resolved { border-color: rgba(113,226,170,.38); color: #71e2aa; }
.review-heading > b.provisional { border-color: rgba(255,211,106,.4); color: var(--accent); }
.review-heading > b.excluded { border-color: rgba(255,104,79,.45); color: #ff8e7c; }
.review-summary { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
.binding-state-actions { display: flex; flex-wrap: wrap; gap: 7px; }
.binding-state-actions button { min-height: 31px; padding: 0 9px; border: 1px solid rgba(255,211,106,.35); background: rgba(255,211,106,.035); color: var(--accent); font: 600 8px 'DM Mono', monospace; cursor: pointer; }
.binding-state-actions button.negative { border-color: rgba(255,142,124,.35); background: rgba(255,104,79,.035); color: #ff9d8d; }
.binding-state-actions small { flex-basis: 100%; color: #817763; font-size: 9px; line-height: 1.4; }
.review-summary > div { min-width: 0; padding: 9px; border: 1px solid var(--line); background: #0b0e0f; }
.review-summary span { display: block; margin-bottom: 5px; font-size: 8px; }
.review-summary strong { display: block; color: #dce3de; font: 600 12px 'DM Mono', monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.review-section { min-width: 0; }
.section-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
.section-heading > div { min-width: 0; }
.section-heading h4 { margin: 0; color: #aeb8b2; font-size: 12px; }
.section-heading small { display: block; margin-top: 3px; color: #68726c; font-size: 9px; line-height: 1.4; }
.section-heading > span { flex: 0 0 auto; padding: 2px 6px; border-radius: 9px; background: rgba(255,255,255,.07); }
.empty-state { margin: 0; padding: 11px; border: 1px dashed var(--line); color: #727d77; font-size: 10px; line-height: 1.5; }
.observation-filmstrip { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(170px, 210px); gap: 8px; padding-bottom: 5px; overflow-x: auto; overscroll-behavior-inline: contain; }
.observation-card { display: flex; flex-direction: column; min-width: 0; border: 1px solid var(--line); background: #0a0d0e; }
.observation-preview { position: relative; aspect-ratio: 16 / 10; overflow: hidden; background: #07090a; }
.observation-preview img { width: 100%; height: 100%; display: block; object-fit: contain; }
.bbox-overlay { position: absolute; border: 2px solid #71e2aa; box-shadow: 0 0 0 1px rgba(0,0,0,.7), inset 0 0 0 1px rgba(0,0,0,.45); pointer-events: none; }
.box-only-preview { width: 100%; height: 100%; display: grid; place-content: center; justify-items: center; gap: 8px; color: #727d77; font: 600 9px 'DM Mono', monospace; text-transform: uppercase; }
.box-only-preview i { width: 35px; height: 56px; border: 2px solid #71e2aa; background: rgba(113,226,170,.035); }
.observation-meta { display: flex; flex: 1; flex-direction: column; gap: 3px; padding: 8px; }
.observation-meta strong { color: #dce3de; font: 600 10px 'DM Mono', monospace; }
.observation-meta small { color: #77827b; font: 500 8px 'DM Mono', monospace; line-height: 1.35; }
.observation-meta .observation-rejection { color: #d0ad5f; }
.observation-card button, .candidate-actions button { min-height: 30px; border: 1px solid rgba(113,226,170,.35); background: rgba(113,226,170,.04); color: #71e2aa; font: 600 8px 'DM Mono', monospace; cursor: pointer; }
.observation-card > button { margin: 0 8px 8px; }
button:disabled { opacity: .45; cursor: default; }
.evidence-list, .worker-list, .candidate-list, .link-list, .conflict-list { margin: 0; padding: 0; list-style: none; border-top: 1px solid var(--line); }
.evidence-list li { display: grid; grid-template-columns: 7px minmax(0, 1fr); gap: 8px; padding: 8px 2px; border-bottom: 1px solid var(--line); }
.evidence-list li > i { width: 6px; height: 6px; margin-top: 4px; border-radius: 50%; background: #71e2aa; box-shadow: 0 0 7px rgba(113,226,170,.45); }
.evidence-list li > i.manual { background: #b994ff; box-shadow: 0 0 7px rgba(185,148,255,.45); }
.evidence-list li > span, .candidate-copy { min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.evidence-list strong, .candidate-copy strong { color: #c1cac4; font-size: 10px; }
.evidence-list small, .candidate-copy small { color: #77827b; font: 500 9px 'DM Mono', monospace; }
.evidence-list em, .candidate-copy em { color: #68736c; font-size: 8px; font-style: normal; line-height: 1.4; overflow-wrap: anywhere; }
.candidate-copy .candidate-conflicts { color: #d78273; }
.worker-list li { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px 8px; padding: 9px 2px; border-bottom: 1px solid var(--line); }
.worker-list li > div { min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.worker-list strong { color: #c1cac4; font-size: 10px; }
.worker-list small, .worker-list p { color: #77827b; font: 500 8px 'DM Mono', monospace; overflow-wrap: anywhere; }
.worker-list p { grid-column: 1 / -1; margin: 0; line-height: 1.4; }
.worker-list b { align-self: start; color: #71e2aa; font: 600 8px 'DM Mono', monospace; text-transform: uppercase; }
.worker-list li:not(.status-ready):not(.status-processing) b { color: #d0ad5f; }
.worker-list li.status-failed b, .worker-list li.status-unavailable b, .worker-list li.status-invalid-response b { color: #ff8e7c; }
.worker-list .worker-rejections { color: #d0ad5f; }
.candidate-list > li, .link-list > li { display: grid; grid-template-columns: 20px minmax(0, 1fr) auto; align-items: center; gap: 8px; min-height: 60px; padding: 8px 0; border-bottom: 1px solid var(--line); }
.manual-roster-heading { margin-top: 2px; }
.manual-roster-picker { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 7px; margin-bottom: 16px; }
.manual-roster-picker select { min-width: 0; height: 34px; padding: 0 8px; border: 1px solid var(--line-strong); background: #0b0e0f; color: #c8d0ca; font-size: 10px; }
.manual-roster-picker button { min-height: 34px; padding: 0 9px; border: 1px solid rgba(113,226,170,.35); background: rgba(113,226,170,.04); color: #71e2aa; font: 600 8px 'DM Mono', monospace; cursor: pointer; }
.manual-roster-picker > small { grid-column: 1 / -1; color: #68736c; font-size: 9px; line-height: 1.4; }
.manual-roster-picker > small.manual-roster-warning { color: #ff9d8d; }
.link-list > li { grid-template-columns: minmax(0, 1fr) auto; }
.link-list > li.rejected { border-bottom-color: rgba(255,104,79,.25); }
.rank { color: #59645e; font: 600 9px 'DM Mono', monospace; text-align: center; }
.candidate-actions { display: flex; flex-direction: column; gap: 5px; }
.candidate-actions button { min-width: 76px; padding: 0 7px; }
.candidate-actions button.negative { border-color: rgba(255,142,124,.35); background: rgba(255,104,79,.035); color: #ff9d8d; }
.conflict-list li { padding: 8px; border: 1px solid rgba(255,211,106,.2); border-top: 0; background: rgba(255,211,106,.035); display: flex; flex-direction: column; gap: 3px; }
.conflict-list li.blocking { border-color: rgba(255,104,79,.3); background: rgba(255,104,79,.045); }
.conflict-list strong { color: #d0ad5f; font: 600 9px 'DM Mono', monospace; text-transform: uppercase; }
.conflict-list li.blocking strong { color: #ff8e7c; }
.conflict-list small { color: #887e67; font-size: 9px; line-height: 1.4; }

@media (max-width: 720px) {
  .review-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .candidate-list > li, .link-list > li { grid-template-columns: 20px minmax(0, 1fr); }
  .link-list > li { grid-template-columns: minmax(0, 1fr); }
  .candidate-actions { grid-column: 2; flex-direction: row; }
  .link-list .candidate-actions { grid-column: 1; }
}
</style>
