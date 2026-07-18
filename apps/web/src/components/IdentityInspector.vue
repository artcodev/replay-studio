<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  canonicalPersonDisplayName,
  canonicalPersonSourceTrackletCount,
  canonicalPersonStatusLabel,
  confirmedRosterPlayer,
  identityConfidenceLabel,
  identityEvidenceKindLabel,
  rosterConfirmationPayload,
  topRosterCandidates,
  type ResolvedRosterCandidate,
} from '../lib/identityResolution'
import type { CanonicalIdentityEvidence, CanonicalPerson } from '../types/identity'
import type { ExternalPlayer } from '../types/match'

const props = withDefaults(defineProps<{
  identity: CanonicalPerson
  rosterPlayers?: ExternalPlayer[]
  dedicatedUnbindActive?: boolean
  disabled?: boolean
}>(), {
  rosterPlayers: () => [],
  dedicatedUnbindActive: false,
  disabled: false,
})

const emit = defineEmits<{
  'confirm-roster': [payload: { canonicalPersonId: string; externalPlayerId: string }]
  'unbind-roster': [payload: { canonicalPersonId: string }]
  'clear-roster-binding': [payload: { canonicalPersonId: string }]
}>()

const displayName = computed(() => canonicalPersonDisplayName(props.identity, props.rosterPlayers))
const statusLabel = computed(() => canonicalPersonStatusLabel(props.identity.identityStatus))
const confidenceLabel = computed(() => identityConfidenceLabel(props.identity.identityConfidence))
const sourceTrackletCount = computed(() => canonicalPersonSourceTrackletCount(props.identity))
const confirmedRoster = computed(() => confirmedRosterPlayer(props.identity, props.rosterPlayers))
const rosterCandidates = computed(() => topRosterCandidates(props.identity, props.rosterPlayers))
const rosterSelection = ref(props.identity.externalPlayerId ?? '')
const canConfirmRosterSelection = computed(() => (
  Boolean(rosterSelection.value)
  && rosterSelection.value !== props.identity.externalPlayerId
  && props.rosterPlayers.some((player) => player.id === rosterSelection.value)
))

watch(
  () => [props.identity.canonicalPersonId, props.identity.externalPlayerId] as const,
  () => {
    rosterSelection.value = props.identity.externalPlayerId ?? ''
  },
)

function evidenceDetail(evidence: CanonicalIdentityEvidence): string {
  const detail: string[] = []
  if (evidence.value !== null && evidence.value !== undefined && String(evidence.value).trim()) {
    detail.push(String(evidence.value))
  }
  if (evidence.confidence !== null && evidence.confidence !== undefined) {
    detail.push(identityConfidenceLabel(evidence.confidence))
  }
  if (evidence.supportCount !== undefined) {
    detail.push(evidence.sampleCount !== undefined
      ? `${evidence.supportCount}/${evidence.sampleCount} samples`
      : `${evidence.supportCount} samples`)
  }
  return detail.join(' · ') || 'Recorded evidence'
}

function confirmCandidate(candidate: ResolvedRosterCandidate) {
  if (props.disabled || props.identity.externalPlayerId === candidate.externalPlayerId) return
  emit('confirm-roster', rosterConfirmationPayload(props.identity.canonicalPersonId, candidate.externalPlayerId))
}

function unbindRoster() {
  if (props.disabled || !props.identity.externalPlayerId) return
  emit('unbind-roster', { canonicalPersonId: props.identity.canonicalPersonId })
}

function clearRosterBinding() {
  if (props.disabled || !props.dedicatedUnbindActive) return
  emit('clear-roster-binding', { canonicalPersonId: props.identity.canonicalPersonId })
}

function confirmRosterSelection() {
  if (props.disabled || !canConfirmRosterSelection.value) return
  emit('confirm-roster', rosterConfirmationPayload(
    props.identity.canonicalPersonId,
    rosterSelection.value,
  ))
}
</script>

<template>
  <section
    class="identity-inspector"
    :class="`status-${identity.identityStatus}`"
    :aria-label="`Canonical identity ${displayName}`"
  >
    <header class="identity-heading">
      <div>
        <span>Canonical person</span>
        <h3>{{ displayName }}</h3>
        <small v-if="identity.externalPlayerId">
          Confirmed roster binding · {{ confirmedRoster?.number ? `#${confirmedRoster.number} · ` : '' }}{{ confirmedRoster?.name || identity.externalPlayerId }}
        </small>
        <small v-else>Anonymous identity · no confirmed roster binding</small>
      </div>
      <b :class="identity.identityStatus">{{ statusLabel }}</b>
    </header>

    <button
      v-if="identity.externalPlayerId"
      class="unbind-roster"
      type="button"
      :disabled="disabled"
      :aria-label="`Unbind roster player from ${displayName}`"
      @click="unbindRoster"
    >
      UNBIND ROSTER PLAYER
    </button>

    <div v-if="dedicatedUnbindActive" class="clear-roster-decision">
      <button
        type="button"
        :disabled="disabled"
        :aria-label="`Clear manual roster Unbind for ${displayName}`"
        @click="clearRosterBinding"
      >
        CLEAR MANUAL UNBIND
      </button>
      <small>Remove the explicit Unbind tombstone and let identity evidence propose a roster player again.</small>
    </div>

    <div class="identity-summary" aria-label="Canonical identity summary">
      <div>
        <span>Identity confidence</span>
        <strong>{{ confidenceLabel }}</strong>
      </div>
      <div>
        <span>Source tracklets</span>
        <strong>{{ sourceTrackletCount }}</strong>
      </div>
      <div>
        <span>Observations</span>
        <strong>{{ identity.observationCount ?? '—' }}</strong>
      </div>
      <div>
        <span>Evidence</span>
        <strong>{{ identity.evidence.length }}</strong>
      </div>
      <div>
        <span>Identity source</span>
        <strong>{{ identity.identitySource || '—' }}</strong>
      </div>
      <div>
        <span>Jersey number</span>
        <strong>{{ identity.jerseyNumber ? `#${identity.jerseyNumber}` : '—' }}</strong>
      </div>
    </div>

    <section class="identity-section evidence-section" aria-labelledby="identity-evidence-heading">
      <div class="identity-section-heading">
        <h4 id="identity-evidence-heading">Identity evidence</h4>
        <span>{{ identity.evidence.length }}</span>
      </div>
      <ul v-if="identity.evidence.length">
        <li v-for="evidence in identity.evidence" :key="evidence.id">
          <i :class="{ manual: evidence.manual || evidence.kind === 'manual' }" aria-hidden="true" />
          <span>
            <strong>{{ identityEvidenceKindLabel(evidence.kind) }} · {{ evidence.label }}</strong>
            <small>{{ evidenceDetail(evidence) }}</small>
            <em v-if="evidence.source || evidence.model">
              {{ evidence.source || 'model' }}{{ evidence.model ? ` · ${evidence.model}` : '' }}
            </em>
          </span>
        </li>
      </ul>
      <p v-else>No identity evidence has been published yet.</p>
    </section>

    <section
      v-if="identity.conflicts.length"
      class="identity-section conflict-section"
      aria-labelledby="identity-conflicts-heading"
    >
      <div class="identity-section-heading">
        <h4 id="identity-conflicts-heading">Conflicts</h4>
        <span>{{ identity.conflicts.length }}</span>
      </div>
      <ul>
        <li v-for="conflict in identity.conflicts" :key="conflict.id" :class="conflict.severity">
          <strong>{{ conflict.code.replaceAll('-', ' ') }}</strong>
          <small>{{ conflict.message }}</small>
        </li>
      </ul>
    </section>

    <section class="identity-section roster-section" aria-labelledby="roster-candidates-heading">
      <div class="identity-section-heading">
        <div>
          <h4 id="full-roster-heading">Roster binding</h4>
          <small>Choose any player, including one not suggested by the resolver.</small>
        </div>
        <span>{{ rosterPlayers.length }}</span>
      </div>
      <div v-if="rosterPlayers.length" class="full-roster-picker" aria-labelledby="full-roster-heading">
        <select v-model="rosterSelection" :disabled="disabled" aria-label="Full roster player">
          <option value="">Choose roster player</option>
          <option v-for="player in rosterPlayers" :key="player.id" :value="player.id">
            {{ player.number ? `#${player.number} · ` : '' }}{{ player.name }}{{ player.team_name ? ` · ${player.team_name}` : '' }}{{ player.position ? ` · ${player.position}` : '' }}
          </option>
        </select>
        <button
          type="button"
          :disabled="disabled || !canConfirmRosterSelection"
          aria-label="Bind selected roster player"
          @click="confirmRosterSelection"
        >
          BIND
        </button>
      </div>
      <p v-else>No roster is available. Bind match data or retry after the provider recovers.</p>

      <div class="identity-section-heading">
        <div>
          <h4 id="roster-candidates-heading">Roster candidates</h4>
          <small>Suggestions stay unconfirmed until you choose one.</small>
        </div>
        <span>{{ identity.rosterCandidates.length }}</span>
      </div>
      <ol v-if="rosterCandidates.length">
        <li v-for="candidate in rosterCandidates" :key="candidate.externalPlayerId">
          <span class="candidate-rank">{{ candidate.rank ?? '·' }}</span>
          <span class="candidate-copy">
            <strong>{{ candidate.number ? `#${candidate.number} · ` : '' }}{{ candidate.name }}</strong>
            <small>
              {{ identityConfidenceLabel(candidate.confidence) }}{{ candidate.position ? ` · ${candidate.position}` : '' }}
            </small>
            <em v-if="candidate.reasons?.length">{{ candidate.reasons.join(' · ') }}</em>
          </span>
          <button
            type="button"
            :disabled="disabled || identity.externalPlayerId === candidate.externalPlayerId"
            :aria-label="identity.externalPlayerId === candidate.externalPlayerId
              ? `${candidate.name} is the confirmed roster player`
              : `Confirm ${candidate.name} for ${identity.displayName}`"
            @click="confirmCandidate(candidate)"
          >
            {{ identity.externalPlayerId === candidate.externalPlayerId ? 'CONFIRMED' : 'CONFIRM' }}
          </button>
        </li>
      </ol>
      <p v-else>No roster candidates are available. Bind match data or wait for jersey evidence.</p>
    </section>
  </section>
</template>

<style scoped>
.identity-inspector {
  display: flex;
  flex-direction: column;
  gap: 14px;
  color: #dce3de;
}

.identity-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--line);
}

.identity-heading > div { min-width: 0; }
.identity-heading span,
.identity-summary span,
.identity-section-heading > span {
  color: #76817b;
  font: 500 9px 'DM Mono', monospace;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.identity-heading h3 { margin: 4px 0 3px; color: #eef2ee; font-size: 16px; line-height: 1.2; }
.identity-heading small { color: #758079; font-size: 10px; line-height: 1.4; }
.identity-heading > b {
  flex: 0 0 auto;
  padding: 4px 6px;
  border: 1px solid var(--line-strong);
  border-radius: 3px;
  color: #89938d;
  font: 600 9px 'DM Mono', monospace;
}
.identity-heading > b.resolved { border-color: rgba(113,226,170,.38); color: #71e2aa; }
.identity-heading > b.provisional { border-color: rgba(255,211,106,.4); color: var(--accent); }
.identity-heading > b.excluded { border-color: rgba(255,104,79,.45); color: #ff8e7c; }
.unbind-roster {
  width: 100%;
  min-height: 32px;
  border: 1px solid rgba(255,142,124,.35);
  background: rgba(255,104,79,.035);
  color: #ff9d8d;
  font: 600 9px 'DM Mono', monospace;
  letter-spacing: .04em;
  cursor: pointer;
}
.unbind-roster:disabled { opacity: .45; cursor: default; }
.clear-roster-decision { display: flex; flex-direction: column; gap: 5px; }
.clear-roster-decision button {
  width: 100%;
  min-height: 32px;
  border: 1px solid rgba(255,211,106,.35);
  background: rgba(255,211,106,.035);
  color: var(--accent);
  font: 600 9px 'DM Mono', monospace;
  letter-spacing: .04em;
  cursor: pointer;
}
.clear-roster-decision button:disabled { opacity: .45; cursor: default; }
.clear-roster-decision small { color: #776f5d; font-size: 9px; line-height: 1.4; }

.identity-summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
}
.identity-summary > div { padding: 9px; border: 1px solid var(--line); background: #0b0e0f; }
.identity-summary span { display: block; margin-bottom: 5px; font-size: 8px; }
.identity-summary strong { color: #dce3de; font: 600 13px 'DM Mono', monospace; }

.identity-section { padding-top: 2px; }
.identity-section-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; margin-bottom: 7px; }
.identity-section-heading > div { min-width: 0; }
.identity-section-heading h4 { margin: 0; color: #aeb8b2; font-size: 11px; }
.identity-section-heading small { display: block; margin-top: 3px; color: #68726c; font-size: 9px; line-height: 1.35; }
.identity-section-heading > span { padding: 2px 5px; border-radius: 8px; background: rgba(255,255,255,.07); }
.identity-section ul,
.identity-section ol { margin: 0; padding: 0; list-style: none; border-top: 1px solid var(--line); }
.identity-section p { margin: 0; padding: 10px; border: 1px dashed var(--line); color: #6c7771; font-size: 10px; line-height: 1.45; }

.evidence-section li { display: grid; grid-template-columns: 7px minmax(0, 1fr); gap: 8px; padding: 8px 2px; border-bottom: 1px solid var(--line); }
.evidence-section li > i { width: 6px; height: 6px; margin-top: 4px; border-radius: 50%; background: #71e2aa; box-shadow: 0 0 7px rgba(113,226,170,.45); }
.evidence-section li > i.manual { background: #b994ff; box-shadow: 0 0 7px rgba(185,148,255,.45); }
.evidence-section li > span,
.candidate-copy { min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.evidence-section strong,
.candidate-copy strong { color: #bfc8c2; font-size: 10px; }
.evidence-section small,
.candidate-copy small { color: #77827b; font: 500 9px 'DM Mono', monospace; }
.evidence-section em,
.candidate-copy em { color: #5f6b64; font-size: 8px; font-style: normal; line-height: 1.35; }

.conflict-section li { padding: 8px; border: 1px solid rgba(255,211,106,.2); border-top: 0; background: rgba(255,211,106,.035); display: flex; flex-direction: column; gap: 3px; }
.conflict-section li.blocking { border-color: rgba(255,104,79,.3); background: rgba(255,104,79,.045); }
.conflict-section strong { color: #d0ad5f; font: 600 9px 'DM Mono', monospace; text-transform: uppercase; }
.conflict-section li.blocking strong { color: #ff8e7c; }
.conflict-section small { color: #887e67; font-size: 9px; line-height: 1.4; }

.roster-section ol > li {
  min-height: 55px;
  display: grid;
  grid-template-columns: 19px minmax(0, 1fr) auto;
  align-items: center;
  gap: 8px;
  padding: 7px 0;
  border-bottom: 1px solid var(--line);
}
.full-roster-picker {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 7px;
  margin-bottom: 14px;
}
.full-roster-picker select {
  min-width: 0;
  height: 34px;
  padding: 0 8px;
  border: 1px solid var(--line-strong);
  background: #0b0e0f;
  color: #c8d0ca;
  font-size: 10px;
}
.candidate-rank { color: #59645e; font: 600 9px 'DM Mono', monospace; text-align: center; }
.roster-section button {
  min-width: 65px;
  height: 28px;
  padding: 0 7px;
  border: 1px solid rgba(113,226,170,.35);
  background: rgba(113,226,170,.04);
  color: #71e2aa;
  font: 600 8px 'DM Mono', monospace;
  cursor: pointer;
}
.roster-section button:disabled { border-color: var(--line); color: #68736c; opacity: .7; cursor: default; }

@media (min-width: 1081px) {
  .identity-heading h3 { font-size: 17px; }
  .identity-heading small,
  .identity-section-heading small,
  .identity-section p,
  .evidence-section strong,
  .candidate-copy strong { font-size: 11px; }
  .evidence-section small,
  .candidate-copy small { font-size: 10px; }
}
</style>
