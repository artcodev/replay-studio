<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type {
  ProjectIdentity,
  ProjectIdentityMembership,
  ProjectIdentityMembershipAssignment,
} from '../types/project'

const props = withDefaults(defineProps<{
  identities?: ProjectIdentity[]
  loading?: boolean
  error?: string | null
  assigningMembershipId?: string | null
  assignmentError?: string | null
}>(), {
  identities: () => [],
  loading: false,
  error: null,
  assigningMembershipId: null,
  assignmentError: null,
})

const emit = defineEmits<{
  refresh: []
  assignMembership: [assignment: ProjectIdentityMembershipAssignment]
}>()

const assignmentTargets = ref<Record<string, string>>({})

watch(
  () => props.identities,
  (identities) => {
    assignmentTargets.value = Object.fromEntries(
      identities.flatMap((identity) => identity.memberships.map(
        (membership) => [membership.id, membership.projectPersonId],
      )),
    )
  },
  { immediate: true },
)

const linkedCount = computed(() => props.identities.filter((identity) => identity.rosterPersonId).length)
const sceneMembershipCount = computed(() => props.identities.reduce(
  (total, identity) => total + identity.memberships.length,
  0,
))

function confidenceLabel(value?: number | null) {
  return value == null || !Number.isFinite(value) ? '—' : `${Math.round(value * 100)}%`
}

function assignmentTarget(membership: ProjectIdentityMembership) {
  return assignmentTargets.value[membership.id] ?? membership.projectPersonId
}

function targetIdentityName(projectPersonId: string) {
  return props.identities.find((identity) => identity.id === projectPersonId)?.displayName
    ?? 'selected project person'
}

function canAssign(membership: ProjectIdentityMembership) {
  const target = assignmentTarget(membership)
  return Boolean(
    target
    && target !== membership.projectPersonId
    && !props.assigningMembershipId,
  )
}

function assignMembership(membership: ProjectIdentityMembership) {
  const projectPersonId = assignmentTarget(membership)
  if (!canAssign(membership)) return
  emit('assignMembership', {
    membershipId: membership.id,
    currentProjectPersonId: membership.projectPersonId,
    projectPersonId,
    sceneId: membership.sceneId,
    scenePersonId: membership.scenePersonId,
  })
}
</script>

<template>
  <section class="project-identities" aria-labelledby="project-identities-title">
    <header>
      <div>
        <p>Cross-scene identity graph</p>
        <h2 id="project-identities-title">Project identities</h2>
        <span>One durable person can own observations from several moments and camera angles.</span>
      </div>
      <button type="button" :disabled="loading" @click="emit('refresh')">
        {{ loading ? 'Refreshing…' : 'Refresh' }}
      </button>
    </header>

    <div class="identity-metrics" aria-label="Identity summary">
      <article><strong>{{ identities.length }}</strong><span>project people</span></article>
      <article><strong>{{ linkedCount }}</strong><span>roster linked</span></article>
      <article><strong>{{ sceneMembershipCount }}</strong><span>scene memberships</span></article>
    </div>

    <p v-if="error" class="state error" role="alert">{{ error }}</p>
    <p v-else-if="assignmentError" class="state error" role="alert">{{ assignmentError }}</p>
    <p v-else-if="loading && !identities.length" class="state" role="status">Loading project identities…</p>
    <p v-else-if="!identities.length" class="state">
      No identities have been published yet. Reconstruct a moment to create the project identity graph.
    </p>

    <div v-else class="identity-list">
      <article v-for="identity in identities" :key="identity.id" :class="{ excluded: identity.status === 'excluded' }">
        <div class="identity-heading">
          <div>
            <strong>{{ identity.jerseyNumber ? `#${identity.jerseyNumber} · ` : '' }}{{ identity.displayName }}</strong>
            <span>{{ identity.rosterPersonId ? 'Canonical roster identity' : 'Anonymous project identity' }}</span>
          </div>
          <b>{{ confidenceLabel(identity.identityConfidence) }}</b>
        </div>
        <dl>
          <div><dt>Team</dt><dd>{{ identity.teamId || 'Unassigned' }}</dd></div>
          <div><dt>Role</dt><dd>{{ identity.role || 'Unknown' }}</dd></div>
          <div><dt>Status</dt><dd>{{ identity.status }}</dd></div>
          <div><dt>Scenes</dt><dd>{{ identity.memberships.length }}</dd></div>
        </dl>
        <ul v-if="identity.memberships.length" :aria-label="`${identity.displayName} scene memberships`">
          <li v-for="membership in identity.memberships" :key="membership.id">
            <span class="membership-source"><strong>{{ membership.sceneId }}</strong><small>{{ membership.scenePersonId }}</small></span>
            <span class="membership-evidence"><b>{{ membership.assignmentSource }}</b><small>{{ membership.observationCount }} observations</small></span>
            <div class="membership-assignment">
              <label :for="`identity-target-${membership.id}`">Assign to project person</label>
              <div>
                <select
                  :id="`identity-target-${membership.id}`"
                  v-model="assignmentTargets[membership.id]"
                  :disabled="Boolean(assigningMembershipId)"
                  :aria-label="`Assign ${membership.scenePersonId} from ${identity.displayName} to project person`"
                >
                  <option v-for="target in identities" :key="target.id" :value="target.id">
                    {{ target.jerseyNumber ? `#${target.jerseyNumber} · ` : '' }}{{ target.displayName }}{{ target.id === membership.projectPersonId ? ' (current)' : '' }}
                  </option>
                </select>
                <button
                  type="button"
                  :disabled="!canAssign(membership)"
                  :aria-label="`Assign ${membership.scenePersonId} to ${targetIdentityName(assignmentTarget(membership))}`"
                  @click="assignMembership(membership)"
                >
                  {{ assigningMembershipId === membership.id ? 'Assigning…' : 'Assign' }}
                </button>
              </div>
            </div>
          </li>
        </ul>
      </article>
    </div>
  </section>
</template>

<style scoped>
.project-identities { display: grid; gap: 16px; }
header, .identity-heading, li { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
h2, p, dl, dd { margin: 0; }
header p { color: #82a4c7; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; }
header h2 { margin: 3px 0; font-size: 18px; }
header span, .identity-heading span, small { color: #8fa2b8; font-size: 11px; }
button { min-height: 38px; padding: 0 14px; border: 1px solid #3b5068; border-radius: 8px; background: #152437; color: #e6f1fc; }
button:disabled { opacity: .55; }
.identity-metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.identity-metrics article { padding: 13px; border: 1px solid #2d4157; border-radius: 9px; background: #111e2d; }
.identity-metrics strong, .identity-metrics span { display: block; }
.identity-metrics strong { font-size: 23px; }
.identity-metrics span { margin-top: 3px; color: #8fa2b8; font-size: 11px; }
.state { padding: 18px; border: 1px dashed #3b5068; border-radius: 9px; color: #9eb0c3; }
.state.error { border-style: solid; border-color: #8f4f56; color: #ffb5b5; }
.identity-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 12px; }
.identity-list > article { padding: 14px; border: 1px solid #2d4157; border-radius: 10px; background: #111e2d; }
.identity-list > article.excluded { opacity: .65; }
.identity-heading > div { display: grid; gap: 3px; min-width: 0; }
.identity-heading > b { color: #7fc6ff; font-size: 12px; }
dl { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 7px; padding: 12px 0; }
dl div { min-width: 0; }
dt { color: #71859a; font-size: 9px; text-transform: uppercase; }
dd { overflow: hidden; margin-top: 3px; color: #dce8f3; font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
ul { display: grid; gap: 6px; margin: 0; padding: 10px 0 0; border-top: 1px solid #293b4e; list-style: none; }
li { align-items: end; }
li > span { display: grid; gap: 2px; min-width: 0; }
.membership-evidence { justify-items: end; text-align: right; }
li strong, li b { overflow: hidden; color: #cbd9e6; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
li b { color: #7fc6ff; font-weight: 500; }
.membership-assignment { display: grid; gap: 4px; min-width: min(100%, 280px); }
.membership-assignment > label { color: #71859a; font-size: 9px; text-transform: uppercase; }
.membership-assignment > div { display: grid; grid-template-columns: minmax(120px, 1fr) auto; gap: 6px; }
.membership-assignment select { min-width: 0; min-height: 34px; border: 1px solid #3b5068; border-radius: 7px; background: #0d1926; color: #dce8f3; font-size: 11px; }
.membership-assignment button { min-height: 34px; }
@media (max-width: 720px) {
  header { align-items: flex-start; }
  .identity-metrics { grid-template-columns: 1fr; }
  dl { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  li { display: grid; grid-template-columns: 1fr auto; }
  .membership-assignment { grid-column: 1 / -1; }
}
</style>
