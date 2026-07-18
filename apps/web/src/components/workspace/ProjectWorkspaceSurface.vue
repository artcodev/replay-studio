<script setup lang="ts">
import ProjectDashboard, { type ProjectDashboardTab } from '../ProjectDashboard.vue'
import ProjectMatchSearch from '../ProjectMatchSearch.vue'
import { injectProjectWorkspace } from '../../composables/useProjectWorkspace'

defineProps<{ activeTab: ProjectDashboardTab }>()
const emit = defineEmits<{
  'update:activeTab': [tab: ProjectDashboardTab]
  'retry-route': []
}>()

const workspace = injectProjectWorkspace()
const {
  projects,
  project,
  loading,
  mutationBusy,
  error,
  select: selectProject,
  create: createProject,
} = workspace.catalog
const { assets, segments, openTimeline, selectSegment } = workspace.media
const {
  snapshot: match,
  busy: matchBusy,
  searchQuery: matchSearchQuery,
  searchDate: matchSearchDate,
  candidates: matchCandidates,
  searchLoading: matchSearchLoading,
  searchError: matchSearchError,
  selectingId: selectingMatchId,
  search: searchMatches,
  select: selectMatch,
  refresh: refreshMatch,
} = workspace.match
const {
  rows: identities,
  loading: identitiesLoading,
  error: identitiesError,
  assigningMembershipId: assigningIdentityMembershipId,
  assignmentError: identityAssignmentError,
  refresh: refreshIdentities,
  assign: assignIdentityMembership,
} = workspace.identities
const analysisJobs = workspace.jobs
</script>

<template>
  <section class="project-workspace">
    <div v-if="error && !projects.length" class="project-workspace-fatal">
      <span class="fatal-code">PROJECT API OFFLINE</span>
      <h2>The studio could not load projects.</h2>
      <p>{{ error }}</p>
      <button class="button primary" @click="emit('retry-route')">Try again</button>
    </div>
    <template v-else>
      <div v-if="error" class="project-error-banner" role="alert">
        <span>{{ error }}</span>
        <button type="button" aria-label="Dismiss project error" @click="error = null">×</button>
      </div>
      <ProjectDashboard
        :active-tab="activeTab"
        :projects="projects"
        :project="project"
        :match="match"
        :assets="assets"
        :segments="segments"
        :identities="identities"
        :jobs="analysisJobs.jobs.value"
        :loading="loading || mutationBusy"
        :match-busy="matchBusy || Boolean(selectingMatchId)"
        :jobs-loading="analysisJobs.loading.value"
        :jobs-error="analysisJobs.error.value"
        :identities-loading="identitiesLoading"
        :identities-error="identitiesError"
        :assigning-identity-membership-id="assigningIdentityMembershipId"
        :identity-assignment-error="identityAssignmentError"
        :canceling-job-ids="analysisJobs.cancelingJobIds.value"
        :jobs-updated-at="analysisJobs.lastUpdatedAt.value"
        @update:active-tab="emit('update:activeTab', $event)"
        @select-project="selectProject"
        @create-project="createProject"
        @open-editor="openTimeline()"
        @open-timeline="openTimeline"
        @select-segment="selectSegment"
        @refresh-match="refreshMatch"
        @refresh-identities="refreshIdentities"
        @assign-identity-membership="assignIdentityMembership"
        @cancel-job="analysisJobs.cancel"
        @retry-jobs="analysisJobs.refresh"
      >
        <template #match-tools>
          <ProjectMatchSearch
            v-model:query="matchSearchQuery"
            v-model:date="matchSearchDate"
            :candidates="matchCandidates"
            :loading="matchSearchLoading"
            :selecting-id="selectingMatchId"
            :error="matchSearchError"
            :disabled="loading || !project"
            @search-query="searchMatches('query')"
            @search-date="searchMatches('date')"
            @select="selectMatch"
          />
        </template>
      </ProjectDashboard>
    </template>
  </section>
</template>
