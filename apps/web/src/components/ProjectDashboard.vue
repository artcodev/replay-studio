<script setup lang="ts">
import { computed } from 'vue'
import type { CanonicalMatch } from '../types/match'
import type {
  AnalysisJob,
  Project,
  ProjectAsset,
  ProjectIdentity,
  ProjectIdentityMembershipAssignment,
  ProjectSegment,
} from '../types/project'
import { isAnalysisJobActive } from '../lib/analysisJobs'
import AnalysisJobsPanel from './AnalysisJobsPanel.vue'
import ProjectIdentitiesPanel from './ProjectIdentitiesPanel.vue'
import ProjectMatchTab from './ProjectMatchTab.vue'

export type ProjectDashboardTab = 'overview' | 'match' | 'identities' | 'analysis'

const props = withDefaults(defineProps<{
  projects?: Project[]
  project?: Project | null
  match?: CanonicalMatch | null
  assets?: ProjectAsset[]
  segments?: ProjectSegment[]
  identities?: ProjectIdentity[]
  jobs?: AnalysisJob[]
  activeTab?: ProjectDashboardTab
  loading?: boolean
  matchBusy?: boolean
  jobsLoading?: boolean
  jobsError?: string | null
  identitiesLoading?: boolean
  identitiesError?: string | null
  assigningIdentityMembershipId?: string | null
  identityAssignmentError?: string | null
  cancelingJobIds?: string[]
  jobsUpdatedAt?: string | null
  allowMatchImport?: boolean
}>(), {
  projects: () => [],
  project: null,
  match: null,
  assets: () => [],
  segments: () => [],
  identities: () => [],
  jobs: () => [],
  activeTab: 'overview',
  loading: false,
  matchBusy: false,
  jobsLoading: false,
  jobsError: null,
  identitiesLoading: false,
  identitiesError: null,
  assigningIdentityMembershipId: null,
  identityAssignmentError: null,
  cancelingJobIds: () => [],
  jobsUpdatedAt: null,
  allowMatchImport: false,
})

const emit = defineEmits<{
  'update:activeTab': [tab: ProjectDashboardTab]
  selectProject: [projectId: string]
  createProject: []
  openEditor: []
  openTimeline: [assetId: string]
  selectSegment: [segmentId: string]
  refreshMatch: []
  importMatch: []
  cancelJob: [runId: string]
  retryJobs: []
  refreshIdentities: []
  assignIdentityMembership: [assignment: ProjectIdentityMembershipAssignment]
}>()

const activeJobs = computed(() => props.jobs.filter(isAnalysisJobActive))
const readySegments = computed(() => props.segments.filter((segment) => segment.status === 'ready').length)
const readyTimelineAssets = computed(() => props.assets.filter(
  (asset) => asset.status === 'ready' && Boolean(asset.timelineSceneId),
))

function selectProject(event: Event) {
  const projectId = (event.target as HTMLSelectElement).value
  if (projectId && projectId !== props.project?.id) emit('selectProject', projectId)
}

function updatedLabel(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
}

function durationLabel(seconds: number | null) {
  if (seconds === null || !Number.isFinite(seconds)) return 'Duration pending'
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.round(seconds % 60)
  return `${minutes}:${String(remainder).padStart(2, '0')}`
}

function segmentRange(segment: ProjectSegment) {
  return `${segment.start.toFixed(2)}–${segment.end.toFixed(2)}s`
}
</script>

<template>
  <section class="project-dashboard" aria-labelledby="project-dashboard-title">
    <header class="dashboard-header">
      <div>
        <p class="eyebrow">Replay Studio workspace</p>
        <h1 id="project-dashboard-title">{{ project?.title || 'Projects' }}</h1>
        <span v-if="project">Updated {{ updatedLabel(project.updatedAt) }}</span>
      </div>
      <div class="project-picker">
        <label for="project-dashboard-picker">Project</label>
        <select
          id="project-dashboard-picker"
          :value="project?.id ?? ''"
          :disabled="loading || !projects.length"
          @change="selectProject"
        >
          <option value="" :disabled="Boolean(project)">{{ projects.length ? 'Choose a project…' : 'No projects' }}</option>
          <option v-for="item in projects" :key="item.id" :value="item.id">
            {{ item.title }}{{ item.matchId ? '' : ' · no match assigned' }}
          </option>
        </select>
        <button type="button" @click="emit('createProject')">New project</button>
        <button
          v-if="project"
          type="button"
          class="primary-action"
          :disabled="loading || readyTimelineAssets.length !== 1"
          @click="emit('openEditor')"
        >{{ readyTimelineAssets.length > 1 ? 'Choose timeline below' : readyTimelineAssets.length ? 'Open editor' : 'Timeline unavailable' }}</button>
      </div>
    </header>

    <p v-if="loading && !project" class="dashboard-state" role="status">Loading project…</p>
    <div v-else-if="!project && !projects.length" class="dashboard-empty">
      <strong>Create the first project</strong>
      <p>A project keeps one canonical match, all source videos, moments and analysis jobs together.</p>
      <button type="button" @click="emit('createProject')">Create project</button>
    </div>
    <div v-else-if="!project" class="project-index">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Available workspaces</p>
          <h2>Choose a project</h2>
        </div>
        <span>{{ projects.length }}</span>
      </div>
      <div class="project-index-grid">
        <button
          v-for="item in projects"
          :key="item.id"
          type="button"
          class="project-index-card"
          @click="emit('selectProject', item.id)"
        >
          <span>
            <strong>{{ item.title }}</strong>
            <small>Updated {{ updatedLabel(item.updatedAt) }}</small>
          </span>
          <i>{{ item.matchId ? 'Match assigned' : 'No match assigned' }}</i>
        </button>
      </div>
    </div>

    <template v-else>
      <nav aria-label="Project sections">
        <button
          v-for="tab in (['overview', 'match', 'identities', 'analysis'] as const)"
          :key="tab"
          type="button"
          :class="{ active: activeTab === tab }"
          :aria-current="activeTab === tab ? 'page' : undefined"
          @click="emit('update:activeTab', tab)"
        >
          {{ tab === 'overview' ? 'Overview' : tab === 'match' ? 'Match' : tab === 'identities' ? 'Identities' : 'Analysis' }}
          <span v-if="tab === 'analysis' && activeJobs.length">{{ activeJobs.length }}</span>
        </button>
      </nav>

      <div v-if="activeTab === 'overview'" class="overview-tab">
        <div class="project-metrics" aria-label="Project summary">
          <article><strong>{{ assets.length }}</strong><span>source videos</span></article>
          <article><strong>{{ segments.length }}</strong><span>moments</span></article>
          <article><strong>{{ readySegments }}</strong><span>ready moments</span></article>
          <article><strong>{{ activeJobs.length }}</strong><span>active jobs</span></article>
        </div>

        <section class="overview-section" aria-labelledby="project-assets-title">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Inputs</p>
              <h2 id="project-assets-title">Source videos</h2>
            </div>
            <span>{{ assets.length }}</span>
          </div>
          <div v-if="assets.length" class="asset-grid">
            <article v-for="asset in assets" :key="asset.id">
              <img v-if="asset.posterUrl" :src="asset.posterUrl" alt="" />
              <div>
                <strong>{{ asset.filename }}</strong>
                <span>{{ durationLabel(asset.duration) }} · {{ asset.status }}</span>
                <button
                  type="button"
                  :disabled="!asset.timelineSceneId || asset.status !== 'ready'"
                  @click="emit('openTimeline', asset.id)"
                >{{ asset.timelineSceneId ? 'Open timeline' : 'Timeline pending' }}</button>
              </div>
            </article>
          </div>
          <p v-else class="section-empty">No source video has been added to this project.</p>
        </section>

        <section class="overview-section" aria-labelledby="project-moments-title">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Timeline</p>
              <h2 id="project-moments-title">Moments</h2>
            </div>
            <span>{{ segments.length }}</span>
          </div>
          <div v-if="segments.length" class="segment-list">
            <button
              v-for="segment in segments"
              :key="segment.id"
              type="button"
              :class="{ active: segment.id === project.activeSegmentId }"
              :aria-current="segment.id === project.activeSegmentId ? 'true' : undefined"
              @click="emit('selectSegment', segment.id)"
            >
              <span><strong>{{ segment.label }}</strong><small>{{ segmentRange(segment) }}</small></span>
              <i :class="`status-${segment.status}`">{{ segment.status }}</i>
            </button>
          </div>
          <p v-else class="section-empty">Moments will appear after a source video is processed.</p>
        </section>
      </div>

      <div v-else-if="activeTab === 'match'" class="match-workspace">
        <slot name="match-tools" />
        <ProjectMatchTab
          :match="match"
          :loading="loading"
          :busy="matchBusy"
          :allow-import="allowMatchImport"
          @refresh="emit('refreshMatch')"
          @import="emit('importMatch')"
        />
      </div>

      <ProjectIdentitiesPanel
        v-else-if="activeTab === 'identities'"
        :identities="identities"
        :loading="identitiesLoading"
        :error="identitiesError"
        :assigning-membership-id="assigningIdentityMembershipId"
        :assignment-error="identityAssignmentError"
        @refresh="emit('refreshIdentities')"
        @assign-membership="emit('assignIdentityMembership', $event)"
      />

      <AnalysisJobsPanel
        v-else
        :jobs="jobs"
        :loading="jobsLoading"
        :error="jobsError"
        :canceling-job-ids="cancelingJobIds"
        :last-updated-at="jobsUpdatedAt"
        @cancel="emit('cancelJob', $event)"
        @retry="emit('retryJobs')"
      />
    </template>
  </section>
</template>

<style scoped>
.project-dashboard {
  display: grid;
  gap: 18px;
  min-width: 0;
}

.match-workspace {
  display: grid;
  gap: 16px;
}

.primary-action {
  border-color: #4e83bc;
  background: #285a8d;
}

.dashboard-header,
.project-picker,
.section-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

h1,
h2,
p {
  margin: 0;
}

h1 {
  font-size: clamp(24px, 3vw, 34px);
}

h2 {
  font-size: 17px;
}

.dashboard-header > div:first-child > span,
.eyebrow,
.section-heading > span,
.asset-grid span,
.segment-list small {
  color: #8fa2b8;
  font-size: 12px;
}

.eyebrow {
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: .12em;
}

.project-picker label {
  color: #91a3b8;
  font-size: 11px;
  text-transform: uppercase;
}

select,
button {
  min-height: 38px;
  border: 1px solid #3b5068;
  border-radius: 8px;
  background: #152437;
  color: #e6f1fc;
}

select {
  min-width: 190px;
  padding: 0 11px;
}

button {
  padding: 0 13px;
  cursor: pointer;
}

button:disabled,
select:disabled {
  cursor: not-allowed;
  opacity: .5;
}

nav {
  display: flex;
  gap: 5px;
  padding-bottom: 10px;
  border-bottom: 1px solid #26384b;
}

nav button {
  border-color: transparent;
  background: transparent;
  color: #91a3b8;
}

nav button.active {
  border-color: #3a92ad;
  background: #143241;
  color: #86e7ff;
}

nav button span {
  display: inline-flex;
  min-width: 19px;
  min-height: 19px;
  align-items: center;
  justify-content: center;
  margin-left: 6px;
  border-radius: 50%;
  background: #2d5363;
  font-size: 10px;
}

.dashboard-empty,
.project-index,
.overview-section {
  padding: 17px;
  border: 1px solid #293b50;
  border-radius: 12px;
  background: #101a26;
}

.dashboard-empty {
  display: grid;
  justify-items: start;
  gap: 8px;
}

.dashboard-empty p,
.section-empty {
  color: #98aabd;
  font-size: 13px;
}

.project-index {
  display: grid;
  gap: 12px;
}

.project-index-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 10px;
}

.project-index-card {
  display: flex;
  min-height: 78px;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 14px;
  text-align: left;
}

.project-index-card:hover,
.project-index-card:focus-visible {
  border-color: #55c9e8;
  background: #173044;
}

.project-index-card > span {
  display: grid;
  min-width: 0;
  gap: 5px;
}

.project-index-card strong {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.project-index-card small,
.project-index-card i {
  color: #91a3b8;
  font-size: 11px;
}

.project-index-card i {
  flex: 0 0 auto;
  font-style: normal;
}

.overview-tab {
  display: grid;
  gap: 14px;
}

.project-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.project-metrics article {
  display: grid;
  gap: 4px;
  padding: 14px;
  border: 1px solid #293b50;
  border-radius: 10px;
  background: #101a26;
}

.project-metrics strong {
  font-size: 23px;
}

.project-metrics span {
  color: #91a3b8;
  font-size: 11px;
}

.asset-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 10px;
  margin-top: 12px;
}

.asset-grid article {
  display: grid;
  grid-template-columns: 74px minmax(0, 1fr);
  gap: 10px;
  align-items: center;
  overflow: hidden;
  border-radius: 9px;
  background: #162334;
}

.asset-grid img {
  width: 74px;
  height: 58px;
  object-fit: cover;
}

.asset-grid div {
  display: grid;
  min-width: 0;
  gap: 4px;
  padding: 9px 9px 9px 0;
}

.asset-grid strong {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.asset-grid button {
  min-height: 30px;
  justify-self: start;
  padding: 0 10px;
  border-color: #2d6f83;
  background: #123241;
  color: #8de7ff;
  font-size: 11px;
}

.segment-list {
  display: grid;
  gap: 7px;
  margin-top: 12px;
}

.segment-list button {
  display: flex;
  min-height: 52px;
  align-items: center;
  justify-content: space-between;
  text-align: left;
}

.segment-list button.active {
  border-color: #55c9e8;
  background: #153343;
}

.segment-list button > span {
  display: grid;
  gap: 3px;
}

.segment-list i {
  color: #90a4ba;
  font-size: 10px;
  font-style: normal;
  text-transform: uppercase;
}

.segment-list .status-ready {
  color: #79e2aa;
}

.segment-list .status-failed {
  color: #ff9e93;
}

@media (max-width: 760px) {
  .dashboard-header,
  .project-picker {
    align-items: stretch;
    flex-direction: column;
  }

  select {
    min-width: 0;
  }

  .project-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
