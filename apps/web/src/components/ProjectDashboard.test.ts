import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import type { CanonicalMatch } from '../types/match'
import type { AnalysisJob, Project, ProjectAsset, ProjectIdentity, ProjectSegment } from '../types/project'
import ProjectDashboard from './ProjectDashboard.vue'

const project: Project = {
  id: 'project-1',
  title: 'Spain vs Belgium highlights',
  revision: 4,
  matchId: 'match-1',
  activeSegmentId: 'moment-2',
  createdAt: '2026-07-17T08:00:00Z',
  updatedAt: '2026-07-17T10:00:00Z',
}

const assets: ProjectAsset[] = [{
  id: 'asset-1',
  projectId: project.id,
  filename: 'highlights.mp4',
  timelineSceneId: 'video-root-1',
  duration: 54.5,
  status: 'ready',
  posterUrl: '/media/poster.jpg',
  createdAt: project.createdAt,
}]

const segments: ProjectSegment[] = [
  { id: 'moment-1', projectId: project.id, assetId: 'asset-1', sourceSegmentId: 'shot-1', label: '1-A', start: 0, end: 4.2, status: 'ready' },
  { id: 'moment-2', projectId: project.id, assetId: 'asset-1', sourceSegmentId: 'shot-2', label: '2-A', start: 6.1, end: 11.4, status: 'analyzing' },
]

const match: CanonicalMatch = {
  id: 'match-1',
  revision: 1,
  snapshotId: 'snapshot-1',
  snapshotHash: 'sha256:one',
  name: 'Spain vs Belgium',
  score: { home: 2, away: 1 },
  homeTeam: { id: 'home', name: 'Spain' },
  awayTeam: { id: 'away', name: 'Belgium' },
  roster: [],
  events: [],
  substitutions: [],
  sync: { state: 'manual', syncedAt: null, stale: false, warnings: [] },
}

const jobs: AnalysisJob[] = [{
  id: 'run-1',
  projectId: project.id,
  segmentId: 'moment-2',
  kind: 'reconstruction',
  status: 'running',
  phase: 'tracking',
  progress: {
    completed: 20,
    total: 100,
    percent: 20,
    label: 'Tracking players',
    detail: null,
    etaSeconds: 30,
  },
  createdAt: project.updatedAt,
}]

const identities: ProjectIdentity[] = [{
  id: 'person-1',
  projectId: project.id,
  displayName: 'Anonymous player',
  status: 'active',
  memberships: [{
    id: 'membership-1',
    projectId: project.id,
    projectPersonId: 'person-1',
    sceneId: 'scene-1',
    scenePersonId: 'canonical-1',
    assignmentSource: 'scene-local',
    observationCount: 5,
  }],
}]

describe('ProjectDashboard', () => {
  it('renders project-level assets, moments, active selection and compute summary', async () => {
    const html = await renderToString(createSSRApp(ProjectDashboard, {
      projects: [project],
      project,
      assets,
      segments,
      jobs,
    }))

    expect(html).toContain('Spain vs Belgium highlights')
    expect(html).toMatch(/>1<\/strong><span[^>]*>source videos<\/span>/)
    expect(html).toMatch(/>2<\/strong><span[^>]*>moments<\/span>/)
    expect(html).toMatch(/>1<\/strong><span[^>]*>active jobs<\/span>/)
    expect(html).toContain('highlights.mp4')
    expect(html).toContain('0:55 · ready')
    expect(html).toContain('>Open timeline</button>')
    expect(html).toContain('>1-A<')
    expect(html).toContain('>2-A<')
    expect(html).toContain('aria-current="true"')
  })

  it('delegates the canonical match and compact analysis views without provider controls', async () => {
    const matchHtml = await renderToString(createSSRApp(ProjectDashboard, {
      projects: [project],
      project,
      match,
      activeTab: 'match',
    }))
    const jobsHtml = await renderToString(createSSRApp(ProjectDashboard, {
      projects: [project],
      project,
      jobs,
      activeTab: 'analysis',
    }))

    expect(matchHtml).toContain('Manual match data')
    expect(matchHtml).toContain('Spain vs Belgium')
    expect(matchHtml).not.toContain('provider')
    expect(jobsHtml).toContain('Analysis jobs')
    expect(jobsHtml).toContain('Tracking players')
    expect(jobsHtml).toContain('aria-label="Cancel Reconstruct moment"')

    const identitiesHtml = await renderToString(createSSRApp(ProjectDashboard, {
      projects: [project],
      project,
      identities,
      activeTab: 'identities',
    }))
    expect(identitiesHtml).toContain('Project identities')
    expect(identitiesHtml).toContain('Anonymous player')
  })

  it('renders an intentional first-project empty state', async () => {
    const html = await renderToString(createSSRApp(ProjectDashboard))

    expect(html).toContain('Create the first project')
    expect(html).toContain('one canonical match, all source videos, moments and analysis jobs')
    expect(html).toContain('>Create project</button>')
  })

  it('renders existing projects as a neutral list until one is explicitly opened', async () => {
    const emptyProject = { ...project, id: 'project-empty', matchId: null, title: 'Empty import' }
    const html = await renderToString(createSSRApp(ProjectDashboard, {
      projects: [emptyProject, project],
      project: null,
    }))

    expect(html).toMatch(/<h1[^>]*id="project-dashboard-title"[^>]*>Projects<\/h1>/)
    expect(html).toContain('Choose a project…')
    expect(html).toMatch(/<h2[^>]*>Choose a project<\/h2>/)
    expect(html).toContain('Empty import')
    expect(html).toContain('Spain vs Belgium highlights')
    expect(html).toContain('No match assigned')
    expect(html).toContain('Match assigned')
    expect(html).not.toContain('aria-current="page"')
    expect(html).not.toContain('Open editor')
  })

  it('makes projects without match details explicit in the project picker', async () => {
    const emptyProject = { ...project, id: 'project-empty', matchId: null, title: 'Empty import' }
    const html = await renderToString(createSSRApp(ProjectDashboard, {
      projects: [emptyProject, project],
      project: emptyProject,
    }))

    expect(html).toContain('Empty import · no match assigned')
    expect(html).toContain('Spain vs Belgium highlights</option>')
    expect(html).toMatch(/<option[^>]*value=""[^>]*disabled/)
  })

  it('requires an explicit video choice when a project has several timelines', async () => {
    const html = await renderToString(createSSRApp(ProjectDashboard, {
      projects: [project],
      project,
      assets: [
        assets[0],
        { ...assets[0], id: 'asset-2', timelineSceneId: 'video-root-2', filename: 'angle-2.mp4' },
      ],
    }))

    expect(html).toMatch(/<button[^>]*disabled[^>]*>Choose timeline below<\/button>/)
    expect(html.match(/>Open timeline<\/button>/g)).toHaveLength(2)
  })

  it('publishes the complete integration event contract', () => {
    const events = (ProjectDashboard as unknown as { emits: string[] }).emits
    expect(events).toEqual(expect.arrayContaining([
      'update:activeTab',
      'selectProject',
      'createProject',
      'openEditor',
      'openTimeline',
      'selectSegment',
      'refreshMatch',
      'importMatch',
      'cancelJob',
      'retryJobs',
      'refreshIdentities',
      'assignIdentityMembership',
    ]))
    expect(events).toHaveLength(12)
  })
})
