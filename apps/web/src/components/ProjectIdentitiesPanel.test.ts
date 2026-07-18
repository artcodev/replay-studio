import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { describe, expect, it } from 'vitest'
import type { ProjectIdentity } from '../types/project'
import ProjectIdentitiesPanel from './ProjectIdentitiesPanel.vue'

const identities: ProjectIdentity[] = [{
  id: 'person-8',
  projectId: 'project-1',
  rosterPersonId: 'roster-8',
  displayName: 'Home Eight',
  teamId: 'team-home',
  role: 'Midfielder',
  jerseyNumber: '8',
  status: 'active',
  identityConfidence: 0.91,
  memberships: [
    {
      id: 'membership-a',
      projectId: 'project-1',
      projectPersonId: 'person-8',
      sceneId: 'scene-1-a',
      scenePersonId: 'canonical-a',
      assignmentSource: 'accepted-roster',
      observationCount: 12,
    },
    {
      id: 'membership-b',
      projectId: 'project-1',
      projectPersonId: 'person-8',
      sceneId: 'scene-1-b',
      scenePersonId: 'canonical-b',
      assignmentSource: 'explicit',
      observationCount: 8,
    },
  ],
}, {
  id: 'person-10',
  projectId: 'project-1',
  displayName: 'Away Ten',
  teamId: 'team-away',
  role: 'Forward',
  jerseyNumber: '10',
  status: 'active',
  memberships: [{
    id: 'membership-c',
    projectId: 'project-1',
    projectPersonId: 'person-10',
    sceneId: 'scene-2-a',
    scenePersonId: 'canonical-c',
    assignmentSource: 'scene-local',
    observationCount: 6,
  }],
}]

describe('ProjectIdentitiesPanel', () => {
  it('renders project identities and every scene membership', async () => {
    const html = await renderToString(createSSRApp(ProjectIdentitiesPanel, { identities }))

    expect(html).toContain('Project identities')
    expect(html).toContain('#8 · Home Eight')
    expect(html).toContain('Canonical roster identity')
    expect(html).toContain('scene-1-a')
    expect(html).toContain('canonical-b')
    expect(html).toContain('accepted-roster')
    expect(html).toContain('explicit')
    expect(html).toContain('91%')
    expect(html).toContain('Assign to project person')
    expect(html).toContain('#10 · Away Ten')
    expect(html).toContain('Assign canonical-a from Home Eight to project person')
    expect(html).toMatch(/<button[^>]*disabled[^>]*aria-label="Assign canonical-a to Home Eight"/)
  })

  it('publishes assignment events and visible busy and error states', async () => {
    const busyHtml = await renderToString(createSSRApp(ProjectIdentitiesPanel, {
      identities,
      assigningMembershipId: 'membership-a',
    }))
    const errorHtml = await renderToString(createSSRApp(ProjectIdentitiesPanel, {
      identities,
      assignmentError: 'Membership changed concurrently',
    }))
    const events = (ProjectIdentitiesPanel as unknown as { emits: string[] }).emits

    expect(events).toEqual(expect.arrayContaining(['refresh', 'assignMembership']))
    expect(events).toHaveLength(2)
    expect(busyHtml).toContain('Assigning…')
    expect(busyHtml).toMatch(/<select[^>]*disabled/)
    expect(errorHtml).toContain('role="alert"')
    expect(errorHtml).toContain('Membership changed concurrently')
  })

  it('explains the empty state without leaking provider concepts', async () => {
    const html = await renderToString(createSSRApp(ProjectIdentitiesPanel))

    expect(html).toContain('No identities have been published yet')
    expect(html.toLowerCase()).not.toContain('provider')
  })
})
