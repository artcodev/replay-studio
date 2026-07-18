import { request } from './transport'
import { projectScenePath } from './paths'
import { sceneRequest } from './scenes'
import type { IdentityReviewResponse } from '../../types/identityReview'

export const identityClient = {
  review: (projectId: string, sceneId: string) => request<IdentityReviewResponse>(
    projectScenePath(projectId, sceneId, '/identity-review'),
  ),
  updateRosterBinding: (
    projectId: string,
    sceneId: string,
    canonicalPersonId: string,
    externalPlayerId: string | null,
  ) => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId, `/canonical-people/${encodeURIComponent(canonicalPersonId)}/roster-binding`),
    { method: 'PUT', body: JSON.stringify({ external_player_id: externalPlayerId }) },
  ),
  clearRosterBinding: (projectId: string, sceneId: string, canonicalPersonId: string) => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId, `/canonical-people/${encodeURIComponent(canonicalPersonId)}/roster-binding`),
    { method: 'DELETE' },
  ),
  rejectRosterCandidate: (
    projectId: string,
    sceneId: string,
    canonicalPersonId: string,
    externalPlayerId: string,
  ) => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId, `/canonical-people/${encodeURIComponent(canonicalPersonId)}/roster-rejections`),
    { method: 'POST', body: JSON.stringify({ external_player_id: externalPlayerId }) },
  ),
  clearRosterCandidateRejection: (
    projectId: string,
    sceneId: string,
    canonicalPersonId: string,
    externalPlayerId: string,
  ) => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId, `/canonical-people/${encodeURIComponent(canonicalPersonId)}/roster-rejections/${encodeURIComponent(externalPlayerId)}`),
    { method: 'DELETE' },
  ),
}
