import { projectScenePath } from './paths'
import { sceneRequest } from './scenes'
import type { PlayerAction } from '../../types/playerActions'

export const playerActionClient = {
  upsert: (
    projectId: string,
    sceneId: string,
    action: Pick<PlayerAction, 'id' | 'canonicalPersonId' | 'type' | 'startTime' | 'endTime' | 'keypoints'>,
  ) => sceneRequest(projectId, projectScenePath(projectId, sceneId, '/player-actions'), {
    method: 'POST',
    body: JSON.stringify({
      id: action.id,
      canonicalPersonId: action.canonicalPersonId,
      type: action.type,
      startTime: action.startTime,
      endTime: action.endTime,
      keypoints: action.keypoints,
    }),
  }),
  remove: (projectId: string, sceneId: string, actionId: string) => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId, `/player-actions/${encodeURIComponent(actionId)}`),
    { method: 'DELETE' },
  ),
}
