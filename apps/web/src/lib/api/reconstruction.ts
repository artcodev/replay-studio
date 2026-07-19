import { request } from './transport'
import { projectScenePath } from './paths'
import { sceneRequest } from './scenes'
import type { ModelComparisonQueue } from '../../types/analysis'
import type {
  BallDetectionBackend,
  BallDetectionProfile,
  BallTrajectoryMode,
  JerseyOcrProfile,
  ReconstructionModel,
} from '../../types/reconstruction'
import type { Keyframe } from '../../types/tracking'

export const reconstructionClient = {
  reconstruct: (
    projectId: string,
    sceneId: string,
    model: ReconstructionModel,
    ballBackend: BallDetectionBackend,
    ballDetectionProfile: BallDetectionProfile = 'automatic',
    jerseyOcrProfile: JerseyOcrProfile = 'automatic',
  ) => sceneRequest(projectId, projectScenePath(projectId, sceneId, '/reconstruct'), {
    method: 'POST',
    body: JSON.stringify({
      model,
      ball_backend: ballBackend,
      ball_detection_profile: ballDetectionProfile,
      jersey_ocr_profile: jerseyOcrProfile,
    }),
  }),
  updateBallTrajectory: (
    projectId: string,
    sceneId: string,
    mode: BallTrajectoryMode,
    keyframes?: Keyframe[],
  ) => sceneRequest(projectId, projectScenePath(projectId, sceneId, '/ball-trajectory'), {
    method: 'PUT',
    body: JSON.stringify({ mode, ...(keyframes === undefined ? {} : { keyframes }) }),
  }),
  compareModels: (projectId: string, sceneId: string) => request<ModelComparisonQueue>(
    projectScenePath(projectId, sceneId, '/compare-models'),
    { method: 'POST' },
  ),
}
