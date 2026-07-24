import { request } from './transport'
import { projectScenePath } from './paths'
import { sceneRequest } from './scenes'
import type { ModelComparisonQueue } from '../../types/analysis'
import type {
  BallDetectionBackend,
  BallDetectionProfile,
  BallTrajectoryMode,
  ContactPointProfile,
  JerseyOcrProfile,
  ReconstructionMode,
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
    contactPointProfile: ContactPointProfile = 'bbox-bottom',
    mode: ReconstructionMode = 'full',
    frameRate: number | null = null,
    directCalibrationMaxGapSeconds: number | null = null,
  ) => sceneRequest(projectId, projectScenePath(projectId, sceneId, '/reconstruct'), {
    method: 'POST',
    body: JSON.stringify({
      model,
      ball_backend: ballBackend,
      ball_detection_profile: ballDetectionProfile,
      jersey_ocr_profile: jerseyOcrProfile,
      contact_point_profile: contactPointProfile,
      mode,
      frame_rate: frameRate,
      direct_calibration_max_gap_seconds: directCalibrationMaxGapSeconds,
    }),
  }),
  confirmCalibrationReview: (
    projectId: string,
    sceneId: string,
  ) => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId, '/pitch-calibration/confirm-review'),
    { method: 'POST' },
  ),
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
