import { request } from './transport'
import { projectScenePath } from './paths'
import { sceneRequest } from './scenes'
import type {
  PitchCalibrationAnchor,
  PitchCalibrationDraft,
  PitchCalibrationPreset,
} from '../../types/calibration'

export const calibrationClient = {
  auto: (projectId: string, sceneId: string, sceneTime: number, preset?: PitchCalibrationPreset) => request<PitchCalibrationDraft>(
    projectScenePath(projectId, sceneId, '/pitch-calibration/auto'),
    { method: 'POST', body: JSON.stringify({ scene_time: sceneTime, preset }) },
  ),
  preview: (
    projectId: string,
    sceneId: string,
    sceneTime: number,
    preset: PitchCalibrationPreset,
    anchors: PitchCalibrationAnchor[],
  ) => request<PitchCalibrationDraft>(
    projectScenePath(projectId, sceneId, '/pitch-calibration/preview'),
    { method: 'POST', body: JSON.stringify({ scene_time: sceneTime, preset, anchors }) },
  ),
  apply: (
    projectId: string,
    sceneId: string,
    sceneTime: number,
    preset: PitchCalibrationPreset,
    anchors: PitchCalibrationAnchor[],
  ) => sceneRequest(projectId, projectScenePath(projectId, sceneId, '/pitch-calibration/apply'), {
    method: 'POST',
    body: JSON.stringify({ scene_time: sceneTime, preset, anchors }),
  }),
  setAttackingGoal: (projectId: string, sceneId: string, side: 'left' | 'right') => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId, '/pitch-side'),
    { method: 'POST', body: JSON.stringify({ side }) },
  ),
}
