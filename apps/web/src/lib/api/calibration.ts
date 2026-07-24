import { request } from './transport'
import { projectScenePath } from './paths'
import { sceneRequest } from './scenes'
import type {
  PitchCalibrationAnchor,
  CalibrationBorrowSource,
  CalibrationDraftSource,
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
  borrow: (
    projectId: string,
    sceneId: string,
    sceneTime: number,
    source: CalibrationBorrowSource,
    preset?: PitchCalibrationPreset,
  ) => request<PitchCalibrationDraft>(
    projectScenePath(projectId, sceneId, '/pitch-calibration/borrow'),
    {
      method: 'POST',
      body: JSON.stringify({
        scene_time: sceneTime,
        source,
        preset,
      }),
    },
  ),
  saveDraft: (
    projectId: string,
    sceneId: string,
    sceneTime: number,
    preset: PitchCalibrationPreset,
    anchors: PitchCalibrationAnchor[],
    source: CalibrationDraftSource,
    acceptQualityWarning = false,
  ) => sceneRequest(projectId, projectScenePath(projectId, sceneId, '/pitch-calibration/drafts'), {
    method: 'POST',
    body: JSON.stringify({
      scene_time: sceneTime,
      preset,
      anchors,
      source,
      accept_quality_warning: acceptQualityWarning,
    }),
  }),
  finalizeDrafts: (
    projectId: string,
    sceneId: string,
  ) => sceneRequest(projectId, projectScenePath(projectId, sceneId, '/pitch-calibration/finalize'), {
    method: 'POST',
  }),
  resetCalibration: (
    projectId: string,
    sceneId: string,
  ) => sceneRequest(projectId, projectScenePath(projectId, sceneId, '/pitch-calibration/reset'), {
    method: 'POST',
  }),
  setAttackingGoal: (projectId: string, sceneId: string, side: 'left' | 'right') => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId, '/pitch-side'),
    { method: 'POST', body: JSON.stringify({ side }) },
  ),
}
