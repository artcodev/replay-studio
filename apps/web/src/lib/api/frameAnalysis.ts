import { request } from './transport'
import { projectScenePath } from './paths'
import type {
  FrameAnalysis,
  FrameAnnotationKind,
  FrameIdentityAction,
  FrameIdentityScope,
} from '../../types/analysis'

export type FrameAnnotationWrite = {
  annotationId?: string | null
  sceneTime: number
  bbox: { x: number; y: number; width: number; height: number }
  kind: FrameAnnotationKind
  label?: string | null
  externalPlayerId?: string | null
  action: FrameIdentityAction
  scope: FrameIdentityScope
  mergeTargetId?: string | null
  sourceTrackId?: string | null
  canonicalPersonId?: string | null
  targetObservationId?: string | null
  rangeStart?: number | null
  rangeEnd?: number | null
}

export const frameAnalysisClient = {
  analyze: (projectId: string, sceneId: string, sceneTime: number) => request<FrameAnalysis>(
    projectScenePath(projectId, sceneId, '/analyze-frame'),
    { method: 'POST', body: JSON.stringify({ scene_time: sceneTime }) },
  ),
  saveAnnotation: (projectId: string, sceneId: string, annotation: FrameAnnotationWrite) => request<FrameAnalysis>(
    projectScenePath(projectId, sceneId, '/frame-annotations'),
    {
      method: 'POST',
      body: JSON.stringify({
        annotation_id: annotation.annotationId,
        scene_time: annotation.sceneTime,
        bbox: annotation.bbox,
        kind: annotation.kind,
        label: annotation.label,
        external_player_id: annotation.externalPlayerId,
        action: annotation.action,
        scope: annotation.scope,
        merge_target_id: annotation.mergeTargetId,
        source_track_id: annotation.sourceTrackId,
        canonical_person_id: annotation.canonicalPersonId,
        target_observation_id: annotation.targetObservationId,
        range_start: annotation.rangeStart,
        range_end: annotation.rangeEnd,
      }),
    },
  ),
  deleteAnnotation: (projectId: string, sceneId: string, annotationId: string) => request<FrameAnalysis>(
    projectScenePath(projectId, sceneId, `/frame-annotations/${encodeURIComponent(annotationId)}`),
    { method: 'DELETE' },
  ),
}
