import type { BallDetectionBackend, BallTrajectoryMode, EventBundle, ExternalEvent, FrameAnalysis, FrameAnnotationKind, FrameIdentityAction, FrameIdentityScope, IdentityReviewResponse, Keyframe, ManualMatchImportRequest, MatchDataProviderCatalog, MatchDataProviderId, ModelComparisonReport, PitchCalibrationAnchor, PitchCalibrationDraft, PitchCalibrationPreset, PlayerAction, ReconstructionModel, SceneDocument, SceneMatchBindingResponse, SceneSummary, VideoAsset } from '../types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(requestErrorMessage(body, response.status))
  }
  return response.json() as Promise<T>
}

function requestErrorMessage(body: unknown, status: number): string {
  const detail = body && typeof body === 'object' && 'detail' in body
    ? (body as { detail?: unknown }).detail
    : null
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (!item || typeof item !== 'object') return null
      const row = item as { loc?: unknown; msg?: unknown }
      if (typeof row.msg !== 'string') return null
      const location = Array.isArray(row.loc)
        ? row.loc.filter((part) => part !== 'body').map(String).join('.')
        : ''
      return location ? `${location}: ${row.msg}` : row.msg
    }).filter((message): message is string => Boolean(message))
    if (messages.length) return messages.join('; ')
  }
  return `Request failed (${status})`
}

export const api = {
  listScenes: () => request<SceneSummary[]>('/api/scenes'),
  getScene: (id: string) => request<SceneDocument>(`/api/scenes/${id}`),
  saveScene: (scene: SceneDocument) =>
    request<SceneDocument>(`/api/scenes/${scene.id}`, {
      method: 'PUT',
      body: JSON.stringify(scene),
    }),
  createScene: (eventId?: string, title?: string) =>
    request<SceneDocument>('/api/scenes', {
      method: 'POST',
      body: JSON.stringify({ event_id: eventId, title }),
    }),
  matchDataProviders: () => request<MatchDataProviderCatalog>('/api/catalog/providers'),
  eventsByDate: (date: string, provider?: MatchDataProviderId) =>
    request<ExternalEvent[]>(
      `/api/catalog/events?date=${encodeURIComponent(date)}${provider ? `&provider=${encodeURIComponent(provider)}` : ''}`,
    ),
  searchEvents: (query: string, provider?: MatchDataProviderId) =>
    request<ExternalEvent[]>(
      `/api/catalog/events/search?q=${encodeURIComponent(query)}${provider ? `&provider=${encodeURIComponent(provider)}` : ''}`,
    ),
  eventBundle: (id: string, provider?: MatchDataProviderId) => request<EventBundle>(
    `/api/catalog/events/${encodeURIComponent(id)}${provider ? `?provider=${encodeURIComponent(provider)}` : ''}`,
  ),
  bindSceneMatch: (sceneId: string, eventId: string, provider?: MatchDataProviderId) =>
    request<SceneMatchBindingResponse>(`/api/scenes/${encodeURIComponent(sceneId)}/match-binding`, {
      method: 'POST',
      body: JSON.stringify({ event_id: eventId, ...(provider ? { provider } : {}) }),
    }),
  refreshSceneMatchBinding: (sceneId: string) => request<SceneMatchBindingResponse>(
    `/api/scenes/${encodeURIComponent(sceneId)}/match-binding/refresh`,
    { method: 'POST' },
  ),
  importSceneMatchBinding: (sceneId: string, payload: ManualMatchImportRequest) =>
    request<SceneMatchBindingResponse>(
      `/api/scenes/${encodeURIComponent(sceneId)}/match-binding/import`,
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
    ),
  identityReview: (sceneId: string) => request<IdentityReviewResponse>(
    `/api/scenes/${encodeURIComponent(sceneId)}/identity-review`,
  ),
  video: (id: string) => request<VideoAsset>(`/api/videos/${id}`),
  listVideos: () => request<VideoAsset[]>('/api/videos'),
  uploadVideo: (file: File, title: string, onProgress: (progress: number) => void) =>
    new Promise<VideoAsset>((resolve, reject) => {
      const form = new FormData()
      form.append('file', file)
      if (title.trim()) form.append('title', title.trim())
      const xhr = new XMLHttpRequest()
      xhr.open('POST', '/api/videos')
      xhr.responseType = 'json'
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100))
      })
      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as VideoAsset)
        else reject(new Error(xhr.response?.detail ?? `Upload failed (${xhr.status})`))
      })
      xhr.addEventListener('error', () => reject(new Error('Video upload failed')))
      xhr.send(form)
    }),
  createSegmentScene: (assetId: string, segmentId: string) =>
    request<SceneDocument>(`/api/videos/${assetId}/segments/${segmentId}/scene`, { method: 'POST' }),
  proposeSegmentLayout: (assetId: string) =>
    request<SceneDocument>(`/api/videos/${assetId}/segment-layout/propose`, { method: 'POST' }),
  createMultiPass: (assetId: string, segmentIds: string[]) =>
    request<SceneDocument>(`/api/videos/${assetId}/multi-pass`, {
      method: 'POST',
      body: JSON.stringify({ segment_ids: segmentIds }),
    }),
  reconstructScene: (sceneId: string, model: ReconstructionModel, ballBackend: BallDetectionBackend) =>
    request<SceneDocument>(`/api/scenes/${sceneId}/reconstruct`, {
      method: 'POST',
      body: JSON.stringify({ model, ball_backend: ballBackend }),
    }),
  updateBallTrajectory: (
    sceneId: string,
    mode: BallTrajectoryMode,
    keyframes?: Keyframe[],
  ) => request<SceneDocument>(`/api/scenes/${sceneId}/ball-trajectory`, {
    method: 'PUT',
    body: JSON.stringify({ mode, ...(keyframes === undefined ? {} : { keyframes }) }),
  }),
  upsertPlayerAction: (
    sceneId: string,
    action: Pick<PlayerAction, 'id' | 'canonicalPersonId' | 'type' | 'startTime' | 'endTime' | 'keypoints'>,
  ) => request<SceneDocument>(`/api/scenes/${encodeURIComponent(sceneId)}/player-actions`, {
    method: 'POST',
    // Provenance/review fields are server-owned. Constructing the request
    // explicitly also prevents a full PlayerAction object from leaking extra
    // properties through TypeScript's structural typing at runtime.
    body: JSON.stringify({
      id: action.id,
      canonicalPersonId: action.canonicalPersonId,
      type: action.type,
      startTime: action.startTime,
      endTime: action.endTime,
      keypoints: action.keypoints,
    }),
  }),
  deletePlayerAction: (sceneId: string, actionId: string) => request<SceneDocument>(
    `/api/scenes/${encodeURIComponent(sceneId)}/player-actions/${encodeURIComponent(actionId)}`,
    { method: 'DELETE' },
  ),
  analyzeFrame: (sceneId: string, sceneTime: number) =>
    request<FrameAnalysis>(`/api/scenes/${sceneId}/analyze-frame`, {
      method: 'POST',
      body: JSON.stringify({ scene_time: sceneTime }),
    }),
  saveFrameAnnotation: (
    sceneId: string,
    annotation: {
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
    },
  ) => request<FrameAnalysis>(`/api/scenes/${sceneId}/frame-annotations`, {
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
  }),
  deleteFrameAnnotation: (sceneId: string, annotationId: string) =>
    request<FrameAnalysis>(`/api/scenes/${sceneId}/frame-annotations/${encodeURIComponent(annotationId)}`, {
      method: 'DELETE',
    }),
  updateCanonicalRosterBinding: (
    sceneId: string,
    canonicalPersonId: string,
    externalPlayerId: string | null,
  ) => request<SceneDocument>(
    `/api/scenes/${encodeURIComponent(sceneId)}/canonical-people/${encodeURIComponent(canonicalPersonId)}/roster-binding`,
    {
      method: 'PUT',
      body: JSON.stringify({ external_player_id: externalPlayerId }),
    },
  ),
  clearCanonicalRosterBinding: (
    sceneId: string,
    canonicalPersonId: string,
  ) => request<SceneDocument>(
    `/api/scenes/${encodeURIComponent(sceneId)}/canonical-people/${encodeURIComponent(canonicalPersonId)}/roster-binding`,
    { method: 'DELETE' },
  ),
  rejectRosterCandidate: (
    sceneId: string,
    canonicalPersonId: string,
    externalPlayerId: string,
  ) => request<SceneDocument>(
    `/api/scenes/${encodeURIComponent(sceneId)}/canonical-people/${encodeURIComponent(canonicalPersonId)}/roster-rejections`,
    {
      method: 'POST',
      body: JSON.stringify({ external_player_id: externalPlayerId }),
    },
  ),
  clearRosterCandidateRejection: (
    sceneId: string,
    canonicalPersonId: string,
    externalPlayerId: string,
  ) => request<SceneDocument>(
    `/api/scenes/${encodeURIComponent(sceneId)}/canonical-people/${encodeURIComponent(canonicalPersonId)}/roster-rejections/${encodeURIComponent(externalPlayerId)}`,
    { method: 'DELETE' },
  ),
  compareModels: (sceneId: string) =>
    request<ModelComparisonReport>(`/api/scenes/${sceneId}/compare-models`, { method: 'POST' }),
  autoPitchCalibration: (sceneId: string, sceneTime: number, preset?: PitchCalibrationPreset) =>
    request<PitchCalibrationDraft>(`/api/scenes/${sceneId}/pitch-calibration/auto`, {
      method: 'POST',
      body: JSON.stringify({ scene_time: sceneTime, preset }),
    }),
  previewPitchCalibration: (
    sceneId: string,
    sceneTime: number,
    preset: PitchCalibrationPreset,
    anchors: PitchCalibrationAnchor[],
  ) => request<PitchCalibrationDraft>(`/api/scenes/${sceneId}/pitch-calibration/preview`, {
    method: 'POST',
    body: JSON.stringify({ scene_time: sceneTime, preset, anchors }),
  }),
  applyPitchCalibration: (
    sceneId: string,
    sceneTime: number,
    preset: PitchCalibrationPreset,
    anchors: PitchCalibrationAnchor[],
  ) => request<SceneDocument>(`/api/scenes/${sceneId}/pitch-calibration/apply`, {
    method: 'POST',
    body: JSON.stringify({ scene_time: sceneTime, preset, anchors }),
  }),
  setAttackingGoal: (sceneId: string, side: 'left' | 'right') =>
    request<SceneDocument>(`/api/scenes/${sceneId}/pitch-side`, {
      method: 'POST',
      body: JSON.stringify({ side }),
    }),
}
