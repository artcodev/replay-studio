import { request } from './transport'
import { projectScenePath, projectVideoPath } from './paths'
import type { ReconstructionSeriesWindow } from '../../types/reconstruction'
import type { SceneDocument, SceneSummary } from '../../types/scene'

const RECONSTRUCTION_SERIES_WINDOW_SECONDS = 30

function uniqueRows<T>(rows: T[], key: (row: T) => string): T[] {
  const seen = new Set<string>()
  return rows.filter((row) => {
    const value = key(row)
    if (seen.has(value)) return false
    seen.add(value)
    return true
  })
}

async function hydrateScene(projectId: string, scene: SceneDocument): Promise<SceneDocument> {
  const video = scene.payload?.videoAsset
  if (video?.id) {
    video.mediaUrl = projectVideoPath(projectId, video.id, '/media')
    video.posterUrl = projectVideoPath(projectId, video.id, '/poster')
  }
  if (!scene.payload || !Array.isArray(scene.payload.tracks) || !scene.payload.ball) {
    return scene
  }
  const reconstruction = scene.payload.videoAsset?.reconstruction
  const artifacts = reconstruction?.artifactManifest?.artifacts

  // Dense values from a Scene response are never trusted. Artifacts are the
  // only playback/review source and every editor collection is rebuilt here.
  scene.payload.tracks.forEach((track) => {
    track.keyframes = []
    track.observations = []
  })
  scene.payload.canonicalPeople?.forEach((person) => { person.observations = [] })
  scene.payload.ball.keyframes = []
  scene.payload.ball.automaticKeyframes = []
  scene.payload.ball.manualKeyframes = []
  if (reconstruction?.calibration) reconstruction.calibration.frameEvidence = []
  if (reconstruction?.ballDetection) reconstruction.ballDetection.frames = []

  const hasDenseArtifacts = Boolean(
    artifacts?.identityTimeline
    || artifacts?.ballTrajectory
    || artifacts?.calibrationFrames,
  )
  if (!hasDenseArtifacts) return scene

  const requests: Array<Promise<ReconstructionSeriesWindow>> = []
  for (let start = 0; start < scene.duration; start += RECONSTRUCTION_SERIES_WINDOW_SECONDS) {
    const end = Math.min(scene.duration, start + RECONSTRUCTION_SERIES_WINDOW_SECONDS)
    const parameters = new URLSearchParams({ start: String(start), end: String(end) })
    requests.push(request<ReconstructionSeriesWindow>(
      `${projectScenePath(projectId, scene.id, '/reconstruction-series')}?${parameters.toString()}`,
    ))
  }
  const windows = await Promise.all(requests)
  const trackWindows = new Map<string, ReconstructionSeriesWindow['tracks']>()
  const personWindows = new Map<string, ReconstructionSeriesWindow['canonicalPeople']>()
  for (const window of windows) {
    for (const track of window.tracks) {
      const rows = trackWindows.get(track.id) ?? []
      rows.push(track)
      trackWindows.set(track.id, rows)
    }
    for (const person of window.canonicalPeople) {
      const rows = personWindows.get(person.canonicalPersonId) ?? []
      rows.push(person)
      personWindows.set(person.canonicalPersonId, rows)
    }
  }
  for (const track of scene.payload.tracks) {
    const rows = trackWindows.get(track.id) ?? []
    track.keyframes = uniqueRows(
      rows.flatMap((row) => row.keyframes),
      (row) => `${row.id ?? ''}:${row.t}:${row.x}:${row.z}`,
    ).sort((left, right) => left.t - right.t)
    track.observations = uniqueRows(
      rows.flatMap((row) => row.observations),
      (row) => `${row.observationId ?? row.id ?? ''}:${row.frameIndex}:${row.sceneTime}`,
    ).sort((left, right) => left.sceneTime - right.sceneTime)
  }
  for (const person of scene.payload.canonicalPeople ?? []) {
    const rows = personWindows.get(person.canonicalPersonId) ?? []
    person.observations = uniqueRows(
      rows.flatMap((row) => row.observations),
      (row) => `${row.observationId ?? row.id ?? ''}:${row.frameIndex}:${row.sceneTime}`,
    ).sort((left, right) => left.sceneTime - right.sceneTime)
  }
  const mergeKeyframes = (field: keyof ReconstructionSeriesWindow['ball']) => uniqueRows(
    windows.flatMap((window) => window.ball[field]),
    (row) => `${row.id ?? ''}:${row.t}:${row.x}:${row.z}`,
  ).sort((left, right) => left.t - right.t)
  scene.payload.ball.keyframes = mergeKeyframes('keyframes')
  scene.payload.ball.automaticKeyframes = mergeKeyframes('automaticKeyframes')
  scene.payload.ball.manualKeyframes = mergeKeyframes('manualKeyframes')
  if (reconstruction?.calibration) {
    reconstruction.calibration.frameEvidence = uniqueRows(
      windows.flatMap((window) => window.calibration.frameEvidence),
      (row) => `${row.sourceFrameIndex}:${row.sceneTime}`,
    ).sort((left, right) => left.sceneTime - right.sceneTime)
  }
  if (reconstruction?.ballDetection) {
    reconstruction.ballDetection.frames = uniqueRows(
      windows.flatMap((window) => window.ballDetection.frames),
      (row) => `${String(row.frameIndex ?? '')}:${String(row.t ?? '')}`,
    )
  }
  return scene
}

export async function sceneRequest(
  projectId: string,
  path: string,
  init?: RequestInit,
): Promise<SceneDocument> {
  return hydrateScene(projectId, await request<SceneDocument>(path, init))
}

export const sceneClient = {
  list: (projectId: string, signal?: AbortSignal) => request<SceneSummary[]>(
    `/api/projects/${encodeURIComponent(projectId)}/scenes`,
    { signal },
  ),
  get: (projectId: string, sceneId: string) => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId),
  ),
  exactAnalysisFrameUrl: (
    projectId: string,
    sceneId: string,
    generationKey: string,
    sourceFrameIndex: number,
  ) => projectScenePath(
    projectId,
    sceneId,
    `/analysis-frame-generations/${encodeURIComponent(generationKey)}/frames/${sourceFrameIndex}`,
  ),
  setFrameExcluded: (
    projectId: string,
    sceneId: string,
    sourceFrameIndex: number,
    excluded: boolean,
  ) => sceneRequest(
    projectId,
    projectScenePath(
      projectId,
      sceneId,
      `/frame-exclusions/${sourceFrameIndex}`,
    ),
    { method: 'PUT', body: JSON.stringify({ excluded }) },
  ),
  // Every editor write is a dedicated command carrying only its own domain.
  // No client round-trips a scene document, so the revision fence never
  // arbitrates between the UI and the reconstruction runner.
  saveTitle: (projectId: string, sceneId: string, title: string) => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId, '/title'),
    { method: 'PUT', body: JSON.stringify({ title }) },
  ),
  saveEventBindings: (
    projectId: string,
    sceneId: string,
    bindings: SceneDocument['payload']['eventBindings'],
  ) => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId, '/event-bindings'),
    { method: 'PUT', body: JSON.stringify({ bindings }) },
  ),
  saveTrackMetadata: (
    projectId: string,
    sceneId: string,
    trackId: string,
    metadata: { label?: string; number?: number },
  ) => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId, `/tracks/${encodeURIComponent(trackId)}/metadata`),
    { method: 'PUT', body: JSON.stringify(metadata) },
  ),
  saveTrackTrajectory: (
    projectId: string,
    sceneId: string,
    trackId: string,
    keyframes: Array<{ t: number; x: number; z: number }>,
  ) => sceneRequest(
    projectId,
    projectScenePath(projectId, sceneId, `/tracks/${encodeURIComponent(trackId)}/trajectory`),
    { method: 'PUT', body: JSON.stringify({ keyframes }) },
  ),
  // Event grouping goes through its own endpoint and carries only the
  // layout — never a whole scene document. The server applies it onto the
  // current stored scene, so an editor holding an older revision still
  // saves instead of dead-ending on the document revision fence.
  saveSegmentLayout: (projectId: string, scene: SceneDocument) => sceneRequest(
    projectId,
    projectScenePath(projectId, scene.id, '/segment-layout'),
    {
      method: 'PUT',
      body: JSON.stringify({
        status: scene.payload?.videoAsset?.segmentLayout?.status ?? 'edited',
        segments: (scene.payload?.videoAsset?.segments ?? [])
          .filter((segment) => segment.layout)
          .map((segment) => ({
            id: segment.id,
            group: segment.layout!.group,
            variant: segment.layout!.variant,
            label: segment.layout!.label,
            role: segment.layout!.role,
            confidence: segment.layout!.confidence,
            motionCost: segment.layout!.motionCost,
          })),
      }),
    },
  ),
}
