import { request } from './transport'
import { projectPath, projectVideoPath } from './paths'
import { sceneRequest } from './scenes'
import type { VideoAsset } from '../../types/media'

export const mediaClient = {
  get: (projectId: string, assetId: string) => request<VideoAsset>(
    projectVideoPath(projectId, assetId),
  ),
  upload: (
    projectId: string,
    file: File,
    title: string,
    onProgress: (progress: number) => void,
  ) => new Promise<VideoAsset>((resolve, reject) => {
    const form = new FormData()
    form.append('file', file)
    if (title.trim()) form.append('title', title.trim())
    const xhr = new XMLHttpRequest()
    xhr.open('POST', projectPath(projectId, '/videos'))
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
  createSegmentScene: (projectId: string, assetId: string, segmentId: string) => sceneRequest(
    projectId,
    projectVideoPath(projectId, assetId, `/segments/${encodeURIComponent(segmentId)}/scene`),
    { method: 'POST' },
  ),
  proposeSegmentLayout: (projectId: string, assetId: string) => sceneRequest(
    projectId,
    projectVideoPath(projectId, assetId, '/segment-layout/propose'),
    { method: 'POST' },
  ),
  createComposition: (projectId: string, segmentIds: string[]) => sceneRequest(
    projectId,
    projectPath(projectId, '/compositions'),
    { method: 'POST', body: JSON.stringify({ segment_ids: segmentIds }) },
  ),
}
