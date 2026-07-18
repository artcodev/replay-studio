export function projectPath(projectId: string, suffix = '') {
  return `/api/projects/${encodeURIComponent(projectId)}${suffix}`
}

export function projectScenePath(projectId: string, sceneId: string, suffix = '') {
  return `${projectPath(projectId, `/scenes/${encodeURIComponent(sceneId)}`)}${suffix}`
}

export function projectVideoPath(projectId: string, assetId: string, suffix = '') {
  return `${projectPath(projectId, `/videos/${encodeURIComponent(assetId)}`)}${suffix}`
}
