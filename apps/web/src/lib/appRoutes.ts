/**
 * Canonical browser route contract for Replay Studio.
 *
 * Resource identity belongs in path parameters. Query parameters are limited
 * to transient editor view state, so copying a URL always identifies one
 * project resource unambiguously.
 */

export const PROJECT_WORKSPACE_TABS = [
  'overview',
  'match',
  'identities',
  'analysis',
] as const

export type ProjectWorkspaceTab = (typeof PROJECT_WORKSPACE_TABS)[number]

export const EDITOR_ROUTE_PANELS = [
  'inspector',
  'quality',
  'events',
] as const

export type EditorRoutePanel = (typeof EDITOR_ROUTE_PANELS)[number]

export type EditorRouteView = {
  time?: number
  panel?: EditorRoutePanel
  subject?: string
}

export const APP_ROUTE_NAMES = {
  projects: 'projects',
  projectWorkspace: 'project-workspace',
  videoTimeline: 'video-timeline',
  projectSegment: 'project-segment',
  projectScene: 'project-scene',
} as const

export const APP_ROUTE_PATHS = {
  root: '/',
  projects: '/projects',
  projectWorkspace: '/projects/:projectId/:tab(overview|match|identities|analysis)',
  videoTimeline: '/projects/:projectId/videos/:assetId/timeline',
  projectSegment: '/projects/:projectId/segments/:segmentId',
  projectScene: '/projects/:projectId/scenes/:sceneId',
} as const

export type AppRouteIntent =
  | { name: typeof APP_ROUTE_NAMES.projects }
  | {
      name: typeof APP_ROUTE_NAMES.projectWorkspace
      projectId: string
      tab: ProjectWorkspaceTab
    }
  | {
      name: typeof APP_ROUTE_NAMES.videoTimeline
      projectId: string
      assetId: string
      view?: EditorRouteView
    }
  | {
      name: typeof APP_ROUTE_NAMES.projectSegment
      projectId: string
      segmentId: string
      view?: EditorRouteView
    }
  | {
      name: typeof APP_ROUTE_NAMES.projectScene
      projectId: string
      sceneId: string
      view?: EditorRouteView
    }

export type AppRouteLocation = {
  name: AppRouteIntent['name']
  params?: Record<string, string>
  query?: Record<string, string>
}

const workspaceTabSet = new Set<string>(PROJECT_WORKSPACE_TABS)
const editorPanelSet = new Set<string>(EDITOR_ROUTE_PANELS)

export function isProjectWorkspaceTab(value: unknown): value is ProjectWorkspaceTab {
  return typeof value === 'string' && workspaceTabSet.has(value)
}

export function isEditorRoutePanel(value: unknown): value is EditorRoutePanel {
  return typeof value === 'string' && editorPanelSet.has(value)
}

function routeParam(value: string, label: string) {
  if (!value.trim()) throw new Error(`${label} must not be empty`)
  return value
}

export function projectsRoute(): AppRouteIntent {
  return { name: APP_ROUTE_NAMES.projects }
}

export function projectWorkspaceRoute(
  projectId: string,
  tab: ProjectWorkspaceTab = 'overview',
): AppRouteIntent {
  return {
    name: APP_ROUTE_NAMES.projectWorkspace,
    projectId: routeParam(projectId, 'projectId'),
    tab,
  }
}

export function videoTimelineRoute(
  projectId: string,
  assetId: string,
  view?: EditorRouteView,
): AppRouteIntent {
  return {
    name: APP_ROUTE_NAMES.videoTimeline,
    projectId: routeParam(projectId, 'projectId'),
    assetId: routeParam(assetId, 'assetId'),
    ...(view ? { view } : {}),
  }
}

export function projectSegmentRoute(
  projectId: string,
  segmentId: string,
  view?: EditorRouteView,
): AppRouteIntent {
  return {
    name: APP_ROUTE_NAMES.projectSegment,
    projectId: routeParam(projectId, 'projectId'),
    segmentId: routeParam(segmentId, 'segmentId'),
    ...(view ? { view } : {}),
  }
}

export function projectSceneRoute(
  projectId: string,
  sceneId: string,
  view?: EditorRouteView,
): AppRouteIntent {
  return {
    name: APP_ROUTE_NAMES.projectScene,
    projectId: routeParam(projectId, 'projectId'),
    sceneId: routeParam(sceneId, 'sceneId'),
    ...(view ? { view } : {}),
  }
}

function singleQueryValue(query: URLSearchParams, name: string) {
  const values = query.getAll(name)
  return values.length === 1 ? values[0] : null
}

/** Parse only supported, well-formed editor view state. Unknown keys are ignored. */
export function parseEditorRouteView(query: URLSearchParams): EditorRouteView | undefined {
  const view: EditorRouteView = {}
  const timeValue = singleQueryValue(query, 't')
  if (timeValue !== null && timeValue.trim() !== '') {
    const time = Number(timeValue)
    if (Number.isFinite(time) && time >= 0) view.time = time
  }

  const panel = singleQueryValue(query, 'panel')
  if (isEditorRoutePanel(panel)) view.panel = panel

  const subject = singleQueryValue(query, 'subject')
  if (subject !== null && subject.trim()) view.subject = subject

  return Object.keys(view).length ? view : undefined
}

export function editorRouteQuery(view?: EditorRouteView): Record<string, string> | undefined {
  if (!view) return undefined
  const query: Record<string, string> = {}
  if (Number.isFinite(view.time) && (view.time ?? -1) >= 0) query.t = String(view.time)
  if (isEditorRoutePanel(view.panel)) query.panel = view.panel
  if (view.subject?.trim()) query.subject = view.subject
  return Object.keys(query).length ? query : undefined
}

/** Convert the framework-neutral intent to a Vue Router compatible location. */
export function appRouteLocation(route: AppRouteIntent): AppRouteLocation {
  switch (route.name) {
    case APP_ROUTE_NAMES.projects:
      return { name: route.name }
    case APP_ROUTE_NAMES.projectWorkspace:
      return {
        name: route.name,
        params: { projectId: route.projectId, tab: route.tab },
      }
    case APP_ROUTE_NAMES.videoTimeline:
      return locationWithEditorView({
        name: route.name,
        params: { projectId: route.projectId, assetId: route.assetId },
      }, route.view)
    case APP_ROUTE_NAMES.projectSegment:
      return locationWithEditorView({
        name: route.name,
        params: { projectId: route.projectId, segmentId: route.segmentId },
      }, route.view)
    case APP_ROUTE_NAMES.projectScene:
      return locationWithEditorView({
        name: route.name,
        params: { projectId: route.projectId, sceneId: route.sceneId },
      }, route.view)
  }
}

function locationWithEditorView(location: AppRouteLocation, view?: EditorRouteView) {
  const query = editorRouteQuery(view)
  return query ? { ...location, query } : location
}

function encoded(value: string) {
  return encodeURIComponent(value)
}

function editorRouteSearch(view?: EditorRouteView) {
  const query = editorRouteQuery(view)
  if (!query) return ''
  return `?${new URLSearchParams(query).toString()}`
}

export function appRoutePath(route: AppRouteIntent): string {
  switch (route.name) {
    case APP_ROUTE_NAMES.projects:
      return APP_ROUTE_PATHS.projects
    case APP_ROUTE_NAMES.projectWorkspace:
      return `/projects/${encoded(route.projectId)}/${route.tab}`
    case APP_ROUTE_NAMES.videoTimeline:
      return `/projects/${encoded(route.projectId)}/videos/${encoded(route.assetId)}/timeline${editorRouteSearch(route.view)}`
    case APP_ROUTE_NAMES.projectSegment:
      return `/projects/${encoded(route.projectId)}/segments/${encoded(route.segmentId)}${editorRouteSearch(route.view)}`
    case APP_ROUTE_NAMES.projectScene:
      return `/projects/${encoded(route.projectId)}/scenes/${encoded(route.sceneId)}${editorRouteSearch(route.view)}`
  }
}

function decodedPathSegments(pathname: string): string[] | null {
  const raw = pathname.split('/')
  if (raw[0] !== '') return null
  while (raw.length > 1 && raw[raw.length - 1] === '') raw.pop()
  const segments = raw.slice(1)
  if (segments.some((segment) => !segment)) return null
  try {
    return segments.map((segment) => decodeURIComponent(segment))
  } catch {
    return null
  }
}

/**
 * Parse a browser URL into the same route intent used by navigation actions.
 * `/projects/:projectId` is accepted as the redirect source and normalized to
 * the explicit overview workspace intent.
 */
export function parseAppRoute(input: string): AppRouteIntent | null {
  let url: URL
  try {
    url = new URL(input, 'http://replay-studio.local')
  } catch {
    return null
  }
  const segments = decodedPathSegments(url.pathname)
  if (!segments) return null

  if (!segments.length || (segments.length === 1 && segments[0] === 'projects')) {
    return projectsRoute()
  }
  if (segments[0] !== 'projects') return null

  const projectId = segments[1]
  if (!projectId?.trim()) return null
  if (segments.length === 2) return projectWorkspaceRoute(projectId)

  const third = segments[2]
  if (segments.length === 3 && isProjectWorkspaceTab(third)) {
    return projectWorkspaceRoute(projectId, third)
  }

  const resourceId = segments[3]
  if (!resourceId?.trim()) return null
  const view = parseEditorRouteView(url.searchParams)
  if (segments.length === 5 && third === 'videos' && segments[4] === 'timeline') {
    return videoTimelineRoute(projectId, resourceId, view)
  }
  if (segments.length === 4 && third === 'segments') {
    return projectSegmentRoute(projectId, resourceId, view)
  }
  if (segments.length === 4 && third === 'scenes') {
    return projectSceneRoute(projectId, resourceId, view)
  }
  return null
}
