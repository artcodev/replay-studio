import {
  createRouter,
  createWebHistory,
  type RouterHistory,
  type RouteRecordRaw,
} from 'vue-router'
import {
  APP_ROUTE_NAMES,
  APP_ROUTE_PATHS,
  appRouteLocation,
  projectWorkspaceRoute,
} from './lib/appRoutes'

const projectWorkspacePage = () => import('./pages/ProjectWorkspacePage.vue')
const editorPage = () => import('./pages/EditorPage.vue')

const routes: RouteRecordRaw[] = [
  { path: APP_ROUTE_PATHS.root, redirect: { name: APP_ROUTE_NAMES.projects } },
  {
    path: APP_ROUTE_PATHS.projects,
    name: APP_ROUTE_NAMES.projects,
    component: projectWorkspacePage,
  },
  {
    path: '/projects/:projectId',
    redirect: (to) => appRouteLocation(projectWorkspaceRoute(String(to.params.projectId))),
  },
  {
    path: APP_ROUTE_PATHS.projectWorkspace,
    name: APP_ROUTE_NAMES.projectWorkspace,
    component: projectWorkspacePage,
  },
  {
    path: APP_ROUTE_PATHS.videoTimeline,
    name: APP_ROUTE_NAMES.videoTimeline,
    component: editorPage,
  },
  {
    path: APP_ROUTE_PATHS.projectSegment,
    name: APP_ROUTE_NAMES.projectSegment,
    component: editorPage,
  },
  {
    path: APP_ROUTE_PATHS.projectScene,
    name: APP_ROUTE_NAMES.projectScene,
    component: editorPage,
  },
  { path: '/:pathMatch(.*)*', redirect: APP_ROUTE_PATHS.projects },
]

export function createAppRouter(
  history?: RouterHistory,
) {
  return createRouter({
    history: history ?? createWebHistory(import.meta.env.BASE_URL),
    routes,
    scrollBehavior: () => ({ left: 0, top: 0 }),
  })
}
