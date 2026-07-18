import type { Project } from '../types/project'

/**
 * Resolve only an explicitly addressed project.
 *
 * The collection-level `/projects` route is deliberately neutral: it never
 * restores, guesses or prefers a project. A project becomes current only when
 * its id is present in the route after a user action or direct deep link.
 */
export function resolveExplicitWorkspaceProjectId(
  projects: Project[],
  requestedId?: string | null,
): string | null {
  if (!requestedId) return null
  return projects.some((project) => project.id === requestedId)
    ? requestedId
    : null
}
