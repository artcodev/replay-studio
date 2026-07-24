export class ApiRequestError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
    this.name = 'ApiRequestError'
  }
}

export function isConflictError(cause: unknown): boolean {
  return cause instanceof ApiRequestError && cause.status === 409
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    const message = requestErrorMessage(body, response.status)
    // A bare "409 (Conflict)" in the console hides which fence refused the
    // write. Always print the server's own reason next to the request.
    console.error(`${init?.method ?? 'GET'} ${path} → ${response.status}: ${message}`)
    throw new ApiRequestError(message, response.status)
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
