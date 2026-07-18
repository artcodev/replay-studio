import type { ManualMatchImportRequest } from '../types/match'

function record(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function requiredText(value: unknown): boolean {
  return typeof value === 'string' && Boolean(value.trim())
}

/**
 * Validate only the JSON envelope needed for a useful client error. The API's
 * strict Pydantic contract remains authoritative for every nested field and
 * rejects extras, invalid references and duplicate players atomically.
 */
export function parseManualMatchImport(text: string): ManualMatchImportRequest {
  const source = text.replace(/^\uFEFF/, '').trim()
  if (!source) throw new Error('The roster JSON file is empty.')
  let value: unknown
  try {
    value = JSON.parse(source)
  } catch {
    throw new Error('The roster file is not valid JSON.')
  }
  if (!record(value)) throw new Error('The roster JSON root must be an object.')
  if (!record(value.event) || !requiredText(value.event.id) || !requiredText(value.event.name)) {
    throw new Error('The roster JSON must contain event.id and event.name.')
  }
  if (
    !record(value.teams)
    || !record(value.teams.home)
    || !record(value.teams.away)
    || !requiredText(value.teams.home.id)
    || !requiredText(value.teams.home.name)
    || !requiredText(value.teams.away.id)
    || !requiredText(value.teams.away.name)
  ) {
    throw new Error('The roster JSON must contain named home and away teams with IDs.')
  }
  if (!Array.isArray(value.players) || value.players.length === 0) {
    throw new Error('The roster JSON must contain at least one player.')
  }
  return value as ManualMatchImportRequest
}
