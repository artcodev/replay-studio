import {
  DEFAULT_THREE_VIEW_OPTIONS,
  isInferredPositionRenderMode,
  type ThreeRenderQuality,
  type ThreeViewOptions,
} from './threeViewOptions'

export const THREE_VIEW_PREFERENCES_KEY = 'replay-studio:three-view:v1'

export type ThreeViewPreferences = {
  options: ThreeViewOptions
  renderQuality: ThreeRenderQuality
}

// These options shipped with the v1 storage key. New options may be absent in
// an older payload and are filled from DEFAULT_THREE_VIEW_OPTIONS, but the
// original fields remain required so a partial/corrupt payload is not silently
// accepted as a valid preference set.
const V1_REQUIRED_OPTION_KEYS: ReadonlyArray<keyof ThreeViewOptions> = [
  'models',
  'labels',
  'trajectory',
  'ball',
  'analysisMarkers',
]

export function parseThreeViewPreferences(value: string | null): ThreeViewPreferences | null {
  if (!value) return null
  try {
    const parsed = JSON.parse(value) as Partial<ThreeViewPreferences>
    const options = parsed.options
    if (!options || (parsed.renderQuality !== 'basic' && parsed.renderQuality !== 'enhanced')) return null
    if (V1_REQUIRED_OPTION_KEYS.some((key) => typeof options[key] !== 'boolean')) return null
    const booleanKeys = (Object.keys(DEFAULT_THREE_VIEW_OPTIONS) as Array<keyof ThreeViewOptions>)
      .filter((key) => typeof DEFAULT_THREE_VIEW_OPTIONS[key] === 'boolean')
    if (booleanKeys.some((key) => options[key] !== undefined && typeof options[key] !== 'boolean')) return null
    // inferredPositions is an enum field: an unknown/absent value backfills to
    // the default rather than rejecting an otherwise valid v1 payload.
    const inferredPositions = isInferredPositionRenderMode(options.inferredPositions)
      ? options.inferredPositions
      : DEFAULT_THREE_VIEW_OPTIONS.inferredPositions
    return {
      options: { ...DEFAULT_THREE_VIEW_OPTIONS, ...options, inferredPositions },
      renderQuality: parsed.renderQuality,
    }
  } catch {
    return null
  }
}

export function loadThreeViewPreferences(storage: Pick<Storage, 'getItem'>): ThreeViewPreferences | null {
  try {
    return parseThreeViewPreferences(storage.getItem(THREE_VIEW_PREFERENCES_KEY))
  } catch {
    return null
  }
}

export function saveThreeViewPreferences(
  storage: Pick<Storage, 'setItem'>,
  preferences: ThreeViewPreferences,
) {
  try {
    storage.setItem(THREE_VIEW_PREFERENCES_KEY, JSON.stringify(preferences))
    return true
  } catch {
    return false
  }
}
