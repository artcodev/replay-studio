export type ThreeRenderQuality = 'basic' | 'enhanced'

export type ThreeViewOptionKey =
  | 'models'
  | 'labels'
  | 'trajectory'
  | 'pathTracking'
  | 'allPaths'
  | 'ball'
  | 'analysisMarkers'

export type ThreeViewOptions = Record<ThreeViewOptionKey, boolean>

export type ThreeViewLayerItem = Readonly<{
  key: ThreeViewOptionKey
  label: string
  detail: string
}>

export const THREE_VIEW_LAYER_ITEMS: ReadonlyArray<ThreeViewLayerItem> = Object.freeze([
  { key: 'models', label: 'Player models', detail: '3D players on the pitch' },
  { key: 'labels', label: 'Player labels', detail: 'Names and shirt numbers' },
  { key: 'trajectory', label: 'Ball trajectory', detail: 'Tracked path through the moment' },
  { key: 'pathTracking', label: 'Path tracking', detail: 'Selected player or ball on video + 3D' },
  { key: 'allPaths', label: 'All paths', detail: 'Every player trajectory across the whole moment' },
  { key: 'ball', label: 'Ball', detail: 'Tracked ball model' },
  { key: 'analysisMarkers', label: 'Analysis markers', detail: 'Current-frame detections' },
])

export const DEFAULT_THREE_VIEW_OPTIONS: Readonly<ThreeViewOptions> = Object.freeze({
  models: true,
  labels: true,
  trajectory: true,
  pathTracking: false,
  allPaths: false,
  ball: true,
  analysisMarkers: true,
})

export function withThreeViewOption(
  options: ThreeViewOptions,
  key: ThreeViewOptionKey,
  enabled: boolean,
): ThreeViewOptions {
  return { ...options, [key]: enabled }
}
