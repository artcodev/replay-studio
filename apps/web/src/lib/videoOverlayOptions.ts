export type VideoOverlayOptionKey =
  | 'pitchCalibration'
  | 'projectionDebug'
  | 'playerBoxes'
  | 'ballBoxes'
  | 'identityLabels'
  | 'positionLabels'
  | 'teamSwatches'
  | 'manualMarks'

export type VideoOverlayOptions = Record<VideoOverlayOptionKey, boolean>

export type VideoOverlayGroup = 'geometry' | 'detections' | 'labels' | 'review'

export type VideoOverlayLayerItem = Readonly<{
  key: VideoOverlayOptionKey
  group: VideoOverlayGroup
  label: string
  detail: string
}>

export const VIDEO_OVERLAY_GROUP_LABELS: Readonly<Record<VideoOverlayGroup, string>> =
  Object.freeze({
    geometry: 'Pitch geometry',
    detections: 'Detections',
    labels: 'Labels',
    review: 'Review marks',
  })

export const VIDEO_OVERLAY_LAYER_ITEMS: ReadonlyArray<VideoOverlayLayerItem> =
  Object.freeze([
    {
      key: 'pitchCalibration',
      group: 'geometry',
      label: 'Pitch calibration',
      detail: 'Projected pitch lines of the accepted camera solution',
    },
    {
      key: 'projectionDebug',
      group: 'geometry',
      label: 'Selected object debug',
      detail: 'Stored box contact point, frame homography and metric QA result',
    },
    {
      key: 'playerBoxes',
      group: 'detections',
      label: 'Player boxes',
      detail: 'Detector boxes for people on this frame',
    },
    {
      key: 'ballBoxes',
      group: 'detections',
      label: 'Ball candidates',
      detail: 'Ball detections with the selected candidate highlighted',
    },
    {
      key: 'identityLabels',
      group: 'labels',
      label: 'Identity labels',
      detail: 'Track name, shirt number and canonical person',
    },
    {
      key: 'positionLabels',
      group: 'labels',
      label: 'Pitch positions',
      detail: 'Projected x/z metres per detection',
    },
    {
      key: 'teamSwatches',
      group: 'labels',
      label: 'Team colours',
      detail: 'Kit colour swatch beside each box',
    },
    {
      key: 'manualMarks',
      group: 'review',
      label: 'Manual marks',
      detail: 'Ignored, split and in-progress annotation boxes',
    },
  ])

export const DEFAULT_VIDEO_OVERLAY_OPTIONS: Readonly<VideoOverlayOptions> =
  Object.freeze({
    pitchCalibration: true,
    projectionDebug: false,
    playerBoxes: true,
    ballBoxes: true,
    identityLabels: true,
    positionLabels: true,
    teamSwatches: true,
    manualMarks: true,
  })

export function withVideoOverlayOption(
  options: VideoOverlayOptions,
  key: VideoOverlayOptionKey,
  enabled: boolean,
): VideoOverlayOptions {
  return { ...options, [key]: enabled }
}

/** Layer items grouped for the menu, preserving declaration order. */
export function videoOverlayGroups(): ReadonlyArray<
  Readonly<{ group: VideoOverlayGroup; label: string; items: VideoOverlayLayerItem[] }>
> {
  const groups: Array<{
    group: VideoOverlayGroup
    label: string
    items: VideoOverlayLayerItem[]
  }> = []
  for (const item of VIDEO_OVERLAY_LAYER_ITEMS) {
    const existing = groups.find((entry) => entry.group === item.group)
    if (existing) existing.items.push(item)
    else {
      groups.push({
        group: item.group,
        label: VIDEO_OVERLAY_GROUP_LABELS[item.group],
        items: [item],
      })
    }
  }
  return groups
}
