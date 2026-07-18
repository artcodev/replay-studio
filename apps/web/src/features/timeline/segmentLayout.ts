import type { SceneDocument } from '../../types/scene'
import type { VideoSegment } from '../../types/media'

export type SceneVideo = NonNullable<SceneDocument['payload']['videoAsset']>

export function segmentGroupColor(group = 1) {
  return ['#ffd36a', '#71e2aa', '#76a9ff', '#dc89ff', '#ff8b6b', '#68d9d4'][(group - 1) % 6]
}

export function alphabeticVariant(index: number) {
  let value = index
  let output = ''
  while (value >= 0) {
    output = String.fromCharCode(65 + value % 26) + output
    value = Math.floor(value / 26) - 1
  }
  return output
}

/** Restores the canonical event/variant labels after a grouping edit. */
export function normalizeSegmentLayout(video: SceneVideo, compact = false) {
  if (!video.segments?.length || !video.segmentLayout) return
  const ordered = [...video.segments].sort((left, right) => left.start - right.start)
  const groupOrder = [...new Set(ordered.map((segment) => segment.layout?.group ?? 1))]
  const groupMap = new Map(groupOrder.map((group, index) => [group, compact ? index + 1 : group]))
  const grouped = new Map<number, VideoSegment[]>()
  for (const segment of ordered) {
    const group = groupMap.get(segment.layout?.group ?? 1) ?? 1
    const items = grouped.get(group) ?? []
    items.push(segment)
    grouped.set(group, items)
  }
  video.segmentLayout.groups = [...grouped.entries()].map(([group, items]) => {
    items.forEach((segment, index) => {
      const variant = alphabeticVariant(index)
      const currentRole = segment.layout?.role ?? (index === 0 ? 'original' : 'continuation')
      const role = index === 0 ? 'original' : currentRole === 'original' ? 'continuation' : currentRole
      segment.layout = {
        group,
        variant,
        label: `${group}-${variant}`,
        role,
        confidence: segment.layout?.confidence ?? 1,
        motionCost: segment.layout?.motionCost,
      }
    })
    return {
      id: `event-${group}`,
      index: group,
      label: String(group),
      segmentIds: items.map((item) => item.id),
      replayCount: items.filter((item) => item.layout?.role === 'replay').length,
    }
  })
}

export function canSplitSegmentTail(video: SceneVideo | null, selectedIds: string[]) {
  if (!video?.segments?.length || !video.segmentLayout || !selectedIds.length) return false
  const selectedSet = new Set(selectedIds)
  const selected = video.segments.filter((segment) => selectedSet.has(segment.id))
  const group = selected[0]?.layout?.group
  if (!group || selected.some((segment) => segment.layout?.group !== group)) return false
  const groupSegments = video.segments
    .filter((segment) => segment.layout?.group === group)
    .sort((left, right) => left.start - right.start)
  const selectedIndexes = groupSegments
    .map((segment, index) => selectedSet.has(segment.id) ? index : -1)
    .filter((index) => index >= 0)
  if (!selectedIndexes.length || selectedIndexes[0] === 0) return false
  return selectedIndexes.every((index, offset) => (
    index === selectedIndexes[0] + offset
    && index === groupSegments.length - selectedIndexes.length + offset
  ))
}

export function splitSegmentTail(video: SceneVideo, selectedIds: string[]) {
  if (!canSplitSegmentTail(video, selectedIds) || !video.segmentLayout || !video.segments) return null
  const selectedSet = new Set(selectedIds)
  const selected = video.segments.filter((segment) => selectedSet.has(segment.id))
  const sourceGroup = selected[0]?.layout?.group
  if (!sourceGroup) return null
  for (const segment of video.segments) {
    if (segment.layout && segment.layout.group > sourceGroup) segment.layout.group += 1
  }
  for (const segment of selected) {
    if (!segment.layout) continue
    segment.layout.group = sourceGroup + 1
    segment.layout.confidence = 1
  }
  video.segmentLayout.status = 'edited'
  normalizeSegmentLayout(video)
  return sourceGroup + 1
}

