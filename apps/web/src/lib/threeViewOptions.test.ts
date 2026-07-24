import { describe, expect, it } from 'vitest'
import {
  DEFAULT_THREE_VIEW_OPTIONS,
  THREE_VIEW_LAYER_ITEMS,
  withInferredPositionRenderMode,
  withThreeViewOption,
  type ThreeViewOptions,
} from './threeViewOptions'

describe('threeViewOptions', () => {
  it('uses complete defaults and keeps the selected-object path opt-in', () => {
    expect(DEFAULT_THREE_VIEW_OPTIONS).toEqual({
      models: true,
      labels: true,
      trajectory: true,
      pathTracking: false,
      allPaths: false,
      ball: true,
      analysisMarkers: true,
      inferredPositions: 'transparent',
    })
  })

  it('updates the inferred-position render mode without touching toggles', () => {
    const current: ThreeViewOptions = { ...DEFAULT_THREE_VIEW_OPTIONS }
    const updated = withInferredPositionRenderMode(current, 'transparent')

    expect(updated).not.toBe(current)
    expect(updated.inferredPositions).toBe('transparent')
    expect(current.inferredPositions).toBe('transparent')
    expect(updated.models).toBe(true)
  })

  it('returns an immutable option update', () => {
    const current: ThreeViewOptions = { ...DEFAULT_THREE_VIEW_OPTIONS }
    const updated = withThreeViewOption(current, 'labels', false)

    expect(updated).not.toBe(current)
    expect(updated.labels).toBe(false)
    expect(current.labels).toBe(true)
    expect(updated.models).toBe(true)
  })

  it('describes path tracking as one selected-object layer across video and 3D', () => {
    expect(THREE_VIEW_LAYER_ITEMS.find((item) => item.key === 'pathTracking')).toEqual({
      key: 'pathTracking',
      label: 'Path tracking',
      detail: 'Selected player or ball on video + 3D',
    })
    expect(THREE_VIEW_LAYER_ITEMS.find((item) => item.key === 'trajectory')?.label).toBe('Ball trajectory')
  })
})
