import { describe, expect, it } from 'vitest'
import {
  loadThreeViewPreferences,
  parseThreeViewPreferences,
  saveThreeViewPreferences,
} from './threeViewPreferences'

describe('three view preferences', () => {
  it('restores a complete valid preference', () => {
    expect(parseThreeViewPreferences(JSON.stringify({
      options: { models: true, labels: false, trajectory: true, pathTracking: true, ball: true, analysisMarkers: false },
      renderQuality: 'enhanced',
    }))).toEqual({
      options: { models: true, labels: false, trajectory: true, pathTracking: true, ball: true, analysisMarkers: false },
      renderQuality: 'enhanced',
    })
  })

  it('migrates a stored v1 preference by defaulting the new path layer off', () => {
    expect(parseThreeViewPreferences(JSON.stringify({
      options: { models: false, labels: false, trajectory: true, ball: true, analysisMarkers: false },
      renderQuality: 'basic',
    }))).toEqual({
      options: {
        models: false,
        labels: false,
        trajectory: true,
        pathTracking: false,
        ball: true,
        analysisMarkers: false,
      },
      renderQuality: 'basic',
    })
  })

  it('ignores malformed or incomplete persisted state', () => {
    expect(parseThreeViewPreferences('{broken')).toBeNull()
    expect(parseThreeViewPreferences(JSON.stringify({
      options: { models: true },
      renderQuality: 'basic',
    }))).toBeNull()
    expect(parseThreeViewPreferences(JSON.stringify({
      options: {
        models: true,
        labels: true,
        trajectory: true,
        pathTracking: 'yes',
        ball: true,
        analysisMarkers: true,
      },
      renderQuality: 'basic',
    }))).toBeNull()
  })

  it('does not break the workspace when browser storage is unavailable', () => {
    expect(loadThreeViewPreferences({
      getItem: () => { throw new Error('storage denied') },
    })).toBeNull()

    expect(saveThreeViewPreferences({
      setItem: () => { throw new Error('quota exceeded') },
    }, {
      options: { models: true, labels: true, trajectory: true, pathTracking: false, ball: true, analysisMarkers: true },
      renderQuality: 'basic',
    })).toBe(false)
  })
})
