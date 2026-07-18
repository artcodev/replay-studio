import { describe, expect, it } from 'vitest'
import { buildManualPlayerAction } from './playerActionDraft'

describe('buildManualPlayerAction', () => {
  it('keeps a manual action and its keypoint inside the scene', () => {
    const action = buildManualPlayerAction('person-1', 10, 9.9, 'shot')

    expect(action).not.toBeNull()
    expect(action?.canonicalPersonId).toBe('person-1')
    expect(action?.startTime).toBeGreaterThanOrEqual(0)
    expect(action?.endTime).toBeLessThanOrEqual(10)
    expect(action?.keypoints[0]?.time).toBeGreaterThanOrEqual(action?.startTime ?? 0)
    expect(action?.keypoints[0]?.time).toBeLessThanOrEqual(action?.endTime ?? 0)
  })

  it('rejects a scene that cannot contain an interval', () => {
    expect(buildManualPlayerAction('person-1', 0.001, 0)).toBeNull()
  })
})
