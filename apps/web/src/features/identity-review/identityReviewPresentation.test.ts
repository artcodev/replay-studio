import { describe, expect, it } from 'vitest'
import {
  formatIdentityReviewTime,
  identityObservationBoxStyle,
  orderedIdentityRosterPlayers,
} from './identityReviewPresentation'

describe('identity review presentation', () => {
  it('orders roster players by team, numeric shirt number and name', () => {
    const players = [
      { id: '3', name: 'Zed', number: '10', team_name: 'Away' },
      { id: '2', name: 'Beta', number: '11', team_name: 'Home' },
      { id: '1', name: 'Alpha', number: '2', team_name: 'Home' },
    ]
    expect(orderedIdentityRosterPlayers(players as never).map((player) => player.id)).toEqual(['3', '1', '2'])
  })

  it('formats source time and clips overlay boxes to the frame', () => {
    expect(formatIdentityReviewTime(65.125)).toBe('01:05.125')
    expect(identityObservationBoxStyle({
      id: 'o1',
      frameIndex: 1,
      sceneTime: 0,
      frameWidth: 100,
      frameHeight: 50,
      bbox: { x: -10, y: 10, width: 120, height: 45 },
    })).toEqual({ left: '0%', top: '20%', width: '100%', height: '90%' })
  })
})
