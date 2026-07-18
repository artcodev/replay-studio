export type PlayerActionType =
  | 'idle'
  | 'walk'
  | 'run'
  | 'sprint'
  | 'turn'
  | 'jump'
  | 'fall'
  | 'get-up'
  | 'first-touch'
  | 'drive'
  | 'pass'
  | 'cross'
  | 'shot'
  | 'header'
  | 'throw-in'
  | 'clearance'
  | 'tackle'
  | 'slide-tackle'
  | 'block'
  | 'interception'
  | 'feint'

export type PlayerActionKeypointKind =
  | 'wind-up'
  | 'contact'
  | 'release'
  | 'apex'
  | 'impact'
  | 'recovery'

export type PlayerActionKeypoint = {
  kind: PlayerActionKeypointKind
  time: number
}

export type PlayerActionStatus = 'suggested' | 'confirmed' | 'rejected'
export type PlayerActionSource = 'automatic' | 'manual'

export type PlayerAction = {
  id: string
  canonicalPersonId: string
  type: PlayerActionType
  startTime: number
  endTime: number
  keypoints: PlayerActionKeypoint[]
  confidence: number
  status: PlayerActionStatus
  source: PlayerActionSource
  evidence?: {
    observationIds?: string[]
    ballTrajectoryFingerprint?: string | null
    model?: string | null
    reasons?: string[]
    artifactUri?: string | null
    artifactHash?: string | null
  }
  createdAt?: string
  updatedAt?: string
}
