
/**
 * Project workspace contracts intentionally contain only Replay Studio ids.
 * Provider ids and provider-specific payloads stop at the server adapter layer.
 */
export type Project = {
  id: string
  title: string
  revision: number
  matchId?: string | null
  activeSegmentId?: string | null
  createdAt: string
  updatedAt: string
}


export type ProjectAssetStatus = 'queued' | 'uploading' | 'processing' | 'ready' | 'cancelled' | 'failed'

export type ProjectAsset = {
  id: string
  projectId: string
  /** Root scene that owns this video's complete episode timeline. */
  timelineSceneId?: string | null
  filename: string
  duration: number | null
  status: ProjectAssetStatus
  mediaUrl?: string | null
  posterUrl?: string | null
  createdAt: string
}

export type ProjectSegmentStatus = 'pending' | 'ready' | 'analyzing' | 'failed'

export type ProjectSegment = {
  id: string
  projectId: string
  assetId: string
  sourceSegmentId: string
  sceneId?: string | null
  label: string
  start: number
  end: number
  status: ProjectSegmentStatus
}

export type ProjectIdentityAssignmentSource = 'scene-local' | 'accepted-roster' | 'explicit'

export type ProjectIdentityMembership = {
  id: string
  projectId: string
  projectPersonId: string
  sceneId: string
  scenePersonId: string
  assignmentSource: ProjectIdentityAssignmentSource
  identityStatus?: string | null
  identityConfidence?: number | null
  observationCount: number
  createdAt?: string | null
  updatedAt?: string | null
}

/** Durable provider-neutral identity shared by project scenes and camera angles. */
export type ProjectIdentity = {
  id: string
  projectId: string
  rosterPersonId?: string | null
  displayName: string
  teamId?: string | null
  role?: string | null
  jerseyNumber?: string | null
  status: 'active' | 'excluded'
  identityConfidence?: number | null
  memberships: ProjectIdentityMembership[]
  createdAt?: string | null
  updatedAt?: string | null
}

export type ProjectIdentityMembershipAssignment = {
  membershipId: string
  currentProjectPersonId: string
  projectPersonId: string
  sceneId: string
  scenePersonId: string
}

export type AnalysisJobStatus =
  | 'queued'
  | 'running'
  | 'cancelling'
  | 'cancelled'
  | 'succeeded'
  | 'failed'

export type AnalysisJobKind =
  | 'video-processing'
  | 'reconstruction'
  | 'multi-pass'
  | 'model-comparison'
  | 'match-sync'
  | (string & {})

export type AnalysisJobProgress = {
  completed: number
  total: number
  percent: number
  label: string
  detail: string | null
  etaSeconds: number | null
}

export type AnalysisJob = {
  id: string
  projectId: string
  segmentId: string | null
  kind: AnalysisJobKind
  status: AnalysisJobStatus
  phase: string | null
  progress: AnalysisJobProgress
  createdAt: string
  startedAt?: string | null
  finishedAt?: string | null
  error?: string | null
}
