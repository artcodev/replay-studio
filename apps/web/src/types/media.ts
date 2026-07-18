export type VideoSegment = {
  id: string
  label: string
  start: number
  end: number
  duration: number
  score: number
  recommended?: boolean
  sceneId?: string
  layout?: {
    group: number
    variant: string
    label: string
    role: 'original' | 'replay' | 'continuation'
    confidence: number
    motionCost?: number | null
  }
}

export type SegmentLayout = {
  status: 'proposed' | 'edited' | 'confirmed'
  method: 'scoreboard-change+motion-dtw' | 'shot-order-fallback' | 'empty'
  confidence: number
  scoreChangeTimes: number[]
  groups: Array<{
    id: string
    index: number
    label: string
    segmentIds: string[]
    replayCount: number
  }>
  warnings: string[]
}

export type MultiPassSummary = {
  id: string
  status: 'queued' | 'processing' | 'ready' | 'cancelled' | 'failed'
  parentSceneId: string
  selectedSegmentIds: string[]
  referenceSceneId?: string | null
  currentPass: number
  passes: Array<{
    sceneId: string
    segmentId: string
    label: string
    sourceStart: number
    sourceEnd: number
    status: 'ready' | 'failed'
    quality: number
    trackCount: number
    ballSamples: number
    calibrationStatus: 'ready' | 'approximate' | 'fallback' | 'rejected'
    calibrationConfidence?: number | null
    qualityVerdict: 'pass' | 'review' | 'reject'
    relation?: 'reference' | 'replay-overlap' | 'continuation-before' | 'continuation-after' | 'independent'
    alignment?: {
      relation: 'reference' | 'replay-overlap' | 'continuation-before' | 'continuation-after' | 'independent'
      method: 'identity' | 'motion-dtw' | 'source-continuity' | 'phase-normalized'
      confidence: number
      motionCost: number
      overlap: boolean
      anchors: Array<{ referenceTime: number; passTime: number }>
    }
    error?: string | null
  }>
  consensus?: {
    passesAnalyzed: number
    metricPasses: number
    ballPasses: number
    trackPasses: number
    overlappingPasses?: number
    evidenceScore: number
  } | null
  ballSupport?: {
    referenceSamples: number
    supportedSamples: number
    visualPasses: number
    metricPasses: number
    spatialErrors: number[]
  }
  warnings: string[]
}

export type VideoAsset = {
  id: string
  filename: string
  original_name: string
  content_type: string
  status: 'queued' | 'processing' | 'ready' | 'cancelled' | 'failed'
  stage: string
  progress: number
  duration?: number | null
  width?: number | null
  height?: number | null
  fps?: number | null
  frame_count: number
  scene_id?: string | null
  media_url?: string | null
  poster_url?: string | null
  error?: string | null
  created_at?: string | null
}
