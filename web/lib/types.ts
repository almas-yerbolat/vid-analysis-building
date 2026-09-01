export type VideoStatus =
  | 'uploaded'
  | 'queued'
  | 'probing'
  | 'sampling'
  | 'analyzing'
  | 'aggregating'
  | 'done'
  | 'failed'

export interface Video {
  id: string
  filename: string
  project_name: string
  status: VideoStatus
  is_photo: boolean
  created_at: string
}

export interface StatusEvent {
  status: VideoStatus
  progress_pct: number
  progress_note: string
  error: string
}

export interface Evidence {
  frame_id: string
  ts_ms: number
  thumb_url: string
  full_url: string
  comment: string
  boxes: Array<{ label: string; box_2d: [number, number, number, number] }>
}

export interface Finding {
  id: string
  category: string
  subtype: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  title: string
  comment: string
  confidence: number
  status: string
  evidence: Evidence[]
}

export interface Report {
  video_id: string
  generated_at: string
  meta: {
    duration_s: number
    frames_analyzed: number
    frames_extracted: number
    batches_failed: number
  }
  stage: {
    primary: string
    secondary: string
    confidence: number
    evidence_frames: string[]
  }
  summary_ru: string
  equipment: Array<{ type: string; max_count: number; evidence_frame: string | null }>
  activity_timeline: Array<Record<string, unknown>>
  findings: Finding[]
  stats: Record<Finding['severity'], number>
}

export interface PendingReport {
  status: VideoStatus
  error: string
}
