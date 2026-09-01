import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { ReportDashboard } from '@/components/report-dashboard'
import type { Report } from '@/lib/types'

const reportWithHighAndLowFindings: Report = {
  video_id: 'vid_1',
  generated_at: '2026-09-01T10:00:00Z',
  meta: { duration_s: 120, frames_analyzed: 24, frames_extracted: 30, batches_failed: 0 },
  stage: { primary: 'каркас', secondary: '', confidence: 0.88, evidence_frames: [] },
  summary_ru: 'На объекте ведутся работы по устройству каркаса.',
  equipment: [{ type: 'башенный кран', max_count: 1, evidence_frame: 'frm_1' }],
  activity_timeline: [{ from_ms: 0, to_ms: 120000, activity: 'Монтаж опалубки' }],
  findings: [
    {
      id: 'fnd_1', category: 'тб_от', subtype: 'отсутствие_каски', severity: 'high',
      title: 'Рабочие без касок', comment: 'На перекрытии видны рабочие без каски.',
      confidence: 0.9, status: 'unreviewed',
      evidence: [{ frame_id: 'frm_1', ts_ms: 15000, thumb_url: '/api/frames/frm_1?thumb=1', full_url: '/api/frames/frm_1', comment: 'Два рабочих без касок.', boxes: [] }],
    },
    {
      id: 'fnd_2', category: 'экология_клининг', subtype: 'свалка_мусора', severity: 'low',
      title: 'Мусор у ограждения', comment: 'Строительный мусор складирован у ограждения.',
      confidence: 0.76, status: 'unreviewed', evidence: [],
    },
  ],
  stats: { critical: 0, high: 1, medium: 0, low: 1 },
}

describe('ReportDashboard', () => {
  afterEach(cleanup)

  it('filters findings by severity and search text', async () => {
    const user = userEvent.setup()
    render(<ReportDashboard report={reportWithHighAndLowFindings} />)

    await user.selectOptions(screen.getByLabelText('Серьёзность'), 'high')
    await user.type(screen.getByLabelText('Поиск'), 'каски')

    expect(screen.getByText('Рабочие без касок')).toBeTruthy()
    expect(screen.queryByText('Мусор у ограждения')).toBeNull()
  })
})
