'use client'

import { useMemo, useState } from 'react'

import { FindingList } from '@/components/finding-list'
import { ReportOverview } from '@/components/report-overview'
import type { Finding, Report } from '@/lib/types'

export function ReportDashboard({ report }: { report: Report }) {
  const [severity, setSeverity] = useState<'all' | Finding['severity']>('all')
  const [category, setCategory] = useState('all')
  const [query, setQuery] = useState('')
  const categories = useMemo(() => [...new Set(report.findings.map((finding) => finding.category))], [report.findings])
  const visible = report.findings.filter((finding) =>
    (severity === 'all' || finding.severity === severity) &&
    (category === 'all' || finding.category === category) &&
    `${finding.title} ${finding.comment}`.toLowerCase().includes(query.toLowerCase()),
  )

  return (
    <div className="report-layout">
      <ReportOverview report={report} />
      <section className="report-findings" aria-labelledby="findings-title">
        <div className="findings-header">
          <div>
            <p className="eyebrow">РЕЗУЛЬТАТЫ</p>
            <h1 id="findings-title">Выявленные замечания</h1>
          </div>
          <span className="findings-count">{visible.length} из {report.findings.length}</span>
        </div>
        <div className="finding-filters">
          <label>Поиск<input onChange={(event) => setQuery(event.target.value)} placeholder="Поиск по замечаниям" type="search" value={query} /></label>
          <label>Серьёзность<select onChange={(event) => setSeverity(event.target.value as 'all' | Finding['severity'])} value={severity}><option value="all">Все уровни</option><option value="critical">Критично</option><option value="high">Высокая</option><option value="medium">Средняя</option><option value="low">Низкая</option></select></label>
          <label>Категория<select onChange={(event) => setCategory(event.target.value)} value={category}><option value="all">Все категории</option>{categories.map((item) => <option key={item} value={item}>{item.replaceAll('_', ' ')}</option>)}</select></label>
        </div>
        <FindingList findings={visible} onEvidenceClick={(frameId) => window.history.replaceState(null, '', `?frame=${frameId}`)} />
      </section>
    </div>
  )
}
