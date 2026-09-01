import type { Report } from '@/lib/types'

const severityLabels = {
  critical: 'Критично',
  high: 'Высокая',
  medium: 'Средняя',
  low: 'Низкая',
} as const

const duration = (seconds: number) => `${Math.floor(seconds / 60)}:${Math.round(seconds % 60).toString().padStart(2, '0')}`

const time = (milliseconds: number) => duration(Math.floor(milliseconds / 1000))

export function ReportOverview({ report }: { report: Report }) {
  return (
    <aside className="report-overview" aria-label="Сводка отчёта">
      <section className="overview-summary">
        <p className="eyebrow">ИТОГ ПРОВЕРКИ</p>
        <p>{report.summary_ru}</p>
        <p className="report-meta">Видео {duration(report.meta.duration_s)} · Проанализировано кадров: {report.meta.frames_analyzed}</p>
      </section>

      <section className="overview-section">
        <h2>Стадия работ</h2>
        <strong className="stage-reading">{report.stage.primary}</strong>
        {report.stage.secondary && <span className="stage-secondary">Дополнительно: {report.stage.secondary}</span>}
        <span className="stage-confidence">Достоверность: {Math.round(report.stage.confidence * 100)}%</span>
      </section>

      <section className="overview-section" aria-label="Количество нарушений по серьёзности">
        <h2>Нарушения</h2>
        <ul className="severity-stats">
          {(Object.keys(severityLabels) as Array<keyof typeof severityLabels>).map((severity) => (
            <li className={`severity-${severity}`} key={severity}><strong>{report.stats[severity]}</strong><span>{severityLabels[severity]}</span></li>
          ))}
        </ul>
      </section>

      <section className="overview-section">
        <h2>Техника на объекте</h2>
        {report.equipment.length ? (
          <ul className="equipment-list">
            {report.equipment.map((equipment) => <li key={equipment.type}><span>{equipment.type.replaceAll('_', ' ')}</span><strong>до {equipment.max_count}</strong></li>)}
          </ul>
        ) : <p className="overview-empty">Техника не зафиксирована.</p>}
      </section>

      <section className="overview-section">
        <h2>Ход работ</h2>
        {report.activity_timeline.length ? (
          <ol className="activity-timeline">
            {report.activity_timeline.map((entry) => <li key={`${entry.from_ms}-${entry.to_ms}`}><time>{time(entry.from_ms)}—{time(entry.to_ms)}</time><span>{entry.activity}</span></li>)}
          </ol>
        ) : <p className="overview-empty">Активность не определена.</p>}
      </section>
    </aside>
  )
}
