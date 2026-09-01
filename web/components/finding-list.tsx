import type { Finding } from '@/lib/types'

const severityLabels = {
  critical: 'Критично',
  high: 'Высокая',
  medium: 'Средняя',
  low: 'Низкая',
} as const

const time = (milliseconds: number) => {
  const seconds = Math.floor(milliseconds / 1000)
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`
}

export function FindingList({ findings, onEvidenceClick }: { findings: Finding[]; onEvidenceClick: (frameId: string) => void }) {
  if (!findings.length) return <p className="findings-empty">Нарушений по выбранным условиям не найдено.</p>

  return (
    <ol className="finding-list">
      {findings.map((finding) => (
        <li className="finding-card" key={finding.id}>
          <div className="finding-heading">
            <span className={`severity-chip severity-${finding.severity}`}>{severityLabels[finding.severity]}</span>
            <span className="finding-category">{finding.category.replaceAll('_', ' ')}</span>
          </div>
          <h2>{finding.title}</h2>
          <p>{finding.comment}</p>
          <p className="finding-confidence">Достоверность: {Math.round(finding.confidence * 100)}%</p>
          {finding.evidence.length > 0 && (
            <div className="evidence-strip" aria-label={`Кадры: ${finding.title}`}>
              {finding.evidence.map((evidence) => (
                <button className="evidence-thumb" key={evidence.frame_id} onClick={() => onEvidenceClick(evidence.frame_id)} type="button">
                  <img alt={`Кадр ${time(evidence.ts_ms)}`} src={evidence.thumb_url} />
                  <span>{time(evidence.ts_ms)}</span>
                </button>
              ))}
            </div>
          )}
        </li>
      ))}
    </ol>
  )
}
