import { ReportDashboard } from '@/components/report-dashboard'
import { api } from '@/lib/api'

export default async function ReportPage({ params }: { params: Promise<{ videoId: string }> }) {
  const { videoId } = await params
  const response = await api.getReport(videoId)

  return (
    <main className="console-shell">
      <header className="site-header">
        <a className="wordmark" href="/" aria-label="Контур — главная">КОНТУР</a>
        <span className="header-label">Инспекция стройплощадки</span>
        <span className="header-status">Отчёт готов</span>
      </header>
      {'findings' in response ? <ReportDashboard report={response} /> : <section className="report-pending"><p className="eyebrow">ОТЧЁТ ЕЩЁ НЕ ГОТОВ</p><h1>Проверка продолжается</h1><p>{response.error || 'Отчёт появится после завершения анализа.'}</p><a className="action-button" href={`/processing/${videoId}`}>К ходу проверки</a></section>}
    </main>
  )
}
