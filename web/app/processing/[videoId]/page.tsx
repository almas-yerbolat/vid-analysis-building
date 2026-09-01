import { ProcessingStatus } from '@/components/processing-status'

export default async function ProcessingPage({ params }: { params: Promise<{ videoId: string }> }) {
  const { videoId } = await params

  return (
    <main className="console-shell">
      <header className="site-header">
        <a className="wordmark" href="/" aria-label="Контур — главная">КОНТУР</a>
        <span className="header-label">Инспекция стройплощадки</span>
        <span className="header-status">Анализ выполняется</span>
      </header>
      <ProcessingStatus videoId={videoId} />
    </main>
  )
}
