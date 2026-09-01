'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'

import { api } from '@/lib/api'
import type { StatusEvent, VideoStatus } from '@/lib/types'

const stages: Array<{ status: VideoStatus; label: string }> = [
  { status: 'queued', label: 'В очереди' },
  { status: 'probing', label: 'Проверяем файл' },
  { status: 'sampling', label: 'Отбираем кадры' },
  { status: 'analyzing', label: 'Анализируем объект' },
  { status: 'aggregating', label: 'Собираем отчёт' },
]

const initialStatus: StatusEvent = { status: 'queued', progress_pct: 0, progress_note: 'Подключаемся к статусу проверки', error: '' }

export function ProcessingStatus({ videoId }: { videoId: string }) {
  const router = useRouter()
  const [status, setStatus] = useState<StatusEvent>(initialStatus)
  const [error, setError] = useState('')
  const [connection, setConnection] = useState(0)

  useEffect(() => {
    let closed = false
    const source = new EventSource(api.url(`/api/videos/${videoId}/status`))

    source.onmessage = ({ data }) => {
      try {
        const event = JSON.parse(data) as StatusEvent
        setStatus(event)
        setError('')
        if (event.status === 'done') {
          source.close()
          router.push(`/report/${videoId}`)
        }
      } catch {
        setError('Не удалось прочитать статус проверки. Повторите подключение.')
      }
    }
    source.onerror = () => {
      if (!closed) {
        source.close()
        setError('Потеряна связь со статусом. Повторите попытку.')
      }
    }

    return () => {
      closed = true
      source.close()
    }
  }, [connection, router, videoId])

  const failed = status.status === 'failed'
  const currentStage = stages.findIndex((stage) => stage.status === status.status)

  return (
    <section className="processing-panel" aria-labelledby="processing-title">
      <p className="eyebrow">ИДЁТ ПРОВЕРКА</p>
      <h1 id="processing-title">Проверяем материал</h1>
      <div className="progress-reading">
        <strong>{Math.round(status.progress_pct)}%</strong>
        <span>{status.progress_note}</span>
      </div>
      <progress aria-label="Ход проверки" max="100" value={status.progress_pct} />

      <ol className="processing-stages">
        {stages.map((stage, index) => (
          <li className={index <= currentStage ? 'is-complete' : ''} key={stage.status}>{stage.label}</li>
        ))}
      </ol>

      {failed && <p className="form-error" role="alert">{status.error || 'Анализ не удалось завершить.'}</p>}
      {error && (
        <div className="connection-error">
          <p role="alert">{error}</p>
          <button className="retry-button" onClick={() => setConnection((value) => value + 1)} type="button">Повторить подключение</button>
        </div>
      )}
    </section>
  )
}
