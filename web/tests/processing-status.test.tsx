import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'

import { ProcessingStatus } from '@/components/processing-status'

const mockPush = vi.fn()
const sources: MockEventSource[] = []

class MockEventSource {
  onmessage: ((event: MessageEvent<string>) => void) | null = null
  onerror: (() => void) | null = null
  close = vi.fn()

  constructor(readonly url: string) {
    sources.push(this)
  }
}

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: mockPush }) }))
vi.mock('@/lib/api', () => ({ api: { url: (path: string) => path } }))

describe('ProcessingStatus', () => {
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.clearAllMocks()
    sources.length = 0
  })

  it('shows progress and routes to the report when processing completes', async () => {
    vi.stubGlobal('EventSource', MockEventSource)
    render(<ProcessingStatus videoId="vid_1" />)

    const source = sources[0]
    expect(source?.url).toBe('/api/videos/vid_1/status')
    source?.onmessage?.({ data: JSON.stringify({ status: 'analyzing', progress_pct: 50, progress_note: 'Проверяем кадры', error: '' }) } as MessageEvent<string>)
    expect(await screen.findByText('Проверяем кадры')).toBeTruthy()
    expect(screen.getByText('50%')).toBeTruthy()

    source?.onmessage?.({ data: JSON.stringify({ status: 'done', progress_pct: 100, progress_note: 'Готово', error: '' }) } as MessageEvent<string>)
    await waitFor(() => expect(mockPush).toHaveBeenCalledWith('/report/vid_1'))
    expect(source?.close).toHaveBeenCalled()
  })

  it('offers a retry after the status connection drops', async () => {
    vi.stubGlobal('EventSource', MockEventSource)
    render(<ProcessingStatus videoId="vid_1" />)

    sources[0]?.onerror?.()

    expect((await screen.findByRole('alert')).textContent).toBe('Потеряна связь со статусом. Повторите попытку.')
    expect(screen.getByRole('button', { name: 'Повторить подключение' })).toBeTruthy()
  })

  it('closes on a failed terminal status without offering a connection retry', async () => {
    vi.stubGlobal('EventSource', MockEventSource)
    render(<ProcessingStatus videoId="vid_1" />)

    const source = sources[0]
    source?.onmessage?.({ data: JSON.stringify({ status: 'failed', progress_pct: 60, progress_note: 'Остановка', error: 'Сервер не смог завершить анализ' }) } as MessageEvent<string>)
    source?.onerror?.()

    expect(await screen.findByText('Сервер не смог завершить анализ')).toBeTruthy()
    expect(source?.close).toHaveBeenCalled()
    expect(screen.queryByRole('button', { name: 'Повторить подключение' })).toBeNull()
    expect(mockPush).not.toHaveBeenCalled()
  })
})
