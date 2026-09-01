import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { UploadForm } from '@/components/upload-form'

const mockPush = vi.fn()
const mockUpload = vi.fn()
const mockStartAnalysis = vi.fn()

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: mockPush }) }))
vi.mock('@/lib/api', () => ({
  api: {
    upload: (...args: unknown[]) => mockUpload(...args),
    startAnalysis: (...args: unknown[]) => mockStartAnalysis(...args),
  },
}))

describe('UploadForm', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('routes to processing after upload', async () => {
    mockUpload.mockResolvedValue('vid_1')
    mockStartAnalysis.mockResolvedValue({ video_id: 'vid_1', status: 'queued' })
    const user = userEvent.setup()

    render(<UploadForm />)
    await user.upload(screen.getByLabelText('Файл'), new File(['x'], 'site.mp4', { type: 'video/mp4' }))
    await user.click(screen.getByRole('button', { name: 'Загрузить и анализировать' }))

    await waitFor(() => expect(mockPush).toHaveBeenCalledWith('/processing/vid_1'))
  })

  it('keeps the selected file available when upload fails', async () => {
    mockUpload.mockRejectedValue(new Error('network'))
    const user = userEvent.setup()

    render(<UploadForm />)
    await user.upload(screen.getByLabelText('Файл'), new File(['x'], 'site.mp4', { type: 'video/mp4' }))
    await user.click(screen.getByRole('button', { name: 'Загрузить и анализировать' }))

    expect((await screen.findByRole('alert')).textContent).toBe('Не удалось загрузить файл. Повторите попытку.')
    expect((screen.getByLabelText('Файл') as HTMLInputElement).files?.[0]?.name).toBe('site.mp4')
  })
})
