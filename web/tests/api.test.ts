import { afterEach, describe, expect, expectTypeOf, it, vi } from 'vitest'

import { api } from '@/lib/api'
import type { Finding, Report } from '@/lib/types'

describe('API client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('uploads a video with a project name', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ video_id: 'vid_1' })))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.upload(new File(['x'], 'clip.mp4'), 'ЖК Тест', false)).resolves.toBe('vid_1')

    expect(fetchMock).toHaveBeenCalledWith('/api/videos/upload', expect.objectContaining({ method: 'POST' }))
    expect((fetchMock.mock.calls[0]?.[1] as RequestInit).body).toBeInstanceOf(FormData)
    expect(((fetchMock.mock.calls[0]?.[1] as RequestInit).body as FormData).get('project_name')).toBe('ЖК Тест')
  })

  it('uses the photo upload endpoint for photos', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ video_id: 'vid_photo' })))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.upload(new File(['x'], 'site.jpg'), '', true)).resolves.toBe('vid_photo')
    expect(fetchMock).toHaveBeenCalledWith('/api/photos/upload', expect.objectContaining({ method: 'POST' }))
  })

  it('starts analysis for an uploaded video', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ video_id: 'vid_1', status: 'queued' })))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.startAnalysis('vid_1')).resolves.toEqual({ video_id: 'vid_1', status: 'queued' })
    expect(fetchMock).toHaveBeenCalledWith('/api/videos/vid_1/analyze', { method: 'POST' })
  })

  it('gets a report for a video', async () => {
    const report = { video_id: 'vid_1', findings: [] }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(report)))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.getReport('vid_1')).resolves.toMatchObject(report)
    expect(fetchMock).toHaveBeenCalledWith('/api/videos/vid_1/report')
  })

  it('lists uploaded videos', async () => {
    const videos = [{ id: 'vid_1', filename: 'clip.mp4', project_name: '', status: 'done', is_photo: false, created_at: '2026-09-01T00:00:00+00:00' }]
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(videos)))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.listVideos()).resolves.toEqual(videos)
    expect(fetchMock).toHaveBeenCalledWith('/api/videos')
  })

  it('builds same-origin API URLs by default', () => {
    expect(api.url('/api/videos')).toBe('/api/videos')
  })

  it('models report timeline and finding status as backend literals', () => {
    expectTypeOf<Report['activity_timeline']>().toEqualTypeOf<Array<{ from_ms: number; to_ms: number; activity: string }>>()
    expectTypeOf<Finding['status']>().toEqualTypeOf<'unreviewed'>()
  })
})
