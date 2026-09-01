import { afterEach, describe, expect, it, vi } from 'vitest'

import { api } from '@/lib/api'

describe('API client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('uploads a video with a project name', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ video_id: 'vid_1' })))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.upload(new File(['x'], 'clip.mp4'), 'ЖК Тест', false)).resolves.toBe('vid_1')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/api/videos/upload',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('uses the photo upload endpoint for photos', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ video_id: 'vid_photo' })))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.upload(new File(['x'], 'site.jpg'), '', true)).resolves.toBe('vid_photo')
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/api/photos/upload',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('starts analysis for an uploaded video', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ video_id: 'vid_1', status: 'queued' }))))

    await expect(api.startAnalysis('vid_1')).resolves.toEqual({ video_id: 'vid_1', status: 'queued' })
  })

  it('gets a report for a video', async () => {
    const report = { video_id: 'vid_1', findings: [] }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(report))))

    await expect(api.getReport('vid_1')).resolves.toMatchObject(report)
  })

  it('lists uploaded videos', async () => {
    const videos = [{ id: 'vid_1', filename: 'clip.mp4', project_name: '', status: 'done', is_photo: false, created_at: '2026-09-01T00:00:00+00:00' }]
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(videos))))

    await expect(api.listVideos()).resolves.toEqual(videos)
  })

  it('builds an API URL from the local development base', () => {
    expect(api.url('/api/videos')).toBe('http://localhost:8000/api/videos')
  })
})
