import type { PendingReport, Report, Video, VideoStatus } from '@/lib/types'

const base = process.env.NEXT_PUBLIC_API_BASE_URL ?? ''

export const api = {
  url: (path: string) => `${base}${path}`,

  async upload(file: File, projectName: string, isPhoto: boolean): Promise<string> {
    const body = new FormData()
    body.set('file', file)
    body.set('project_name', projectName)
    const response = await fetch(api.url(isPhoto ? '/api/photos/upload' : '/api/videos/upload'), {
      method: 'POST',
      body,
    })
    if (!response.ok) throw new Error('Не удалось загрузить файл')
    return (await response.json() as { video_id: string }).video_id
  },

  async startAnalysis(videoId: string): Promise<{ video_id: string; status: VideoStatus }> {
    const response = await fetch(api.url(`/api/videos/${videoId}/analyze`), { method: 'POST' })
    if (!response.ok) throw new Error('Не удалось запустить анализ')
    return await response.json() as { video_id: string; status: VideoStatus }
  },

  async getReport(videoId: string): Promise<Report | PendingReport> {
    const response = await fetch(api.url(`/api/videos/${videoId}/report`))
    if (!response.ok && response.status !== 202) throw new Error('Не удалось получить отчёт')
    return await response.json() as Report | PendingReport
  },

  async listVideos(): Promise<Video[]> {
    const response = await fetch(api.url('/api/videos'))
    if (!response.ok) throw new Error('Не удалось получить список проверок')
    return await response.json() as Video[]
  },
}
