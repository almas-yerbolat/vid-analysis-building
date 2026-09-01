'use client'

import { useState, type FormEvent } from 'react'
import { useRouter } from 'next/navigation'

import { api } from '@/lib/api'

export function UploadForm() {
  const router = useRouter()
  const [file, setFile] = useState<File | null>(null)
  const [projectName, setProjectName] = useState('')
  const [isPhoto, setIsPhoto] = useState(false)
  const [uploadedVideoId, setUploadedVideoId] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!file) {
      setError('Выберите файл для проверки.')
      return
    }

    setError('')
    setSubmitting(true)
    let videoId = uploadedVideoId
    try {
      if (!videoId) {
        videoId = await api.upload(file, projectName, isPhoto)
        setUploadedVideoId(videoId)
      }
      await api.startAnalysis(videoId)
      router.push(`/processing/${videoId}`)
    } catch {
      setError(videoId ? 'Не удалось запустить анализ. Повторите попытку.' : 'Не удалось загрузить файл. Повторите попытку.')
      setSubmitting(false)
    }
  }

  return (
    <form className="upload-form" onSubmit={submit}>
      <fieldset className="media-selector">
        <legend>Материал</legend>
        <label><input checked={!isPhoto} name="media-type" onChange={() => { setIsPhoto(false); setUploadedVideoId(null) }} type="radio" /> Видео</label>
        <label><input checked={isPhoto} name="media-type" onChange={() => { setIsPhoto(true); setUploadedVideoId(null) }} type="radio" /> Фото</label>
      </fieldset>

      <label className="form-field" htmlFor="project-name">
        <span>Проект <small>необязательно</small></span>
        <input id="project-name" name="project-name" onChange={(event) => { setProjectName(event.target.value); setUploadedVideoId(null) }} placeholder="Например, ЖК Северный" value={projectName} />
      </label>

      <label className="file-field" htmlFor="media-file">
        <span>Файл</span>
        <input accept={isPhoto ? 'image/*' : 'video/*'} aria-label="Файл" id="media-file" name="file" onChange={(event) => { setFile(event.target.files?.[0] ?? null); setUploadedVideoId(null) }} type="file" />
        <strong>{file?.name ?? 'Выберите файл с устройства'}</strong>
      </label>

      {error && <p className="form-error" role="alert">{error}</p>}

      <button className="action-button" disabled={submitting} type="submit">
        {submitting ? 'Запускаем анализ…' : 'Загрузить и анализировать'}
      </button>
    </form>
  )
}
