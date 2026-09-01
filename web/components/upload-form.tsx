'use client'

import { useState, type FormEvent } from 'react'
import { useRouter } from 'next/navigation'

import { api } from '@/lib/api'

export function UploadForm() {
  const router = useRouter()
  const [file, setFile] = useState<File | null>(null)
  const [projectName, setProjectName] = useState('')
  const [isPhoto, setIsPhoto] = useState(false)
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
    try {
      const videoId = await api.upload(file, projectName, isPhoto)
      await api.startAnalysis(videoId)
      router.push(`/processing/${videoId}`)
    } catch {
      setError('Не удалось загрузить файл. Повторите попытку.')
      setSubmitting(false)
    }
  }

  return (
    <form className="upload-form" onSubmit={submit}>
      <fieldset className="media-selector">
        <legend>Материал</legend>
        <label><input checked={!isPhoto} name="media-type" onChange={() => setIsPhoto(false)} type="radio" /> Видео</label>
        <label><input checked={isPhoto} name="media-type" onChange={() => setIsPhoto(true)} type="radio" /> Фото</label>
      </fieldset>

      <label className="form-field" htmlFor="project-name">
        <span>Проект <small>необязательно</small></span>
        <input id="project-name" name="project-name" onChange={(event) => setProjectName(event.target.value)} placeholder="Например, ЖК Северный" value={projectName} />
      </label>

      <label className="file-field" htmlFor="media-file">
        <span>Файл</span>
        <input accept={isPhoto ? 'image/*' : 'video/*'} aria-label="Файл" id="media-file" name="file" onChange={(event) => setFile(event.target.files?.[0] ?? null)} type="file" />
        <strong>{file?.name ?? 'Выберите файл с устройства'}</strong>
      </label>

      {error && <p className="form-error" role="alert">{error}</p>}

      <button className="action-button" disabled={submitting} type="submit">
        {submitting ? 'Запускаем анализ…' : 'Загрузить и анализировать'}
      </button>
    </form>
  )
}
