import { useEffect, useRef, useState } from 'react'
import { askQuestion, getHealth } from './api'
import './App.css'

const MAX_BYTES = 10 * 1024 * 1024 // keep in sync with backend MAX_CONTENT_LENGTH

function App() {
  const [image, setImage] = useState(null) // File
  const [previewUrl, setPreviewUrl] = useState('')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(null) // { answer, mock, latency_ms }
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [mockBanner, setMockBanner] = useState(false)
  const fileInputRef = useRef(null)

  // Probe the backend once so we can show a "mock mode" hint.
  useEffect(() => {
    getHealth()
      .then((h) => setMockBanner(Boolean(h.mock)))
      .catch(() => {}) // health failure is non-fatal for the UI
  }, [])

  // Revoke object URLs when they change/unmount to avoid leaks.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  function selectImage(file) {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError('Please choose an image file.')
      return
    }
    if (file.size > MAX_BYTES) {
      setError('Image is larger than 10 MB.')
      return
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setImage(file)
    setPreviewUrl(URL.createObjectURL(file))
    setAnswer(null)
    setError('')
  }

  function onFileChange(e) {
    selectImage(e.target.files?.[0])
  }

  function onDrop(e) {
    e.preventDefault()
    selectImage(e.dataTransfer.files?.[0])
  }

  async function onSubmit(e) {
    e.preventDefault()
    setError('')
    setAnswer(null)
    if (!image) return setError('Pick a chart image first.')
    if (!question.trim()) return setError('Type a question.')

    setLoading(true)
    try {
      const result = await askQuestion(image, question.trim())
      setAnswer(result)
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  const canSubmit = image && question.trim() && !loading

  return (
    <div className="app">
      <header className="header">
        <h1>Chart&nbsp;VQA</h1>
        <p className="subtitle">Ask a question about a chart, get a short answer.</p>
        {mockBanner && (
          <span className="badge" title="The backend is returning mock answers.">
            mock mode
          </span>
        )}
      </header>

      <form className="card" onSubmit={onSubmit}>
        {/* Image picker / dropzone */}
        <button
          type="button"
          className={`picker ${previewUrl ? 'has-image' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          aria-label="Choose a chart image"
        >
          {previewUrl ? (
            <img className="preview" src={previewUrl} alt="Selected chart preview" />
          ) : (
            <span className="picker-empty">
              <svg className="picker-icon" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  fill="currentColor"
                  d="M21 19V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2M8.5 11l2.5 3 3.5-4.5L19 17H5z"
                />
              </svg>
              <span className="picker-text">Click to choose a chart image</span>
              <span className="picker-hint">or drag &amp; drop · PNG/JPG · up to 10&nbsp;MB</span>
            </span>
          )}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={onFileChange}
        />
        {image && (
          <p className="filename" title={image.name}>
            {image.name} · {(image.size / 1024).toFixed(0)} KB
          </p>
        )}

        {/* Question field */}
        <label className="field">
          <span className="label">Question</span>
          <input
            type="text"
            className="text-input"
            placeholder="e.g. What was the revenue in 2024?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
        </label>

        <button type="submit" className="submit" disabled={!canSubmit}>
          {loading ? 'Thinking…' : 'Ask'}
        </button>

        {error && <p className="error" role="alert">{error}</p>}

        {answer && (
          <div className="answer" aria-live="polite">
            <span className="answer-label">Answer</span>
            <span className="answer-text">{answer.answer}</span>
            <span className="answer-meta">
              {answer.mock ? 'mock · ' : ''}
              {Number(answer.latency_ms).toFixed(0)} ms
            </span>
          </div>
        )}
      </form>
    </div>
  )
}

export default App
