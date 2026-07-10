import { useEffect, useRef, useState } from 'react'
import { askQuestionStream, getHealth, warmVlm } from './api'
import './App.css'

const MAX_BYTES = 10 * 1024 * 1024 // keep in sync with backend MAX_CONTENT_LENGTH

// Per-stage progress rows, in the order they should appear. Keyed by the backend's
// SSE `stage` name (app.py's _ask_events) — vlm_start is intentionally omitted (a
// near-instant no-op for VLM_PROVIDER=cloudrun; showing it would just flicker).
const STAGE_LABELS = [
  { stage: 'chart_gate', label: 'Verifying image' },
  { stage: 'guard', label: 'Verifying question' },
  { stage: 'vlm', label: 'Processing model' },
]

// Baked in at BUILD time (Vite's import.meta.env is compile-time, not runtime — see
// frontend/cloudbuild.yaml / Dockerfile). Empty in local dev by default: no login wall,
// matches the backend's AUTH_ENABLED=0 default (see docs/REVIEW_AND_ROADMAP.md §3.7).
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || ''
const HAS_AUTH = Boolean(GOOGLE_CLIENT_ID)
const TOKEN_STORAGE_KEY = 'chartqa_gid_token'

// Mirror of the backend's _question_too_weak guard for instant feedback.
// Count letters/digits in any language (so CJK questions pass), reject junk.
function questionTooWeak(q) {
  const meaningful = (q.match(/[\p{L}\p{N}]/gu) || []).length
  return meaningful < 3
}

function App() {
  const [image, setImage] = useState(null) // File
  const [previewUrl, setPreviewUrl] = useState('')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(null) // { answer, mock, latency_ms, is_chart, chart_confidence }
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  // Per-stage progress, keyed by SSE stage name: { status: 'start'|'done', elapsed_ms }.
  // A stage key only appears once its "start" event has arrived, so the UI reveals rows
  // as the pipeline actually reaches them instead of showing all three upfront.
  const [stages, setStages] = useState({})
  const [mockBanner, setMockBanner] = useState(false)
  // Google ID token (Phase 3.7 required sign-in). null when signed out; also null
  // permanently when HAS_AUTH is false (no login wall configured for this deploy).
  // sessionStorage (not localStorage): survives a refresh, cleared when the tab closes.
  const [token, setToken] = useState(() =>
    HAS_AUTH ? sessionStorage.getItem(TOKEN_STORAGE_KEY) : null
  )
  const fileInputRef = useRef(null)
  const abortRef = useRef(null)
  const signinButtonRef = useRef(null)

  function handleSignedIn(idToken) {
    setToken(idToken)
    sessionStorage.setItem(TOKEN_STORAGE_KEY, idToken)
    warmVlm(idToken) // nudge the remote VLM right after sign-in, not before
  }

  function handleSignOut() {
    setToken(null)
    sessionStorage.removeItem(TOKEN_STORAGE_KEY)
    window.google?.accounts?.id?.disableAutoSelect()
  }

  // Probe the backend once so we can show a "mock mode" status pill. When there's no
  // login wall (HAS_AUTH false), warm the VLM on load like before; with a login wall,
  // warming happens on sign-in instead (handleSignedIn) — no reason to wake a billed
  // GPU for a visitor who hasn't authenticated yet.
  useEffect(() => {
    getHealth()
      .then((h) => {
        setMockBanner(Boolean(h.mock))
        if (!h.mock && !HAS_AUTH) warmVlm()
      })
      .catch(() => {}) // health failure is non-fatal for the UI
  }, [])

  // Render the Google "Sign in" button once its script has loaded and we're signed out.
  useEffect(() => {
    if (!HAS_AUTH || token) return
    let cancelled = false
    function tryRender() {
      if (cancelled) return
      if (window.google?.accounts?.id && signinButtonRef.current) {
        window.google.accounts.id.initialize({
          client_id: GOOGLE_CLIENT_ID,
          callback: (response) => handleSignedIn(response.credential),
        })
        window.google.accounts.id.renderButton(signinButtonRef.current, {
          theme: 'outline', size: 'large', text: 'signin_with',
        })
      } else {
        setTimeout(tryRender, 100) // GIS script loads async — poll briefly until ready
      }
    }
    tryRender()
    return () => {
      cancelled = true
    }
  }, [token])

  // Revoke object URLs when they change/unmount to avoid leaks.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  // Abort any in-flight request if the component unmounts.
  useEffect(() => {
    return () => abortRef.current?.abort()
  }, [])

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
    setStages({})
    const q = question.trim()
    if (!image) return setError('Please upload an image.')
    if (!q) return setError('Please type a question.')
    if (questionTooWeak(q)) return setError('Please ask a more specific question.')

    // Cancel any in-flight request so a fast resubmit can't race.
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setLoading(true)
    try {
      const result = await askQuestionStream(image, q, {
        signal: controller.signal,
        token,
        onEvent: (event) => {
          setStages((prev) => ({
            ...prev,
            [event.stage]: { status: event.status, elapsed_ms: event.elapsed_ms },
          }))
        },
      })
      if (result.blocked) {
        // Guard (Layer 2/3) or the chart gate rejected the request.
        setError(result.reason || 'That question was blocked.')
        return
      }
      setAnswer(result)
    } catch (err) {
      if (err.name === 'AbortError') return // superseded by a newer request
      if (err.authExpired) {
        // Session expired mid-visit (Google ID tokens last ~1h) — drop it and let the
        // sign-in gate reappear instead of showing a confusing generic error.
        handleSignOut()
        setError('Your session expired — please sign in again.')
        return
      }
      setError(err.message || 'Something went wrong.')
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
        setLoading(false)
      }
    }
  }

  const canSubmit = image && question.trim() && !loading

  return (
    <div className="page">
      <nav className="nav-bar">
        <span className="brand">
          <span className="brand-glyph" aria-hidden="true">⚡</span>
          <span className="brand-name">Chart&nbsp;VQA</span>
        </span>
        <span className={`status-pill ${mockBanner ? 'is-mock' : 'is-live'}`}>
          <span className="status-dot" aria-hidden="true" />
          {mockBanner ? 'mock backend' : 'live'}
        </span>
        {HAS_AUTH && token && (
          <button type="button" className="signout" onClick={handleSignOut}>
            Sign out
          </button>
        )}
      </nav>

      <main className="container">
        <header className="hero">
          <p className="eyebrow">CHART QUESTION ANSWERING</p>
          <h1 className="hero-title">Ask a question about a chart.</h1>
          <p className="hero-sub">
            Upload a chart, type a question, get a short answer. Powered by a
            vision-language model behind <code className="chip">POST /api/ask</code>.
          </p>
        </header>

        {HAS_AUTH && !token ? (
          <div className="card signin-card">
            <p className="signin-prompt">Sign in with Google to ask a question.</p>
            <div ref={signinButtonRef} />
          </div>
        ) : (
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
            <p className="filename">
              <code className="chip">{image.name}</code>
              <span className="filesize">{(image.size / 1024).toFixed(0)} KB</span>
            </p>
          )}

          {/* Question field — disabled until an image is uploaded */}
          <label className="field">
            <span className="label">Question</span>
            <input
              type="text"
              className="text-input"
              placeholder={
                image
                  ? 'e.g. What was the revenue in 2024?'
                  : 'Upload a chart image first…'
              }
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={!image || loading}
            />
          </label>

          <button type="submit" className="submit" disabled={!canSubmit}>
            {loading ? (
              <span className="submit-loading">
                <span className="spinner" aria-hidden="true" />
                Processing…
              </span>
            ) : (
              'Ask'
            )}
          </button>

          {loading && (
            <ul className="stage-list" role="status" aria-live="polite">
              {STAGE_LABELS.filter(({ stage }) => stages[stage]).map(({ stage, label }) => {
                const s = stages[stage]
                const done = s.status === 'done'
                return (
                  <li key={stage} className={`stage-row ${done ? 'stage-done' : ''}`}>
                    {done ? (
                      <span className="stage-check" aria-hidden="true">✓</span>
                    ) : (
                      <span className="spinner stage-spinner" aria-hidden="true" />
                    )}
                    <span className="stage-label">{label}…</span>
                    {done && (
                      <span className="stage-time">{Math.round(s.elapsed_ms)} ms</span>
                    )}
                  </li>
                )
              })}
            </ul>
          )}

          {error && (
            <p className="notice" role="alert">
              <span className="notice-dot" aria-hidden="true" />
              {error}
            </p>
          )}

          {!loading && answer && (
            <div className="result" aria-live="polite">
              {/* Chart-detection indicator — shows the CLIP result on every upload */}
              {typeof answer.chart_confidence === 'number' &&
                (answer.is_chart ? (
                  <p className="detect detect-ok">
                    ✓ Chart detected · {Math.round(answer.chart_confidence * 100)}%
                  </p>
                ) : (
                  <p className="detect detect-warn" role="alert">
                    ⚠️ This doesn&apos;t look like a chart (
                    {Math.round(answer.chart_confidence * 100)}%) — results may be
                    unreliable.
                  </p>
                ))}

              {answer.disclaimer ? (
                <div className="disclaimer">
                  <span className="disclaimer-label">Mock mode</span>
                  <span className="disclaimer-text">{answer.disclaimer}</span>
                  <span className="answer-meta">
                    {Number(answer.latency_ms).toFixed(0)} ms
                  </span>
                </div>
              ) : (
                <div className="answer">
                  <span className="answer-label">ANSWER</span>
                  <span className="answer-text">{answer.answer}</span>
                  <span className="answer-meta">
                    {answer.mock ? 'mock · ' : ''}
                    {Number(answer.latency_ms).toFixed(0)} ms
                  </span>
                </div>
              )}
            </div>
          )}
        </form>
        )}
      </main>
    </div>
  )
}

export default App
