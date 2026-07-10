// Thin client for the backend API. Calls go to same-origin /api/* and the Vite
// dev server proxies them to Flask (see vite.config.js).

// `token` is a Google ID token (see App.jsx's sign-in flow); omitted/null when the
// backend has no login wall (AUTH_ENABLED=0 — local/--dev, matches VITE_GOOGLE_CLIENT_ID
// being unset there too).
function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function askQuestion(imageFile, question, { signal, token } = {}) {
  const form = new FormData()
  form.append('image', imageFile)
  form.append('question', question)

  let res
  try {
    res = await fetch('/api/ask', {
      method: 'POST', body: form, signal, headers: authHeaders(token),
    })
  } catch (err) {
    if (err.name === 'AbortError') throw err // caller decides what to do
    throw new Error('Could not reach the server. Is the backend running?')
  }

  let data = {}
  try {
    data = await res.json()
  } catch {
    // non-JSON response (e.g. 413 from the upload limit) — fall through
  }

  if (!res.ok) {
    if (res.status === 401) {
      const err = new Error(data.error || 'Please sign in again.')
      err.authExpired = true // App.jsx clears the stored token and re-prompts sign-in
      throw err
    }
    if (res.status === 413) {
      throw new Error('That image is too large (max 10 MB).')
    }
    throw new Error(data.error || `Request failed (${res.status}).`)
  }
  return data // { answer, mock, latency_ms }
}

// Same pipeline/contract as askQuestion, but consumes /api/ask/stream (Server-Sent
// Events) so the caller gets real per-stage progress instead of one opaque wait.
// `onEvent` is called once per SSE event: {stage, status, elapsed_ms} for progress,
// or {stage: "result", body, status_code} exactly once at the end. Browsers' native
// EventSource only supports GET, so this uses fetch() + a manual stream reader
// instead — the only way to POST (the image) and still stream the response.
export async function askQuestionStream(imageFile, question, { signal, token, onEvent } = {}) {
  const form = new FormData()
  form.append('image', imageFile)
  form.append('question', question)

  let res
  try {
    res = await fetch('/api/ask/stream', {
      method: 'POST', body: form, signal, headers: authHeaders(token),
    })
  } catch (err) {
    if (err.name === 'AbortError') throw err
    throw new Error('Could not reach the server. Is the backend running?')
  }

  if (!res.ok) {
    // The stream never started (e.g. a 401 from an expired token, before any SSE
    // bytes were written) — same error handling as the non-streaming path.
    let data = {}
    try {
      data = await res.json()
    } catch {
      // non-JSON response — fall through to the generic error below
    }
    if (res.status === 401) {
      const err = new Error(data.error || 'Please sign in again.')
      err.authExpired = true
      throw err
    }
    if (res.status === 413) throw new Error('That image is too large (max 10 MB).')
    throw new Error(data.error || `Request failed (${res.status}).`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finalEvent = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n\n')
    buffer = lines.pop() // last chunk may be incomplete — keep it for the next read
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const event = JSON.parse(line.slice('data: '.length))
      if (event.stage === 'result') finalEvent = event
      else onEvent?.(event)
    }
  }

  if (!finalEvent) throw new Error('Connection lost before the answer arrived.')
  if (finalEvent.status_code === 401) {
    const err = new Error(finalEvent.body.error || 'Please sign in again.')
    err.authExpired = true
    throw err
  }
  if (finalEvent.status_code >= 400) {
    throw new Error(finalEvent.body.error || `Request failed (${finalEvent.status_code}).`)
  }
  return finalEvent.body // same shape as askQuestion's return value
}

export async function getHealth() {
  const res = await fetch('/api/health')
  if (!res.ok) throw new Error('Health check failed.')
  return res.json() // { status, mock }
}

// Fire-and-forget nudge for the remote VLM (RunPod dev pod / Cloud Run instance) so it
// has a head start before the user's first real question. Never throws — a failed warm
// ping just means the first /api/ask absorbs the full cold-start latency instead.
export function warmVlm(token) {
  fetch('/api/vlm/warm', { headers: authHeaders(token) }).catch(() => {})
}
