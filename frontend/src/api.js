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
