// Thin client for the backend API. Calls go to same-origin /api/* and the Vite
// dev server proxies them to Flask (see vite.config.js).

export async function askQuestion(imageFile, question) {
  const form = new FormData()
  form.append('image', imageFile)
  form.append('question', question)

  let res
  try {
    res = await fetch('/api/ask', { method: 'POST', body: form })
  } catch {
    throw new Error('Could not reach the server. Is the backend running?')
  }

  let data = {}
  try {
    data = await res.json()
  } catch {
    // non-JSON response (e.g. 413 from the upload limit) — fall through
  }

  if (!res.ok) {
    throw new Error(data.error || `Request failed (${res.status}).`)
  }
  return data // { answer, mock, latency_ms }
}

export async function getHealth() {
  const res = await fetch('/api/health')
  if (!res.ok) throw new Error('Health check failed.')
  return res.json() // { status, mock }
}
