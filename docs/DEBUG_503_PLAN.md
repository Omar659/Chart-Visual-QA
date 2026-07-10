# Debug plan — recurring 503 on `/api/ask` (Cloud Run, 2026-07-10)

## Status of evidence so far (do NOT re-derive; verified via real logs)

Already ruled out (fixed + confirmed deployed, yet 503 persists):
- ~~C-compiler crash in chartqa-vlm~~ — fixed (`build-essential`), revision `chartqa-vlm-00004-nw6` logs `Model ready — serving.`
- ~~MSYS path-mangled `QWEN_ADAPTER_PATH`~~ — fixed (`MSYS2_ARG_CONV_EXCL`), env var verified clean on the live revision.
- ~~Unauthenticated warm ping 403~~ — fixed (`vlm_provider.py` now sends ID tokens).
- ~~HF Hub live network calls at model init~~ — fixed (`HF_HUB_OFFLINE=1`), new boot logs show zero HF calls.
- ~~Flask dev server~~ — backend now runs gunicorn (confirmed in logs: `Starting gunicorn 23.0.0`, `Using worker: gthread`).
- nginx proxy timeout is NOT the weak link (`proxy_read_timeout 300s` in `frontend/nginx.conf.template`).
- Auth works: `/api/ask` without token → 401; user CAN sign in (origin fixed).

**The decisive new evidence** — backend log timeline of the latest failure
(revision `chartqa-backend-00004-cmz`, all fixes deployed):

```
12:21:01  GET 200 /api/vlm/warm              ← sign-in triggered warm
12:21:07  warm ping Read timed out (5s)      ← expected: VLM cold, timeout=5 by design
12:21:19  POST 503 /api/ask                  ← FAILED ~18s in — far too fast for any
                                                configured timeout (VLM_TIMEOUT=120,
                                                gunicorn=180, Cloud Run=300, nginx=300)
12:21:19  Redis connected                     ← same second: instance still COLD-STARTING
12:22:00  presidio/guard models still loading
12:22:19  gunicorn boots AGAIN                ← NEW instance = the serving one DIED
```

Backend code can only return 503 from one line (`app.py`: `ensure_running` false), and
`VLM_PROVIDER=cloudrun` makes `ensure_running()` return True unconditionally — so this
503 is **Cloud Run infrastructure**, generated when the container instance died or was
unreachable mid-request. An instance death immediately after serving is followed by a
fresh gunicorn boot — exactly what the log shows.

## Primary hypothesis — H1: OOM kill during guard-model warmup

Backend deploys with `--memory=2Gi`. `post_worker_init` warmup loads, concurrently with
serving: CLIP (~600MB) + toxic-bert (~450MB) + deberta-v3-base (~750MB fp32) + spaCy
`en_core_web_lg` (~600MB) + CPU torch runtime + Flask/redis/presidio overhead ≈ ~3GB
peak > 2Gi limit → Cloud Run OOM-kills the container mid-request → 503 → replacement
instance boots. Never seen locally because docker-compose imposes no 2Gi cap.

Consistent with EVERY observed 503 so far, including the pre-gunicorn ones (each 503 was
followed by cold-boot logs — previously misread as "the request landed on a cold
instance", but the causality is likely reversed: the request **killed** the instance).

### H1 decisive test (run FIRST — one command, yes/no answer)

```bash
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="chartqa-backend" AND severity>=WARNING AND timestamp>="2026-07-10T11:00:00Z"' \
  --project chartv-qa --format="value(timestamp,textPayload)" --limit 50
```

Cloud Run logs OOM explicitly as
`Memory limit of 2048 MiB exceeded with <N> MiB used. Consider increasing the memory limit`.
If present at ~12:21 → H1 CONFIRMED. Also check the request log entry itself:

```bash
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="chartqa-backend" AND httpRequest.status=503' \
  --project chartv-qa --format=json --limit 5
```

The 503's `textPayload` from Cloud Run's front end states the infra reason (e.g. "The
request failed because either the HTTP response was malformed or connection to the
instance had an error").

### H1 fix (apply only after confirmation)

1. `scripts/gcloud_deploy_app.sh`: backend `--memory=2Gi` → `--memory=4Gi` (1 vCPU
   supports up to 4Gi on Cloud Run; cost impact ≈ zero at demo traffic with min=0 —
   memory is billed only while an instance is up).
2. Add `--cpu-boost` to the backend deploy (extra CPU during cold start — the ~4 model
   loads currently share 1 vCPU with the incoming request).
3. Redeploy backend only (image unchanged — this is config, use
   `gcloud run services update chartqa-backend --memory=4Gi --cpu-boost` for speed, THEN
   mirror the flags into `gcloud_deploy_app.sh` + commit so it's not drift).
4. Re-test: `curl /api/health`, then the user repeats the browser test.

## Secondary hypotheses (check in order if H1 disproved, or as follow-ups after it)

- **H2 — requests land during cold start (readiness gap).** Cloud Run marks the
  container ready as soon as it listens on `$PORT`; gunicorn listens immediately while
  guard models still load in the warmup thread. Even after the OOM fix, the first
  request per cold start will be slow (models lazy-load inline). Fix (separate,
  optional): a `--startup-probe` hitting a new endpoint that returns 200 only once
  `guard.is_available()` shows the models loaded, so Cloud Run doesn't route until warm.
  Trade-off: longer perceived cold start, but no first-request stall.
- **H3 — VLM cold start exceeds VLM_TIMEOUT=120s on the real /predict call.** GPU
  instance provisioning (~30-60s) + 17.5GB model load to VRAM (~60s) + first-request
  bitsandbytes/triton JIT compile can total > 120s. Symptom would be: backend survives
  (no instance death), `/api/ask` returns 500/error after exactly ~120s. Check
  `chartqa-vlm` logs for a `/predict` POST arriving and how long it took; check backend
  logs for a requests.Timeout traceback. Fix: `VLM_TIMEOUT=240` in
  `gcloud_deploy_app.sh` (gunicorn timeout=180 must ALSO rise to ≥ 300 — see
  `backend/gunicorn.conf.py`; keep Cloud Run `--timeout=300`).
- **H4 — first real /predict crashes the VLM** (JIT compile at generate-time, VRAM
  overflow at 4bit + long input). Only reachable once H1/H3 pass. Check `chartqa-vlm`
  logs during a real attempt: a traceback after "Model ready" = H4. Test directly
  without the browser: `gcloud run services proxy chartqa-vlm --port 9090` then POST a
  small image+question to `localhost:9090/predict`.
- **H5 — Upstash Redis latency/misbehavior on the request path.** rediss:// handshake
  per cold instance is fine (log shows it connects), but if fail-open isn't working as
  designed a Redis error could bubble. Low probability (code is fail-open by design and
  tested); only revisit if H1-H4 all pass.

## Execution order

1. Run the two `gcloud logging read` commands (H1 test). — 2 min
2. If OOM confirmed: apply memory/cpu-boost fix via `gcloud run services update`,
   verify with a browser retry, then persist flags in `gcloud_deploy_app.sh` + commit.
3. While the user retests: watch `chartqa-vlm` logs live for the /predict call — this
   pre-checks H3/H4 in the same attempt.
4. If the ask now succeeds end-to-end: update `docs/REVIEW_AND_ROADMAP.md` §3.7 and the
   gcloud-deploy skill gotchas (OOM entry), commit.
5. If it fails differently: the failure mode tells us which of H2-H5 to chase — each has
   its own check above.

## Success criterion

A signed-in browser session uploads a chart, asks a question, and gets a REAL model
answer back (not 503, not disclaimer), confirmed twice: once against a cold stack and
once warm.
