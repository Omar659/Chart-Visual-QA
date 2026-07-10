"""Google Cloud service-to-service ID tokens — the single place this logic lives.

Used by both ``model_adapter.py`` (calling the private GPU VLM service) and
``guard_llm.py`` (calling the private guard service) so neither carries its own copy of
the metadata-server call. Both services are deployed WITHOUT
``--allow-unauthenticated`` — Cloud Run's own infra verifies the token before a request
ever reaches the container, so an expensive/sensitive backend service can't be hit
directly by the public internet, only by whichever identity holds ``run.invoker`` on it
(this backend's own service account, granted by ``scripts/gcloud_deploy_app.sh``).

No extra dependency: the token comes from the Cloud Run instance's metadata server, which
only exists when actually running on Cloud Run (or GCE/GKE) — this is never reachable
from a laptop, RunPod, or docker-compose, which is exactly where callers pass ``mode="none"``.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import requests

_METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/identity"
)


def auth_header(url: str, mode: str) -> dict[str, str]:
    """Authorization header for a call to ``url``, per ``mode``.

    ``none``: no header — the target is reachable without auth (RunPod tunnel,
    docker-compose, a big-GPU dev box).
    ``gcp_id_token``: fetch a Google-signed ID token (audience = ``url``'s own
    scheme+host) from the metadata server and send it as a Bearer token.
    """
    if mode.strip().lower() != "gcp_id_token":
        return {}
    return {"Authorization": f"Bearer {fetch_id_token(url)}"}


def fetch_id_token(url: str) -> str:
    """Fetch a Google-signed ID token whose audience is ``url``'s scheme+host."""
    parsed = urlsplit(url)
    audience = f"{parsed.scheme}://{parsed.netloc}"
    resp = requests.get(
        _METADATA_URL,
        params={"audience": audience},
        headers={"Metadata-Flavor": "Google"},
        timeout=5,
    )
    resp.raise_for_status()
    return resp.text
