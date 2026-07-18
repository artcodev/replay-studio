from __future__ import annotations

"""HTTP transport for the identity worker; no batching or domain policy."""

import httpx

from .identity_worker_contract import IdentityWorkerError


class IdentityWorkerTransportError(IdentityWorkerError):
    pass


def fetch_identity_readiness(worker_url: str, *, timeout: float) -> object:
    try:
        response = httpx.get(
            f"{worker_url.rstrip('/')}/health/ready",
            timeout=max(0.1, float(timeout)),
        )
        response.raise_for_status()
        return response.json()
    except (OSError, ValueError, httpx.HTTPError) as exc:
        raise IdentityWorkerTransportError(str(exc)) from exc


def post_identity_batch(
    worker_url: str,
    *,
    files: list[tuple[str, tuple[str, bytes, str]]],
    manifest: str,
    timeout: float,
) -> object:
    try:
        response = httpx.post(
            f"{worker_url.rstrip('/')}/v1/embeddings",
            data={"manifest": manifest},
            files=files,
            timeout=float(timeout),
        )
        response.raise_for_status()
        return response.json()
    except (OSError, ValueError, httpx.HTTPError) as exc:
        raise IdentityWorkerTransportError(str(exc)) from exc
