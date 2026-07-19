from __future__ import annotations

"""Stable, content-based identities for local model checkpoints."""

from hashlib import sha256
from pathlib import Path
from threading import Lock


_digest_cache: dict[tuple[str, int, int, int], str] = {}
_digest_lock = Lock()


def file_content_sha256(path: str | Path) -> str:
    checkpoint = Path(path).expanduser().resolve()
    stat = checkpoint.stat()
    cache_key = (
        str(checkpoint),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )
    with _digest_lock:
        cached = _digest_cache.get(cache_key)
    if cached is not None:
        return cached

    digest = sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()

    final_stat = checkpoint.stat()
    final_key = (
        str(checkpoint),
        int(final_stat.st_size),
        int(final_stat.st_mtime_ns),
        int(final_stat.st_ctime_ns),
    )
    if final_key != cache_key:
        raise RuntimeError(f"Checkpoint changed while it was being hashed: {checkpoint}")

    with _digest_lock:
        stale_keys = [key for key in _digest_cache if key[0] == str(checkpoint)]
        for stale_key in stale_keys:
            _digest_cache.pop(stale_key, None)
        _digest_cache[cache_key] = value
    return value


def checkpoint_content_identity(path: str | Path) -> dict:
    checkpoint = Path(path).expanduser().resolve()
    identity: dict = {"name": checkpoint.name}
    if checkpoint.is_file():
        stat = checkpoint.stat()
        identity.update(
            {
                "size": int(stat.st_size),
                "sha256": file_content_sha256(checkpoint),
            }
        )
    return identity


__all__ = ("checkpoint_content_identity", "file_content_sha256")
