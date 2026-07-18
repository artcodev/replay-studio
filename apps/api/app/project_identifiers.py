from __future__ import annotations

"""Deterministic internal identifiers shared by project-domain repositories."""

import hashlib


def stable_identifier(prefix: str, *parts: object, length: int = 24) -> str:
    material = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{digest}"
