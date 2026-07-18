from __future__ import annotations

from hashlib import md5, sha256
from pathlib import Path

import numpy as np


ROLE_NAMES = ("ball", "goalkeeper", "other", "player", "referee")


def role_evidence_from_logits(raw_scores: object) -> tuple[str | None, float | None]:
    """Convert auxiliary PRTReID logits into bounded role evidence."""

    try:
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None, None
    if scores.size == 0 or not np.isfinite(scores).all():
        return None, None
    role_index = int(np.argmax(scores))
    if role_index >= len(ROLE_NAMES):
        return None, None
    shifted = scores - float(scores[role_index])
    weights = np.exp(shifted)
    denominator = float(weights.sum())
    if not np.isfinite(weights).all() or not np.isfinite(denominator) or denominator <= 0:
        return None, None
    confidence = float(weights[role_index] / denominator)
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        return None, None
    return ROLE_NAMES[role_index], confidence


def file_md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
