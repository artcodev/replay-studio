from __future__ import annotations

"""Typed application contract for the optional identity worker."""

from dataclasses import dataclass, field


IDENTITY_BACKEND = "prtreid-bpbreid-soccernet"
IDENTITY_EMBEDDING_DIMENSION = 256
EVIDENCE_FINGERPRINT_VERSION = "pixel-evidence-v1"
KNOWN_IDENTITY_ROLES = frozenset(
    {"ball", "goalkeeper", "other", "player", "referee"}
)


class IdentityWorkerError(RuntimeError):
    pass


@dataclass(slots=True)
class IdentityWorkerBatchResult:
    items_by_observation_id: dict[str, dict] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
