from __future__ import annotations

"""Typed application contract for the optional identity worker."""

from dataclasses import dataclass, field


IDENTITY_BACKEND = "prtreid-bpbreid-soccernet"
IDENTITY_EMBEDDING_DIMENSION = 256
# v2: fingerprints are computed over decoded store-crop pixels (the worker
# no longer sees frames or bboxes).
EVIDENCE_FINGERPRINT_VERSION = "pixel-evidence-v2"
# v2: batches upload person-crop-store bytes with a flat crop manifest.
IDENTITY_REQUEST_CONTRACT_VERSION = 2
KNOWN_IDENTITY_ROLES = frozenset(
    {"ball", "goalkeeper", "other", "player", "referee"}
)


class IdentityWorkerError(RuntimeError):
    pass


@dataclass(slots=True)
class IdentityWorkerBatchResult:
    items_by_observation_id: dict[str, dict] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
