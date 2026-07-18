"""Audit diagnostics for canonical identity resolution results."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Sequence

import numpy as np

from .identity_resolution_components import IdentityNode
from .identity_resolution_contract import (
    IdentityEdge,
    IdentityGroup,
    IdentityResolverConfig,
    IdentityTracklet,
)


def empty_identity_diagnostics() -> dict:
    return {
        "schemaVersion": 1,
        "trackletCount": 0,
        "canonicalIdentityCount": 0,
        "allTrackletsPreserved": True,
        "identityObservationCoverage": 1.0,
        "associationConfidenceP10": None,
        "associationConfidenceP50": None,
        "acceptedAssociationConfidenceP10": None,
        "reviewAssociationConfidenceP50": None,
        "strongReidBidirectionalEdgeCount": 0,
        "groundTruthAvailable": False,
        "estimatedIdSwitchCount": None,
        "duplicateOverlapSeconds": None,
    }


def build_identity_diagnostics(
    *,
    tracklets: Sequence[IdentityTracklet],
    nodes: Sequence[IdentityNode],
    groups: Sequence[IdentityGroup],
    accepted: Sequence[IdentityEdge],
    review: Sequence[IdentityEdge],
    rejected: Sequence[IdentityEdge],
    tracklet_to_identity: dict[str, str],
    config: IdentityResolverConfig,
) -> dict:
    """Build deterministic quality and preservation metrics for one solve."""

    rejection_counts = Counter(reason for edge in rejected for reason in edge.reasons)
    observation_count = sum(item.observation_count for item in tracklets)
    accepted_scores = [float(edge.score) for edge in accepted if edge.score is not None]
    review_scores = [float(edge.score) for edge in review if edge.score is not None]
    evidence_scores = [*accepted_scores, *review_scores]
    return {
        "schemaVersion": 1,
        "trackletCount": len(tracklets),
        "eligibleTrackletCount": sum(not item.manual_excluded for item in tracklets),
        "excludedTrackletCount": sum(item.manual_excluded for item in tracklets),
        "canonicalIdentityCount": len(groups),
        "resolvedIdentityCount": sum(group.status == "resolved" for group in groups),
        "provisionalIdentityCount": sum(group.status == "provisional" for group in groups),
        "excludedIdentityCount": sum(group.status == "excluded" for group in groups),
        "acceptedEdgeCount": len(accepted),
        "autoStitchCount": sum(edge.source == "automatic" for edge in accepted),
        "manualStitchCount": sum(edge.source == "manual" for edge in accepted),
        "reviewEdgeCount": len(review),
        "rejectedEdgeCount": len(rejected),
        "ambiguousEdgeCount": sum(
            any(reason.startswith("ambiguous-") for reason in edge.reasons)
            for edge in review
        ),
        "manualConflictCount": sum(bool(node.conflict_reasons) for node in nodes),
        "rejectionReasonCounts": dict(sorted(rejection_counts.items())),
        "observationCount": observation_count,
        "preservedObservationCount": observation_count,
        "identityObservationCoverage": 1.0,
        "associationConfidenceP10": (
            round(float(np.percentile(evidence_scores, 10)), 6)
            if evidence_scores
            else None
        ),
        "associationConfidenceP50": (
            round(float(np.percentile(evidence_scores, 50)), 6)
            if evidence_scores
            else None
        ),
        "acceptedAssociationConfidenceP10": (
            round(float(np.percentile(accepted_scores, 10)), 6)
            if accepted_scores
            else None
        ),
        "reviewAssociationConfidenceP50": (
            round(float(np.percentile(review_scores, 50)), 6)
            if review_scores
            else None
        ),
        "strongReidBidirectionalEdgeCount": sum(
            "strong-reid" in edge.reasons
            and edge.reid_strong_support_left >= config.min_strong_reid_samples
            and edge.reid_strong_support_right >= config.min_strong_reid_samples
            for edge in accepted
        ),
        "groundTruthAvailable": False,
        "estimatedIdSwitchCount": None,
        "duplicateOverlapSeconds": None,
        "allTrackletsPreserved": len(tracklet_to_identity) == len(tracklets),
        "inputTrackletDigest": sha256(
            "\n".join(sorted(item.id for item in tracklets)).encode("utf-8")
        ).hexdigest(),
    }


__all__ = ["build_identity_diagnostics", "empty_identity_diagnostics"]
