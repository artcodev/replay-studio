"""Durable human decisions over review-only identity hypotheses."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any


ROSTER_REJECTION_SCHEMA = "roster-candidate-rejection-v1"


class IdentityDecisionError(RuntimeError):
    pass


def _canonical_person(scene: dict, canonical_person_id: str) -> dict:
    matches = [
        person
        for person in scene.get("payload", {}).get("canonicalPeople") or []
        if str(person.get("canonicalPersonId") or person.get("id") or "")
        == canonical_person_id
    ]
    if len(matches) != 1:
        raise IdentityDecisionError("The canonical person no longer exists")
    return matches[0]


def rejected_roster_candidate_ids(
    scene: dict,
    canonical_person_id: str,
) -> set[str]:
    decisions = scene.get("payload", {}).get("identityReviewDecisions") or {}
    return {
        str(item.get("externalPlayerId"))
        for item in decisions.get("rosterRejections") or []
        if item.get("schema") == ROSTER_REJECTION_SCHEMA
        and str(item.get("canonicalPersonId") or "") == canonical_person_id
        and str(item.get("externalPlayerId") or "")
    }


def reject_roster_candidate(
    scene: dict,
    canonical_person_id: str,
    external_player_id: str,
) -> dict[str, Any]:
    person = _canonical_person(scene, canonical_person_id)
    external_player_id = str(external_player_id or "").strip()
    if not external_player_id:
        raise IdentityDecisionError("Roster candidate ID is required")
    if str(person.get("externalPlayerId") or "") == external_player_id:
        raise IdentityDecisionError(
            "Unbind the confirmed roster player before rejecting that candidate"
        )
    candidate_ids = {
        str(item.get("externalPlayerId") or "")
        for item in person.get("rosterCandidates") or []
    }
    if external_player_id not in candidate_ids:
        raise IdentityDecisionError(
            "Only a published roster suggestion can be rejected"
        )
    roster_ids = {
        str(item.get("id") or "")
        for item in (
            scene.get("payload", {}).get("matchBinding", {}).get("players") or []
        )
        if str(item.get("id") or "")
    }
    if external_player_id not in roster_ids:
        raise IdentityDecisionError("Roster candidate is absent from the saved match")
    payload = scene.setdefault("payload", {})
    review = payload.setdefault("identityReviewDecisions", {})
    records = list(review.get("rosterRejections") or [])
    existing = next(
        (
            item
            for item in records
            if item.get("schema") == ROSTER_REJECTION_SCHEMA
            and str(item.get("canonicalPersonId") or "") == canonical_person_id
            and str(item.get("externalPlayerId") or "") == external_player_id
        ),
        None,
    )
    if existing is not None:
        return existing
    observation_ids = sorted(
        str(item.get("observationId") or item.get("id") or "")
        for item in person.get("observations") or []
        if str(item.get("observationId") or item.get("id") or "")
    )
    digest = sha256(
        f"{ROSTER_REJECTION_SCHEMA}:{canonical_person_id}:{external_player_id}".encode(
            "utf-8"
        )
    ).hexdigest()[:20]
    record = {
        "id": f"roster-rejection-{digest}",
        "schema": ROSTER_REJECTION_SCHEMA,
        "canonicalPersonId": canonical_person_id,
        "externalPlayerId": external_player_id,
        "anchorObservationId": observation_ids[0] if observation_ids else None,
        "createdAt": datetime.now(UTC).isoformat(),
    }
    records.append(record)
    review["rosterRejections"] = sorted(
        records,
        key=lambda item: (
            str(item.get("canonicalPersonId") or ""),
            str(item.get("externalPlayerId") or ""),
            str(item.get("id") or ""),
        ),
    )
    return record


def clear_roster_candidate_rejection(
    scene: dict,
    canonical_person_id: str,
    external_player_id: str,
) -> dict[str, Any]:
    _canonical_person(scene, canonical_person_id)
    review = scene.setdefault("payload", {}).setdefault(
        "identityReviewDecisions", {}
    )
    records = list(review.get("rosterRejections") or [])
    removed = [
        item
        for item in records
        if item.get("schema") == ROSTER_REJECTION_SCHEMA
        and str(item.get("canonicalPersonId") or "") == canonical_person_id
        and str(item.get("externalPlayerId") or "") == external_player_id
    ]
    if len(removed) != 1:
        raise IdentityDecisionError("Roster candidate rejection was not found")
    removed_id = str(removed[0].get("id") or "")
    review["rosterRejections"] = [
        item for item in records if str(item.get("id") or "") != removed_id
    ]
    return removed[0]


__all__ = [
    "IdentityDecisionError",
    "ROSTER_REJECTION_SCHEMA",
    "clear_roster_candidate_rejection",
    "reject_roster_candidate",
    "rejected_roster_candidate_ids",
]
