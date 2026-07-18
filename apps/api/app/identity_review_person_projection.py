"""Canonical-person, evidence and roster-hypothesis review projection."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .identity_decisions import rejected_roster_candidate_ids
from .identity_review_observation_projection import representative_observations


_EVIDENCE_FIELDS = {
    "id",
    "kind",
    "label",
    "value",
    "confidence",
    "supportCount",
    "sampleCount",
    "source",
    "model",
    "frameIndices",
    "manual",
    "status",
    "votes",
    "uniqueEvidenceFingerprintCount",
    "duplicateEvidenceFingerprintCount",
    "selectionPolicy",
    "selectedFrameIndices",
    "selectedQualities",
    "selectedEvidenceFingerprints",
    "partition",
    "sourceSceneId",
    "sourceCanonicalPersonId",
    "signals",
    "alignmentConfidence",
    "alignmentMethod",
    "observationCount",
}
_CANDIDATE_FIELDS = {
    "externalPlayerId",
    "rank",
    "score",
    "confidence",
    "identitySignalScore",
    "name",
    "number",
    "position",
    "teamId",
    "reasons",
    "conflicts",
    "eligible",
    "proposalStatus",
    "requiresManualConfirmation",
    "evidence",
}
_CONFLICT_FIELDS = {
    "id",
    "code",
    "message",
    "severity",
    "relatedCanonicalPersonIds",
    "relatedTrackletIds",
    "reasons",
    "externalPlayerId",
    "expectedNumber",
    "observedNumber",
    "bindingAnnotationIds",
    "rosterStatus",
}


def _review_evidence(values: object) -> list[dict]:
    result: list[dict] = []
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, Mapping):
            continue
        identifier = str(value.get("id") or "")
        kind = str(value.get("kind") or "")
        if not identifier or not kind:
            continue
        row = {
            key: deepcopy(item)
            for key, item in value.items()
            if key in _EVIDENCE_FIELDS
        }
        row["id"] = identifier
        row["kind"] = kind
        row["label"] = str(value.get("label") or kind.replace("-", " ").title())
        result.append(row)
    return result


def _candidate_evidence(values: object) -> list[dict]:
    result: list[dict] = []
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, Mapping) or not value.get("code"):
            continue
        result.append(
            {
                "code": str(value["code"]),
                "scoreDelta": float(value.get("scoreDelta") or 0.0),
                "confidence": float(value.get("confidence") or 0.0),
                "source": str(value.get("source") or "unknown"),
                "details": [str(item) for item in value.get("details") or []],
            }
        )
    return result


def _review_candidates(
    scene: Mapping[str, Any],
    canonical_id: str,
    values: object,
) -> list[dict]:
    rejected = rejected_roster_candidate_ids(scene, canonical_id)
    result: list[dict] = []
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, Mapping):
            continue
        external_id = str(value.get("externalPlayerId") or "")
        if not external_id or external_id in rejected:
            continue
        row = {
            key: deepcopy(item)
            for key, item in value.items()
            if key in _CANDIDATE_FIELDS and key != "evidence"
        }
        row["externalPlayerId"] = external_id
        row["evidence"] = _candidate_evidence(value.get("evidence"))
        result.append(row)
    return sorted(
        result,
        key=lambda item: (
            int(item.get("rank") or 10_000),
            -float(item.get("score") or item.get("confidence") or 0.0),
            item["externalPlayerId"],
        ),
    )


def _review_conflicts(canonical_id: str, values: object) -> list[dict]:
    result: list[dict] = []
    for index, value in enumerate(values if isinstance(values, list) else []):
        if not isinstance(value, Mapping):
            continue
        code = str(value.get("code") or "identity-conflict")
        row = {
            key: deepcopy(item)
            for key, item in value.items()
            if key in _CONFLICT_FIELDS
        }
        row.update(
            {
                "id": str(value.get("id") or f"{canonical_id}:review:{index}"),
                "code": code,
                "message": str(
                    value.get("message")
                    or f"Identity evidence requires review: {code.replace('-', ' ')}."
                ),
                "severity": (
                    "blocking" if value.get("severity") == "blocking" else "review"
                ),
            }
        )
        result.append(row)
    return result


def identity_review_items(
    scene: Mapping[str, Any],
    people: object,
    reid_by_observation: Mapping[str, dict[str, Any]],
    jersey_by_observation: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for person in people if isinstance(people, list) else []:
        canonical_id = str(person.get("canonicalPersonId") or person.get("id") or "")
        if not canonical_id:
            continue
        candidates = _review_candidates(
            scene,
            canonical_id,
            person.get("rosterCandidates"),
        )
        conflicts = _review_conflicts(canonical_id, person.get("conflicts"))
        external_id = person.get("externalPlayerId")
        identity_status = str(person.get("identityStatus") or "provisional")
        resolution_state = (
            "excluded"
            if identity_status == "excluded"
            else "conflict"
            if conflicts
            else "bound"
            if external_id
            else "suggested"
            if candidates
            else "anonymous"
        )
        priority = {
            "conflict": 0,
            "suggested": 1,
            "anonymous": 2,
            "bound": 3,
            "excluded": 4,
        }[resolution_state]
        items.append(
            {
                "canonicalPersonId": canonical_id,
                "displayName": person.get("displayName") or canonical_id,
                "identityStatus": identity_status,
                "identityConfidence": person.get("identityConfidence"),
                "identitySource": person.get("identitySource"),
                "teamId": person.get("teamId"),
                "role": person.get("role"),
                "jerseyNumber": person.get("jerseyNumber"),
                "candidateNumber": person.get("candidateNumber"),
                "externalPlayerId": external_id,
                "renderTrackId": person.get("renderTrackId"),
                "observationCount": int(person.get("observationCount") or 0),
                "resolutionState": resolution_state,
                "priority": priority,
                "representativeObservations": representative_observations(
                    person,
                    reid_by_observation,
                    jersey_by_observation,
                ),
                "evidence": _review_evidence(person.get("evidence")),
                "rosterCandidates": candidates,
                "conflicts": conflicts,
            }
        )
    return sorted(
        items,
        key=lambda item: (
            item["priority"],
            item.get("teamId") or "unknown",
            item.get("displayName") or item["canonicalPersonId"],
            item["canonicalPersonId"],
        ),
    )


__all__ = ("identity_review_items",)
