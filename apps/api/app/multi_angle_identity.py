"""Conservative identity evidence fusion across aligned replay angles.

Cross-view proximity is never identity evidence: two cameras can observe every
player at the same match time, and a weak clock alignment can put unrelated
events on top of each other.  This module therefore merges only explicit
roster identity or a reliable jersey number, after alignment and semantic
conflict checks.  Unmatched people remain review candidates and never create
duplicates in the reference canonical graph.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable


@dataclass(frozen=True)
class _Edge:
    source_id: str
    target_id: str
    score: float
    signals: tuple[str, ...]


def _person_id(person: dict[str, Any]) -> str:
    return str(person.get("canonicalPersonId") or person.get("id") or "").strip()


def _value(person: dict[str, Any], key: str) -> str | None:
    value = person.get(key)
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _namespace(source_scene_id: str, kind: str, identifier: str) -> str:
    digest = sha256(source_scene_id.encode("utf-8")).hexdigest()[:10]
    return f"angle-{digest}:{kind}:{identifier}"


def _has_identity_conflict(person: dict[str, Any]) -> bool:
    blocking_codes = {
        "jersey-ocr-conflict",
        "manual-roster-jersey-conflict",
        "manual-roster-player-missing",
    }
    return any(
        str(item.get("code") or "") in blocking_codes
        for item in person.get("conflicts") or []
        if isinstance(item, dict)
    )


def _edge(source: dict[str, Any], target: dict[str, Any]) -> _Edge | None:
    source_id, target_id = _person_id(source), _person_id(target)
    if not source_id or not target_id:
        return None

    source_team, target_team = _value(source, "teamId"), _value(target, "teamId")
    source_role, target_role = _value(source, "role"), _value(target, "role")
    source_external = _value(source, "externalPlayerId")
    target_external = _value(target, "externalPlayerId")
    source_jersey, target_jersey = _value(source, "jerseyNumber"), _value(target, "jerseyNumber")

    if _has_identity_conflict(source) or _has_identity_conflict(target):
        return None
    if source_team and target_team and source_team != target_team:
        return None
    if source_role and target_role and source_role != target_role:
        return None
    if source_external and target_external and source_external != target_external:
        return None
    if bool(source_external) != bool(target_external):
        # A bound identity cannot be transferred to an unbound person using a
        # shirt number alone.  Confirm the same external player on both angles
        # or leave the relation as review-only.
        return None
    if source_jersey and target_jersey and source_jersey != target_jersey:
        return None

    signals: list[str] = []
    score = 0.0
    if source_external and source_external == target_external:
        signals.append("external-player-match")
        score = 1.0
    if source_jersey and source_jersey == target_jersey:
        signals.append("reliable-jersey-match")
        score = max(score, 0.90)
    if not signals:
        return None
    return _Edge(source_id, target_id, score, tuple(signals))


def _alignment_usable(alignment: dict[str, Any]) -> bool:
    if not alignment.get("overlap"):
        return False
    try:
        confidence = float(alignment.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    method = str(alignment.get("method") or "")
    minimum = 0.50 if method.startswith("manual") else 0.65
    return confidence >= minimum and len(alignment.get("anchors") or []) >= 2


def _accepted_edges(edges: Iterable[_Edge], margin: float = 0.08) -> list[_Edge]:
    candidates = list(edges)
    by_source: dict[str, list[_Edge]] = {}
    by_target: dict[str, list[_Edge]] = {}
    for edge in candidates:
        by_source.setdefault(edge.source_id, []).append(edge)
        by_target.setdefault(edge.target_id, []).append(edge)

    result: list[_Edge] = []
    for edge in sorted(candidates, key=lambda item: (-item.score, item.source_id, item.target_id)):
        source_scores = sorted(
            (item.score for item in by_source[edge.source_id]), reverse=True
        )
        target_scores = sorted(
            (item.score for item in by_target[edge.target_id]), reverse=True
        )
        source_ambiguous = len(source_scores) > 1 and source_scores[0] - source_scores[1] < margin
        target_ambiguous = len(target_scores) > 1 and target_scores[0] - target_scores[1] < margin
        if source_ambiguous or target_ambiguous:
            continue
        if edge.score == source_scores[0] and edge.score == target_scores[0]:
            result.append(edge)
    return result


def fuse_aligned_identity_passes(
    reference_people: list[dict[str, Any]],
    aligned_passes: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach namespaced cross-angle evidence to the reference identity graph."""

    people = deepcopy(reference_people)
    target_by_id = {_person_id(person): person for person in people if _person_id(person)}
    diagnostics: dict[str, Any] = {
        "schemaVersion": 1,
        "sourcePassCount": 0,
        "usableAlignedPassCount": 0,
        "matchedIdentityCount": 0,
        "ambiguousOrUnmatchedCount": 0,
        "skippedPasses": [],
        "matches": [],
        "reviewCandidates": [],
    }

    for pass_input in aligned_passes:
        if not isinstance(pass_input, dict):
            continue
        diagnostics["sourcePassCount"] += 1
        source_scene_id = str(pass_input.get("sceneId") or "").strip()
        alignment = pass_input.get("alignment") or {}
        source_people = [
            person
            for person in (pass_input.get("canonicalPeople") or [])
            if isinstance(person, dict) and _person_id(person)
        ]
        if not source_scene_id or not _alignment_usable(alignment):
            diagnostics["skippedPasses"].append(
                {
                    "sceneId": source_scene_id or None,
                    "reason": "alignment-not-usable",
                    "confidence": alignment.get("confidence"),
                    "method": alignment.get("method"),
                }
            )
            diagnostics["ambiguousOrUnmatchedCount"] += len(source_people)
            continue
        diagnostics["usableAlignedPassCount"] += 1

        source_by_id = {_person_id(person): person for person in source_people}
        edges = [
            edge
            for source in source_people
            for target in people
            if (edge := _edge(source, target)) is not None
        ]
        accepted = _accepted_edges(edges)
        accepted_sources = {edge.source_id for edge in accepted}
        diagnostics["ambiguousOrUnmatchedCount"] += len(source_people) - len(accepted_sources)

        for edge in accepted:
            source = source_by_id[edge.source_id]
            target = target_by_id[edge.target_id]
            namespaced_observations = [
                {
                    **deepcopy(observation),
                    "id": _namespace(
                        source_scene_id,
                        "observation",
                        str(observation.get("observationId") or observation.get("id") or index),
                    ),
                    "observationId": _namespace(
                        source_scene_id,
                        "observation",
                        str(observation.get("observationId") or observation.get("id") or index),
                    ),
                    "sourceSceneId": source_scene_id,
                    "sourceCanonicalPersonId": edge.source_id,
                }
                for index, observation in enumerate(source.get("observations") or [])
                if isinstance(observation, dict)
            ]
            evidence = {
                "id": _namespace(source_scene_id, "identity-evidence", edge.source_id),
                "kind": "multi-angle-identity",
                "label": "Aligned replay identity evidence",
                "sourceSceneId": source_scene_id,
                "sourceCanonicalPersonId": edge.source_id,
                "signals": list(edge.signals),
                "confidence": edge.score,
                "alignmentConfidence": alignment.get("confidence"),
                "alignmentMethod": alignment.get("method"),
                "observationCount": len(namespaced_observations),
            }
            target.setdefault("evidence", []).append(evidence)
            target.setdefault("multiAngleEvidence", []).append(
                {
                    **evidence,
                    "observations": namespaced_observations,
                    "sourceTrackletIds": [
                        _namespace(source_scene_id, "tracklet", str(identifier))
                        for identifier in source.get("sourceTrackletIds")
                        or source.get("memberTrackletIds")
                        or []
                    ],
                }
            )
            target["sourcePassIds"] = sorted(
                {
                    *target.get("sourcePassIds", []),
                    source_scene_id,
                }
            )
            diagnostics["matchedIdentityCount"] += 1
            diagnostics["matches"].append(
                {
                    "sourceSceneId": source_scene_id,
                    "sourceCanonicalPersonId": edge.source_id,
                    "referenceCanonicalPersonId": edge.target_id,
                    "score": edge.score,
                    "signals": list(edge.signals),
                }
            )

        for source in source_people:
            source_id = _person_id(source)
            if source_id in accepted_sources:
                continue
            diagnostics["reviewCandidates"].append(
                {
                    "sourceSceneId": source_scene_id,
                    "sourceCanonicalPersonId": source_id,
                    "teamId": source.get("teamId"),
                    "role": source.get("role"),
                    "jerseyNumber": source.get("jerseyNumber"),
                    "externalPlayerId": source.get("externalPlayerId"),
                    "reason": "no-unique-independent-identity-match",
                }
            )

    return people, diagnostics


__all__ = ["fuse_aligned_identity_passes"]
