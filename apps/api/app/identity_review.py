"""Compact, auditable identity-review projections for the editor.

The scene document intentionally retains every immutable observation.  The
review UI should not need to download that complete graph merely to show the
few crops and hypotheses that need a decision, so this module derives a small
read-only queue without creating new identity evidence.
"""

from __future__ import annotations

from copy import deepcopy
from math import sqrt
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import cv2

from .config import get_settings
from .identity_decisions import rejected_roster_candidate_ids


class IdentityReviewError(RuntimeError):
    """Raised when persisted review evidence cannot be resolved safely."""


def _identity_diagnostics(scene: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(
        (
            scene.get("payload", {})
            .get("videoAsset", {})
            .get("reconstruction", {})
            .get("diagnostics", {})
            .get("identity", {})
        )
        or {}
    )


def _crop_evidence_by_observation(
    diagnostics: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    reid: dict[str, dict[str, Any]] = {}
    jersey: dict[str, dict[str, Any]] = {}
    for item in (diagnostics.get("reid") or {}).get("crops") or []:
        observation_id = str(item.get("observationId") or "")
        if observation_id:
            reid[observation_id] = deepcopy(item)
    for item in (diagnostics.get("jerseyOcr") or {}).get("crops") or []:
        observation_id = str(item.get("observationId") or "")
        if observation_id:
            jersey[observation_id] = deepcopy(item)
    return reid, jersey


def _observation_score(
    observation: Mapping[str, Any],
    reid: Mapping[str, Any] | None,
    jersey: Mapping[str, Any] | None,
) -> float:
    bbox = observation.get("bbox") or {}
    area = max(0.0, float(bbox.get("width") or 0.0)) * max(
        0.0, float(bbox.get("height") or 0.0)
    )
    confidence = max(0.0, min(1.0, float(observation.get("confidence") or 0.0)))
    score = confidence * min(1.0, sqrt(area) / 180.0)
    if reid and reid.get("status") in {"usable", "ready"}:
        score += 0.25
    if reid and reid.get("usable") is True:
        score += 0.25
    if jersey and jersey.get("status") == "recognized":
        score += 0.45
    elif jersey and jersey.get("status") in {"low-confidence", "ambiguous"}:
        score += 0.18
    return score


def _representative_observations(
    person: Mapping[str, Any],
    reid_by_observation: Mapping[str, dict[str, Any]],
    jersey_by_observation: Mapping[str, dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, float, str, Mapping[str, Any]]] = []
    for observation in person.get("observations") or []:
        observation_id = str(
            observation.get("observationId") or observation.get("id") or ""
        )
        if not observation_id or not observation.get("bbox"):
            continue
        scene_time = float(observation.get("sceneTime") or 0.0)
        reid = reid_by_observation.get(observation_id)
        jersey = jersey_by_observation.get(observation_id)
        ranked.append(
            (
                _observation_score(observation, reid, jersey),
                scene_time,
                observation_id,
                observation,
            )
        )

    selected: list[tuple[float, float, str, Mapping[str, Any]]] = []
    for candidate in sorted(ranked, key=lambda item: (-item[0], item[1], item[2])):
        if any(abs(candidate[1] - previous[1]) < 0.35 for previous in selected):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: (item[1], item[2]))

    result: list[dict[str, Any]] = []
    for quality, scene_time, observation_id, observation in selected:
        result.append(
            {
                "id": observation_id,
                "observationId": observation_id,
                "frameIndex": observation.get("frameIndex"),
                "sourceFrameIndex": observation.get("sourceFrameIndex")
                or observation.get("frameIndex"),
                "sceneTime": scene_time,
                "sourceTime": observation.get("sourceTime"),
                "bbox": deepcopy(observation.get("bbox")),
                "confidence": observation.get("confidence"),
                "reviewQuality": round(quality, 6),
                "quality": round(quality, 6),
                "cropUrl": None,
                "rejectionReasons": list(
                    dict.fromkeys(
                        [
                            *(
                                (reid_by_observation.get(observation_id) or {}).get(
                                    "rejectionReasons"
                                )
                                or []
                            ),
                            *(
                                (jersey_by_observation.get(observation_id) or {}).get(
                                    "rejectionReasons"
                                )
                                or []
                            ),
                        ]
                    )
                ),
                "reid": deepcopy(reid_by_observation.get(observation_id)),
                "jerseyOcr": deepcopy(jersey_by_observation.get(observation_id)),
            }
        )
    return result


def _roster_status(binding: Mapping[str, Any]) -> dict[str, Any]:
    players = binding.get("players") or []
    warnings = [str(value) for value in binding.get("warnings") or []]
    quality = (
        binding.get("rosterQuality")
        if isinstance(binding.get("rosterQuality"), dict)
        else {}
    )
    explicit = (
        binding.get("coverage")
        if isinstance(binding.get("coverage"), dict)
        else {}
    )
    complete = (
        quality.get("automaticIdentityEligible")
        if "automaticIdentityEligible" in quality
        else explicit.get("rosterComplete")
    )
    truncated = any("first five" in warning.lower() for warning in warnings)
    if complete is True:
        status = "ready"
    elif not players:
        status = "unavailable"
    elif complete is False or truncated or len(players) < 11:
        status = "incomplete"
    else:
        status = "review"
    return {
        "status": status,
        "playerCount": len(players),
        "complete": complete is True,
        "automaticIdentityEligible": bool(
            quality.get("automaticIdentityEligible", complete is True)
        ),
        "manualIdentityEligible": bool(
            quality.get("manualIdentityEligible", bool(players))
        ),
        "reasons": list(quality.get("reasons") or []),
        "warnings": warnings,
    }


def build_identity_review(
    scene: Mapping[str, Any],
    *,
    worker_health: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic review queue without mutating the scene."""

    payload = scene.get("payload", {})
    people = payload.get("canonicalPeople") or []
    diagnostics = _identity_diagnostics(scene)
    reid_by_observation, jersey_by_observation = _crop_evidence_by_observation(
        diagnostics
    )
    items: list[dict[str, Any]] = []
    for person in people:
        canonical_id = str(person.get("canonicalPersonId") or person.get("id") or "")
        if not canonical_id:
            continue
        candidates = sorted(
            [
                deepcopy(candidate)
                for candidate in person.get("rosterCandidates") or []
                if str(candidate.get("externalPlayerId") or "")
                not in rejected_roster_candidate_ids(scene, canonical_id)
            ],
            key=lambda item: (
                int(item.get("rank") or 10_000),
                -float(item.get("score") or item.get("confidence") or 0.0),
                str(item.get("externalPlayerId") or ""),
            ),
        )
        conflicts = deepcopy(person.get("conflicts") or [])
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
        observations = _representative_observations(
            person,
            reid_by_observation,
            jersey_by_observation,
        )
        for observation in observations:
            observation["cropUrl"] = (
                f"/api/scenes/{quote(str(scene.get('id') or ''), safe='')}/"
                "identity-observations/"
                f"{quote(str(observation['observationId']), safe='')}/crop"
            )
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
                "representativeObservations": observations,
                "evidence": deepcopy(person.get("evidence") or []),
                "rosterCandidates": candidates,
                "conflicts": conflicts,
            }
        )
    items.sort(
        key=lambda item: (
            item["priority"],
            item.get("teamId") or "unknown",
            item.get("displayName") or item["canonicalPersonId"],
            item["canonicalPersonId"],
        )
    )
    binding = payload.get("matchBinding") or {}
    return {
        "sceneId": scene.get("id"),
        "revision": scene.get("revision", 0),
        "matchBinding": {
            "source": binding.get("source"),
            "eventId": binding.get("eventId"),
            "fetchedAt": binding.get("fetchedAt"),
            "roster": _roster_status(binding),
        },
        "workers": deepcopy(worker_health or {}),
        "summary": {
            "canonicalPersonCount": len(items),
            "boundCount": sum(item["resolutionState"] == "bound" for item in items),
            "suggestedCount": sum(
                item["resolutionState"] == "suggested" for item in items
            ),
            "conflictCount": sum(
                item["resolutionState"] == "conflict" for item in items
            ),
            "anonymousCount": sum(
                item["resolutionState"] == "anonymous" for item in items
            ),
            "excludedCount": sum(
                item["resolutionState"] == "excluded" for item in items
            ),
        },
        "items": items,
    }


def identity_observation_crop(
    scene: Mapping[str, Any],
    observation_id: str,
    *,
    media_root: Path | None = None,
    padding_ratio: float = 0.12,
) -> bytes:
    """Return the exact sampled-frame crop used by identity review.

    The observation is looked up from the persisted canonical graph; callers
    cannot choose an arbitrary frame path or bbox.
    """

    match: Mapping[str, Any] | None = None
    for person in scene.get("payload", {}).get("canonicalPeople") or []:
        for observation in person.get("observations") or []:
            identifier = str(
                observation.get("observationId") or observation.get("id") or ""
            )
            if identifier == observation_id:
                if match is not None:
                    raise IdentityReviewError(
                        "Observation ID is ambiguous across canonical people"
                    )
                match = observation
    if match is None:
        raise IdentityReviewError("Identity observation was not found")
    video = scene.get("payload", {}).get("videoAsset") or {}
    asset_id = str(video.get("id") or "")
    frame_index = match.get("sourceFrameIndex") or match.get("frameIndex")
    if not asset_id or frame_index is None:
        raise IdentityReviewError("Identity observation has no source frame")
    root = Path(media_root or get_settings().media_root).resolve()
    frame_path = (
        root / asset_id / "frames" / f"frame_{int(frame_index):05d}.jpg"
    ).resolve()
    if not frame_path.is_relative_to(root):
        raise IdentityReviewError("Identity source frame path is invalid")
    image = cv2.imread(str(frame_path))
    if image is None:
        raise IdentityReviewError("Identity source frame is unavailable")
    bbox = match.get("bbox") or {}
    x = float(bbox.get("x") or 0.0)
    y = float(bbox.get("y") or 0.0)
    width = float(bbox.get("width") or 0.0)
    height = float(bbox.get("height") or 0.0)
    if width <= 0.0 or height <= 0.0:
        raise IdentityReviewError("Identity observation has an invalid bbox")
    pad_x = width * max(0.0, min(0.5, padding_ratio))
    pad_y = height * max(0.0, min(0.5, padding_ratio))
    image_height, image_width = image.shape[:2]
    x1 = max(0, min(image_width, int(x - pad_x)))
    y1 = max(0, min(image_height, int(y - pad_y)))
    x2 = max(0, min(image_width, int(x + width + pad_x + 0.999)))
    y2 = max(0, min(image_height, int(y + height + pad_y + 0.999)))
    if x2 <= x1 or y2 <= y1:
        raise IdentityReviewError("Identity observation crop is empty")
    crop = image[y1:y2, x1:x2]
    encoded, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not encoded:
        raise IdentityReviewError("Identity observation crop could not be encoded")
    return bytes(buffer)


__all__ = [
    "IdentityReviewError",
    "build_identity_review",
    "identity_observation_crop",
]
