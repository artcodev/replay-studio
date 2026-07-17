"""Durable, compact manual action annotations for canonical people.

Dense pose/model output deliberately does not belong in the scene document.
This module owns only the reviewed semantic intervals which a later animation
layer may consume.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from math import isfinite
import re
from uuid import uuid4

from .store import scene_store


class PlayerActionError(ValueError):
    """A requested action edit is incompatible with the current scene."""


_ACTION_TYPES = frozenset(
    {
        "idle",
        "walk",
        "run",
        "sprint",
        "turn",
        "jump",
        "fall",
        "get-up",
        "first-touch",
        "drive",
        "pass",
        "cross",
        "shot",
        "header",
        "throw-in",
        "clearance",
        "tackle",
        "slide-tackle",
        "block",
        "interception",
        "feint",
    }
)
_KEYPOINT_KINDS = frozenset(
    {"wind-up", "contact", "release", "apex", "impact", "recovery"}
)
_ACTION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,119}$")


def _milliseconds(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PlayerActionError(f"{label} must be a finite number") from exc
    if not isfinite(number):
        raise PlayerActionError(f"{label} must be a finite number")
    return round(number, 3)


def _canonical_person_ids(scene: dict) -> set[str]:
    payload = scene.get("payload") or {}
    people = payload.get("canonicalPeople") or []
    # Canonical people are authoritative once the identity layer has run.
    # Legacy/demo scenes predate that layer, so their render-track identity is
    # still a valid canonical key (the reconstruction migration uses the same
    # fallback).
    subjects = people if people else payload.get("tracks") or []
    return {
        str(item.get("canonicalPersonId") or item.get("id") or "").strip()
        for item in subjects
        if str(item.get("canonicalPersonId") or item.get("id") or "").strip()
    }


def _saved_actions(scene: dict) -> tuple[dict, list[dict], dict[str, int]]:
    payload = scene.setdefault("payload", {})
    raw_actions = payload.get("playerActions")
    if raw_actions is None:
        raw_actions = []
    if not isinstance(raw_actions, list) or any(
        not isinstance(item, dict) for item in raw_actions
    ):
        raise PlayerActionError("The saved player-action collection is invalid")

    ids: dict[str, int] = {}
    duplicates: set[str] = set()
    for index, item in enumerate(raw_actions):
        identifier = str(item.get("id") or "").strip()
        if not identifier:
            raise PlayerActionError("A saved player action has no stable id")
        if identifier in ids:
            duplicates.add(identifier)
        else:
            ids[identifier] = index
    if duplicates:
        raise PlayerActionError(
            f"Saved player-action ids are not unique: {', '.join(sorted(duplicates))}"
        )
    return payload, deepcopy(raw_actions), ids


def _normalized_action(scene: dict, request: dict, existing: dict | None) -> dict:
    duration = _milliseconds(scene.get("duration"), "Scene duration")
    canonical_person_id = str(request.get("canonical_person_id") or "").strip()
    if canonical_person_id not in _canonical_person_ids(scene):
        raise PlayerActionError("The canonical person no longer exists")

    start_time = _milliseconds(request.get("start_time"), "Action start time")
    end_time = _milliseconds(request.get("end_time"), "Action end time")
    if start_time < 0 or end_time > duration:
        raise PlayerActionError(
            f"Action times must be between 0 and the scene duration ({duration:g}s)"
        )
    if start_time >= end_time:
        raise PlayerActionError("Action start time must be before end time")

    if existing is not None:
        if existing.get("source") != "manual":
            raise PlayerActionError(
                "Automatic action suggestions cannot be overwritten by a manual edit"
            )
        existing_owner = str(existing.get("canonicalPersonId") or "").strip()
        if existing_owner and existing_owner != canonical_person_id:
            raise PlayerActionError(
                "An existing action id cannot be reassigned to another canonical person"
            )

    action_type = str(request.get("type") or "").strip()
    if action_type not in _ACTION_TYPES:
        raise PlayerActionError("Unsupported player action type")

    # Millisecond normalization makes drag/save cycles deterministic. Exact
    # duplicate markers collapse, while repeated semantic events at different
    # times (for example contacts during a drive) remain representable.
    keypoints_by_key: dict[tuple[float, str], dict] = {}
    requested_keypoints = request.get("keypoints") or []
    if not isinstance(requested_keypoints, list) or len(requested_keypoints) > 24:
        raise PlayerActionError("An action may contain at most 24 compact keypoints")
    for item in requested_keypoints:
        if not isinstance(item, dict):
            raise PlayerActionError("Every action keypoint must be an object")
        time = _milliseconds(item.get("time"), "Action keypoint time")
        kind = str(item.get("kind") or "").strip()
        if kind not in _KEYPOINT_KINDS:
            raise PlayerActionError("Unsupported player action keypoint kind")
        if time < start_time or time > end_time:
            raise PlayerActionError(
                f"Action keypoint {kind!r} at {time:g}s is outside its action interval"
            )
        keypoints_by_key[(time, kind)] = {"kind": kind, "time": time}
    keypoints = [
        keypoints_by_key[key]
        for key in sorted(keypoints_by_key, key=lambda value: (value[0], value[1]))
    ]

    now = datetime.now(UTC).isoformat()
    action_id = str(request.get("id") or "").strip()
    if not action_id:
        action_id = f"action-{uuid4().hex}"
    elif _ACTION_ID_PATTERN.fullmatch(action_id) is None:
        raise PlayerActionError("Player action id has an invalid format")
    result = {
        "id": action_id,
        "canonicalPersonId": canonical_person_id,
        "type": action_type,
        "startTime": start_time,
        "endTime": end_time,
        "keypoints": keypoints,
        "confidence": 1.0,
        "status": "confirmed",
        "source": "manual",
        "createdAt": (
            str(existing.get("createdAt"))
            if existing is not None and existing.get("createdAt")
            else now
        ),
        "updatedAt": now,
    }
    # Imported/manual evidence remains useful after an interval adjustment.
    # Automatic suggestions are rejected above and need a dedicated review
    # operation rather than being silently converted by this manual endpoint.
    if existing is not None and isinstance(existing.get("evidence"), dict):
        result["evidence"] = deepcopy(existing["evidence"])
    return result


def upsert_player_action(
    scene: dict,
    request: dict,
    *,
    persist: bool = True,
) -> dict:
    """Create/update one manual action without touching unrelated hypotheses."""

    payload, actions, ids = _saved_actions(scene)

    requested_id = str(request.get("id") or "").strip()
    existing_index = ids.get(requested_id) if requested_id else None
    existing = actions[existing_index] if existing_index is not None else None
    normalized = _normalized_action(scene, request, existing)
    if existing_index is None:
        # Generated UUIDs are collision-resistant; still fail closed if a
        # future id generator or imported document violates that invariant.
        if normalized["id"] in ids:
            raise PlayerActionError("The player-action id already exists")
        actions.append(normalized)
    else:
        actions[existing_index] = normalized

    payload["playerActions"] = actions
    if persist:
        scene_store.put(scene)
    return normalized


def delete_player_action(
    scene: dict,
    action_id: str,
    *,
    persist: bool = True,
) -> dict:
    """Delete exactly one action, preserving all unrelated suggestions."""

    identifier = str(action_id or "").strip()
    if not identifier:
        raise PlayerActionError("Player action id is required")
    payload, actions, ids = _saved_actions(scene)
    existing_index = ids.get(identifier)
    if existing_index is None:
        raise PlayerActionError("Player action not found")
    existing = actions[existing_index]
    if existing.get("source") != "manual":
        raise PlayerActionError("Automatic action suggestions cannot be deleted here")
    payload["playerActions"] = [
        item for index, item in enumerate(actions) if index != existing_index
    ]
    if persist:
        scene_store.put(scene)
    return existing
