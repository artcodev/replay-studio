"""Pure compaction and hydration codecs for reconstruction artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .artifact_store import ReconstructionArtifactError


MATERIALIZED_ARTIFACTS_KEY = "_materializedArtifactNames"


@dataclass(frozen=True)
class IdentityTimelineEncoding:
    compact_tracks: list[dict[str, Any]]
    compact_people: list[dict[str, Any]]
    payload: dict[str, Any]


@dataclass(frozen=True)
class CalibrationFramesEncoding:
    compact_calibration: dict[str, Any]
    compact_ball_detection: dict[str, Any]
    payload: dict[str, Any]


def compact_identity_diagnostics(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in diagnostics.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def materialized_artifacts(reconstruction: Mapping[str, Any]) -> set[str]:
    raw = reconstruction.get(MATERIALIZED_ARTIFACTS_KEY, ())
    if not isinstance(raw, (list, tuple, set)):
        raise ReconstructionArtifactError(
            "Reconstruction materialization marker is malformed"
        )
    return {str(name) for name in raw}


def set_materialized_artifacts(
    reconstruction: dict[str, Any],
    names: Iterable[str],
) -> None:
    remaining = {str(name) for name in names}
    if remaining:
        reconstruction[MATERIALIZED_ARTIFACTS_KEY] = sorted(remaining)
    else:
        reconstruction.pop(MATERIALIZED_ARTIFACTS_KEY, None)


def mark_materialized_artifacts(
    reconstruction: dict[str, Any],
    names: Iterable[str],
) -> None:
    merged = materialized_artifacts(reconstruction)
    merged.update(str(name) for name in names)
    set_materialized_artifacts(reconstruction, merged)


def compact_mapping(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return deepcopy(value)
    if isinstance(value, Mapping):
        return {
            str(key): compact
            for key, item in value.items()
            if (compact := compact_mapping(item)) is not None
        }
    return None


def encode_identity_timeline(
    scene_id: str,
    tracks: Iterable[object],
    people: Iterable[object],
) -> IdentityTimelineEncoding:
    compact_tracks: list[dict[str, Any]] = []
    dense_tracks: list[dict[str, Any]] = []
    for track in tracks:
        if not isinstance(track, Mapping):
            continue
        keyframes = deepcopy(track.get("keyframes") or [])
        observations = deepcopy(track.get("observations") or [])
        compact = {
            key: deepcopy(value)
            for key, value in track.items()
            if key not in {"keyframes", "observations"}
        }
        compact["keyframeCount"] = len(keyframes)
        compact["observationCount"] = len(observations)
        compact_tracks.append(compact)
        dense_tracks.append(
            {
                "id": str(track.get("id") or ""),
                "keyframes": keyframes,
                "observations": observations,
            }
        )

    compact_people: list[dict[str, Any]] = []
    dense_people: list[dict[str, Any]] = []
    for person in people:
        if not isinstance(person, Mapping):
            continue
        person_id = str(
            person.get("canonicalPersonId") or person.get("id") or ""
        )
        observations = deepcopy(person.get("observations") or [])
        compact = {
            key: deepcopy(value)
            for key, value in person.items()
            if key != "observations"
        }
        compact["observationCount"] = len(observations)
        compact_people.append(compact)
        dense_people.append(
            {
                "canonicalPersonId": person_id,
                "observations": observations,
            }
        )

    return IdentityTimelineEncoding(
        compact_tracks=compact_tracks,
        compact_people=compact_people,
        payload={
            "sceneId": scene_id,
            "tracks": dense_tracks,
            "canonicalPeople": dense_people,
        },
    )


def ball_artifact_payload(
    scene_id: str,
    ball: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sceneId": scene_id,
        "mode": ball.get("mode") or "automatic",
        "keyframes": deepcopy(ball.get("keyframes") or []),
        "automaticKeyframes": deepcopy(ball.get("automaticKeyframes") or []),
        "manualKeyframes": deepcopy(ball.get("manualKeyframes") or []),
        "diagnostics": deepcopy(ball.get("diagnostics") or {}),
        "automaticDiagnostics": deepcopy(
            ball.get("automaticDiagnostics") or {}
        ),
        "manualDiagnostics": deepcopy(ball.get("manualDiagnostics") or {}),
    }


def compact_ball(ball: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        key: deepcopy(value)
        for key, value in ball.items()
        if key
        not in {
            "keyframes",
            "automaticKeyframes",
            "manualKeyframes",
            "diagnostics",
            "automaticDiagnostics",
            "manualDiagnostics",
        }
    }
    for key in ("diagnostics", "automaticDiagnostics", "manualDiagnostics"):
        compact[key] = compact_mapping(ball.get(key) or {})
    compact.update(
        {
            "keyframeCount": len(ball.get("keyframes") or []),
            "automaticKeyframeCount": len(
                ball.get("automaticKeyframes") or []
            ),
            "manualKeyframeCount": len(ball.get("manualKeyframes") or []),
        }
    )
    return compact


def encode_calibration_frames(
    scene_id: str,
    calibration: Mapping[str, Any],
    ball_detection: Mapping[str, Any],
) -> CalibrationFramesEncoding:
    frame_evidence = deepcopy(calibration.get("frameEvidence") or [])
    compact_calibration = {
        key: deepcopy(value)
        for key, value in calibration.items()
        if key != "frameEvidence"
    }
    compact_calibration["frameEvidenceCount"] = len(frame_evidence)
    compact_detection = compact_mapping(ball_detection)
    return CalibrationFramesEncoding(
        compact_calibration=compact_calibration,
        compact_ball_detection=(
            compact_detection if isinstance(compact_detection, dict) else {}
        ),
        payload={
            "sceneId": scene_id,
            "frameEvidence": frame_evidence,
            "ballDetection": deepcopy(ball_detection),
        },
    )


def _consume_count(
    value: dict[str, Any],
    key: str,
    actual: int,
    *,
    owner: str,
) -> None:
    if key in value and value[key] != actual:
        raise ReconstructionArtifactError(
            f"{owner} {key} does not match its dense artifact"
        )
    value.pop(key, None)


def hydrate_identity_timeline(payload: dict[str, Any], dense: Mapping[str, Any]) -> None:
    tracks_by_id = {
        str(item.get("id") or ""): item
        for item in dense.get("tracks") or []
        if isinstance(item, Mapping)
    }
    for track in payload.get("tracks") or []:
        if not isinstance(track, dict):
            continue
        values = tracks_by_id.get(str(track.get("id") or ""), {})
        keyframes = deepcopy(values.get("keyframes") or [])
        observations = deepcopy(values.get("observations") or [])
        owner = f"Track {str(track.get('id') or '')!r}"
        _consume_count(track, "keyframeCount", len(keyframes), owner=owner)
        _consume_count(track, "observationCount", len(observations), owner=owner)
        track["keyframes"] = keyframes
        track["observations"] = observations

    people_by_id = {
        str(item.get("canonicalPersonId") or ""): item
        for item in dense.get("canonicalPeople") or []
        if isinstance(item, Mapping)
    }
    for person in payload.get("canonicalPeople") or []:
        if not isinstance(person, dict):
            continue
        person_id = str(
            person.get("canonicalPersonId") or person.get("id") or ""
        )
        observations = deepcopy(
            people_by_id.get(person_id, {}).get("observations") or []
        )
        _consume_count(
            person,
            "observationCount",
            len(observations),
            owner=f"Canonical person {person_id!r}",
        )
        person["observations"] = observations


def hydrate_ball(payload: dict[str, Any], dense: Mapping[str, Any]) -> None:
    ball = payload.setdefault("ball", {})
    for key, count_key in {
        "keyframes": "keyframeCount",
        "automaticKeyframes": "automaticKeyframeCount",
        "manualKeyframes": "manualKeyframeCount",
    }.items():
        values = deepcopy(dense.get(key) or [])
        _consume_count(ball, count_key, len(values), owner="Ball trajectory")
        ball[key] = values
    for key in ("diagnostics", "automaticDiagnostics", "manualDiagnostics"):
        ball[key] = deepcopy(dense.get(key) or {})


def hydrate_calibration(
    reconstruction: dict[str, Any],
    dense: Mapping[str, Any],
) -> None:
    calibration = reconstruction.setdefault("calibration", {})
    frame_evidence = deepcopy(dense.get("frameEvidence") or [])
    _consume_count(
        calibration,
        "frameEvidenceCount",
        len(frame_evidence),
        owner="Calibration",
    )
    calibration["frameEvidence"] = frame_evidence
    reconstruction["ballDetection"] = deepcopy(dense.get("ballDetection") or {})
