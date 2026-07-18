from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from .jersey_ocr_contract import JerseyEvidenceSummary, normalize_jersey_number
from .reconstruction_track_state import TrackState
from .reconstruction_identity_persistence import track_state_observations


def published_identity_observations(
    track: TrackState,
    rendered: dict | None,
    canonical_id: str,
    source_start: float,
) -> list[dict]:
    raw = track_state_observations(
        track, canonical_person_id=canonical_id, source_start=source_start
    )
    source = (rendered.get("observations") if rendered is not None else raw) or []
    result: list[dict] = []
    for observation in source:
        enriched = {
            **deepcopy(observation),
            "id": observation.get("id") or observation.get("observationId"),
            "observationId": observation.get("observationId")
            or observation.get("id"),
            "canonicalPersonId": canonical_id,
        }
        if not enriched.get("sourceTrackletId"):
            matching = next(
                (
                    item
                    for item in raw
                    if int(item["frameIndex"]) == int(enriched["frameIndex"])
                ),
                None,
            )
            enriched["sourceTrackletId"] = (
                matching.get("sourceTrackletId") if matching else track.local_tracklet_id
            )
        result.append(enriched)
    return result


def _jersey_evidence_payload(
    canonical_id: str,
    summary: JerseyEvidenceSummary,
    resolver_diagnostics: Mapping | None,
) -> dict:
    model_version = (resolver_diagnostics or {}).get("jerseyOcr", {}).get(
        "modelVersion"
    )
    if isinstance(model_version, Mapping):
        model_version = model_version.get("modelVersion")
    return {
        "id": f"{canonical_id}:jersey-ocr",
        "kind": "jersey-ocr",
        "label": "Jersey number OCR",
        "value": summary.jersey_number or summary.candidate_number,
        "confidence": round(float(summary.confidence), 6),
        "supportCount": summary.support_count,
        "sampleCount": summary.selected_sample_count,
        "source": "jersey-ocr-worker",
        "model": model_version,
        "frameIndices": [
            int(item.frame_index)
            for item in summary.selected_observations
            if item.frame_index is not None
        ],
        "status": summary.status,
        "votes": [
            {
                "number": vote.number,
                "supportCount": vote.support_count,
                "weightShare": round(vote.weight_share, 6),
            }
            for vote in summary.votes
        ],
    }


def append_jersey_conflicts(
    conflicts: list[dict],
    *,
    canonical_id: str,
    track: TrackState,
    summary: JerseyEvidenceSummary,
    bound_roster_player: object | None,
) -> None:
    if summary.status == "conflict":
        conflicts.append(
            {
                "id": f"{canonical_id}:jersey-ocr-conflict",
                "code": "jersey-ocr-conflict",
                "message": (
                    "Independent jersey OCR readings disagree; no shirt number "
                    "or roster identity was accepted."
                ),
                "severity": "review",
                "relatedTrackletIds": list(summary.tracklet_ids),
                "reasons": list(summary.conflict_reasons),
            }
        )
    observed_number = normalize_jersey_number(summary.jersey_number)
    expected_number = normalize_jersey_number(
        getattr(bound_roster_player, "jersey_number", None)
    )
    if (
        summary.status != "reliable"
        or bound_roster_player is None
        or observed_number is None
        or expected_number is None
        or observed_number == expected_number
    ):
        return
    conflicts.append(
        {
            "id": f"{canonical_id}:manual-roster-jersey-conflict",
            "code": "manual-roster-jersey-conflict",
            "message": (
                "The confirmed roster player has a different shirt number from "
                "repeated reliable OCR; the manual binding was retained for review."
            ),
            "severity": "review",
            "externalPlayerId": getattr(bound_roster_player, "external_player_id", None),
            "expectedNumber": expected_number,
            "observedNumber": observed_number,
            "bindingAnnotationIds": sorted(track.roster_binding_annotation_ids),
            "relatedTrackletIds": list(summary.tracklet_ids),
        }
    )


def identity_evidence_projection(
    track: TrackState,
    *,
    canonical_id: str,
    observation_count: int,
    jersey_summary: JerseyEvidenceSummary | None,
    resolver_diagnostics: Mapping | None,
) -> list[dict]:
    evidence = deepcopy(track.identity_evidence)
    if jersey_summary is not None:
        evidence.append(
            _jersey_evidence_payload(
                canonical_id, jersey_summary, resolver_diagnostics
            )
        )
    if track.positive_annotation_ids:
        evidence.append(
            {
                "id": f"{canonical_id}:manual",
                "kind": "manual",
                "label": "Confirmed by frame annotation",
                "supportCount": len(track.positive_annotation_ids),
                "manual": True,
            }
        )
    if track.reid_feature_count:
        evidence.append(
            {
                "id": f"{canonical_id}:reid",
                "kind": "reid",
                "label": "Soccer player appearance embedding",
                "supportCount": track.reid_observation_count,
                "sampleCount": track.reid_feature_count,
                "uniqueEvidenceFingerprintCount": track.reid_observation_count,
                "duplicateEvidenceFingerprintCount": track.reid_duplicate_evidence_count,
                "source": "identity-worker",
                "selectionPolicy": (
                    "pixel-deduplicated-quality-ranked-temporally-separated-v2"
                ),
                "selectedFrameIndices": [
                    int(item["frameIndex"]) for item in track.reid_selected_metadata
                ],
                "selectedQualities": [
                    round(float(item["quality"]), 4)
                    for item in track.reid_selected_metadata
                ],
                "selectedEvidenceFingerprints": [
                    item.get("evidenceFingerprint")
                    for item in track.reid_selected_metadata
                    if item.get("evidenceFingerprint")
                ],
            }
        )
    evidence.append(
        {
            "id": f"{canonical_id}:trajectory",
            "kind": "trajectory",
            "label": "Continuous local tracklet observations",
            "supportCount": observation_count,
        }
    )
    return evidence
