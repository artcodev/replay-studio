from __future__ import annotations

"""Adapt local reconstruction tracklets to the audited global identity resolver."""

from copy import deepcopy
from dataclasses import asdict
from typing import Mapping

from .identity_resolution_contract import IdentityTracklet
from .identity_resolver import resolve_identities
from .jersey_ocr_contract import JerseyEvidenceSummary
from .reconstruction_track_state import TrackState
from .reconstruction_identity_merging import merge_raw_track_states
from .reconstruction_identity_semantics import annotation_role, annotation_team


def _tracklet_endpoint_pitch(point: dict) -> tuple[float, float] | None:
    if point.get("pitchX") is None or point.get("pitchZ") is None:
        return None
    return float(point["pitchX"]), float(point["pitchZ"])


def resolve_canonical_track_states(
    tracks: list[TrackState],
    preliminary_mapping: dict[int, str],
    jersey_evidence: Mapping[str, JerseyEvidenceSummary] | None = None,
) -> tuple[list[TrackState], dict]:
    """Run the conservative offline tracklet-to-identity resolver.

    Screen/pitch proximity only rejects impossible transitions. It can never
    create an identity link without ReID, reliable jersey OCR, an external
    roster ID, or an explicit manual decision.
    """

    inputs: list[IdentityTracklet] = []
    by_tracklet: dict[str, TrackState] = {}
    for track in tracks:
        if not track.points:
            continue
        points = sorted(
            track.points,
            key=lambda item: (float(item["t"]), item["frameIndex"]),
        )
        tracklet_id = track.local_tracklet_id
        by_tracklet[tracklet_id] = track
        external_id = track.manual_external_player_id
        positive_annotation_ids = track.positive_annotation_ids
        jersey_fields = (
            jersey_evidence[tracklet_id].identity_resolver_fields()
            if jersey_evidence is not None and tracklet_id in jersey_evidence
            else {
                "jersey_number": None,
                "jersey_confidence": 0.0,
                "jersey_sample_count": 0,
            }
        )
        inputs.append(
            IdentityTracklet(
                id=tracklet_id,
                start_time=float(points[0]["t"]),
                end_time=float(points[-1]["t"]),
                team_id=preliminary_mapping.get(track.id)
                or annotation_team(track.manual_kind),
                role=annotation_role(track.manual_kind) or track.role,
                external_player_id=external_id,
                jersey_number=jersey_fields["jersey_number"],
                jersey_confidence=float(jersey_fields["jersey_confidence"] or 0.0),
                jersey_sample_count=int(jersey_fields["jersey_sample_count"] or 0),
                mean_reid_embedding=(
                    tuple(float(value) for value in track.reid_feature)
                    if track.reid_feature is not None
                    else None
                ),
                reid_embeddings=tuple(
                    tuple(float(value) for value in sample)
                    for sample in track.reid_samples
                ),
                start_pitch=_tracklet_endpoint_pitch(points[0]),
                end_pitch=_tracklet_endpoint_pitch(points[-1]),
                start_uncertainty_metres=points[0].get(
                    "positionUncertaintyMetres"
                ),
                end_uncertainty_metres=points[-1].get(
                    "positionUncertaintyMetres"
                ),
                observation_count=len(points),
                manual_confirmed=bool(positive_annotation_ids or external_id),
                manual_identity_id=(
                    f"canonical:{next(iter(track.manual_identity_owner_ids))}"
                    if len(track.manual_identity_owner_ids) == 1
                    else f"external:{external_id}"
                    if external_id
                    else None
                ),
                manual_team=bool(
                    positive_annotation_ids
                    and annotation_team(track.manual_kind) is not None
                ),
                manual_role=bool(
                    positive_annotation_ids
                    and annotation_role(track.manual_kind) is not None
                ),
            )
        )

    resolution = resolve_identities(inputs)
    result: list[TrackState] = []
    review_by_tracklet: dict[str, list] = {}
    for edge in resolution.review_edges:
        review_by_tracklet.setdefault(edge.predecessor_id, []).append(edge)
        review_by_tracklet.setdefault(edge.successor_id, []).append(edge)

    for group in resolution.groups:
        members = [by_tracklet[tracklet_id] for tracklet_id in group.tracklet_ids]
        target = min(
            members,
            key=lambda item: (
                0
                if item.positive_annotation_ids or item.manual_external_player_id
                else 1,
                float(item.points[0]["t"]),
                item.id,
            ),
        )
        for source in members:
            if source is target:
                continue
            merge_raw_track_states(target, source)
        target.identity_group_id = group.id
        target.identity_status = group.status
        target.identity_confidence = float(group.confidence)
        if group.external_player_id and target.roster_binding_state is None:
            target.manual_external_player_id = group.external_player_id

        group_tracklets = set(group.tracklet_ids)
        for edge in resolution.accepted_edges:
            if (
                edge.predecessor_id not in group_tracklets
                or edge.successor_id not in group_tracklets
            ):
                continue
            reasons = set(edge.reasons)
            kind = (
                "manual"
                if edge.source == "manual"
                else "jersey-ocr"
                if "reliable-jersey-match" in reasons
                else "reid"
            )
            target.identity_evidence.append(
                {
                    "id": f"{group.id}:{edge.predecessor_id}:{edge.successor_id}",
                    "kind": kind,
                    "label": (
                        "Manual identity merge"
                        if edge.source == "manual"
                        else "Offline tracklet stitch"
                    ),
                    "value": ", ".join(edge.reasons),
                    "confidence": edge.score,
                    "source": edge.source,
                    "model": "global-tracklet-resolver-v1",
                    "manual": edge.source == "manual",
                }
            )
        seen_review_edges: set[tuple[str, str]] = set()
        for tracklet_id in group.tracklet_ids:
            for edge in review_by_tracklet.get(tracklet_id, []):
                edge_key = (edge.predecessor_id, edge.successor_id)
                if edge_key in seen_review_edges:
                    continue
                seen_review_edges.add(edge_key)
                target.identity_conflicts.append(
                    {
                        "id": f"review:{edge.predecessor_id}:{edge.successor_id}",
                        "code": "identity-association-review",
                        "message": (
                            f"Possible link {edge.predecessor_id} → "
                            f"{edge.successor_id} was not accepted "
                            f"({', '.join(edge.reasons)})."
                        ),
                        "severity": "review",
                        "relatedTrackletIds": [
                            edge.predecessor_id,
                            edge.successor_id,
                        ],
                    }
                )
        result.append(target)

    diagnostics = deepcopy(resolution.diagnostics)
    diagnostics.update(
        {
            "schemaVersion": 1,
            "provider": "global-tracklet-resolver-v1",
            "identityEvidencePolicy": "strong-reid-or-reliable-jersey-or-manual",
            "acceptedEdges": [asdict(edge) for edge in resolution.accepted_edges],
            "reviewEdges": [asdict(edge) for edge in resolution.review_edges],
            "jerseyReliableTrackletCount": sum(
                summary.status == "reliable"
                for summary in (jersey_evidence or {}).values()
            ),
            "jerseyProvisionalTrackletCount": sum(
                summary.status == "provisional"
                for summary in (jersey_evidence or {}).values()
            ),
            "jerseyConflictingTrackletCount": sum(
                summary.status == "conflict"
                for summary in (jersey_evidence or {}).values()
            ),
        }
    )
    return sorted(
        result,
        key=lambda item: (float(item.points[0]["t"]), item.id),
    ), diagnostics


__all__ = ["resolve_canonical_track_states"]
