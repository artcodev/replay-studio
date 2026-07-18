"""Map raw jersey evidence onto final or pre-resolver identity partitions."""

from __future__ import annotations

from dataclasses import replace
from .jersey_ocr_contract import JerseyEvidenceSummary, JerseyOcrObservation
from .jersey_ocr_fusion import aggregate_canonical_people, aggregate_tracklets
from .reconstruction_track_state import TrackState
from .reconstruction_jersey_policy import JERSEY_OCR_FUSION_CONFIG


def aggregate_jersey_evidence_for_final_tracks(
    tracks: list[TrackState],
    diagnostics: dict,
) -> tuple[dict[str, JerseyEvidenceSummary], dict]:
    """Reassign raw OCR evidence by immutable observation ID after split/merge."""

    crop_to_observation = {
        str(item.get("cropId")): str(item.get("observationId"))
        for item in diagnostics.get("crops") or []
        if item.get("cropId") and item.get("observationId")
    }
    owners: dict[str, set[str]] = {}
    for track in tracks:
        canonical_id = str(track.canonical_person_id or "")
        if not canonical_id:
            continue
        for point in track.points:
            observation_id = str(point.get("observationId") or "")
            if observation_id:
                owners.setdefault(observation_id, set()).add(canonical_id)

    raw_rows = [
        item
        for item in diagnostics.get("crops") or []
        if "ocrConfidence" in item and item.get("cropId") and item.get("trackletId")
    ]
    source_observations: list[JerseyOcrObservation] = []
    invalid_raw_crop_ids: list[str] = []
    for item in raw_rows:
        crop_id = str(item["cropId"])
        try:
            source_observations.append(
                JerseyOcrObservation(
                    id=crop_id,
                    tracklet_id=str(item["trackletId"]),
                    timestamp_seconds=float(item.get("timestamp") or 0.0),
                    raw_number=item.get("rawNumber"),
                    ocr_confidence=float(item.get("ocrConfidence") or 0.0),
                    frame_quality=float(item.get("frameQuality") or 0.0),
                    back_visibility=float(item.get("backVisibility") or 0.0),
                    frame_index=(
                        int(item["frameIndex"])
                        if item.get("frameIndex") is not None
                        else None
                    ),
                    source=str(item.get("source") or "jersey-ocr-worker"),
                    evidence_fingerprint=(
                        str(item.get("evidenceFingerprint"))
                        if item.get("evidenceFingerprint")
                        else None
                    ),
                )
            )
        except (TypeError, ValueError):
            invalid_raw_crop_ids.append(crop_id)

    reassigned: list[JerseyOcrObservation] = []
    unmapped_crop_ids: list[str] = []
    ambiguous_crop_ids: list[str] = []
    for observation in source_observations:
        video_observation_id = crop_to_observation.get(observation.id)
        if video_observation_id is None:
            unmapped_crop_ids.append(observation.id)
            continue
        canonical_owners = owners.get(video_observation_id) or set()
        if len(canonical_owners) != 1:
            (ambiguous_crop_ids if canonical_owners else unmapped_crop_ids).append(
                observation.id
            )
            continue
        reassigned.append(
            replace(
                observation,
                tracklet_id=f"final:{next(iter(canonical_owners))}",
            )
        )

    final_tracklets = aggregate_tracklets(
        reassigned,
        config=JERSEY_OCR_FUSION_CONFIG,
    )
    final_mapping = {
        tracklet_id: tracklet_id.removeprefix("final:")
        for tracklet_id in final_tracklets
    }
    canonical = (
        aggregate_canonical_people(
            final_tracklets,
            final_mapping,
            config=JERSEY_OCR_FUSION_CONFIG,
        )
        if final_tracklets
        else {}
    )
    return canonical, {
        "evidenceSource": "raw-crop-results",
        "rawCandidateCropCount": len(source_observations),
        "mappedRawCropCount": len(reassigned),
        "invalidRawCropIds": sorted(set(invalid_raw_crop_ids)),
        "finalSelectedCropCount": sum(
            summary.selected_sample_count for summary in final_tracklets.values()
        ),
        "unmappedRawCropIds": sorted(set(unmapped_crop_ids)),
        "ambiguousRawCropIds": sorted(set(ambiguous_crop_ids)),
        "finalTrackletCount": len(final_tracklets),
    }


def partition_local_jersey_evidence_for_resolver(
    tracks: list[TrackState],
    diagnostics: dict,
) -> tuple[dict[str, JerseyEvidenceSummary], dict]:
    """Re-key raw OCR evidence to each pre-resolver split partition."""

    original_ids = {track.id: track.canonical_person_id for track in tracks}
    temporary_ids = {
        track.id: f"resolver-partition:{track.local_tracklet_id}" for track in tracks
    }
    try:
        for track in tracks:
            track.canonical_person_id = temporary_ids[track.id]
        by_temporary_id, mapping_diagnostics = (
            aggregate_jersey_evidence_for_final_tracks(tracks, diagnostics)
        )
    finally:
        for track in tracks:
            track.canonical_person_id = original_ids[track.id]
    return (
        {
            track.local_tracklet_id: by_temporary_id[temporary_ids[track.id]]
            for track in tracks
            if temporary_ids[track.id] in by_temporary_id
        },
        mapping_diagnostics,
    )
