from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .pitch_calibration_contract import PitchCalibration
from .reconstruction_person_detection_contract import Detection
from .reconstruction_identity_correction_service import apply_track_identity_corrections
from .reconstruction_canonical_identity_resolution import (
    resolve_canonical_track_states,
)
from .reconstruction_identity_scene_corrections import (
    apply_scene_track_identity_corrections,
)
from .reconstruction_identity_splitting import apply_canonical_split_corrections
from .reconstruction_canonical_people_projection import canonical_people_documents
from .reconstruction_identity_persistence import assign_persistent_canonical_person_ids
from .reconstruction_jersey_inference import run_jersey_ocr_for_tracklets
from .reconstruction_jersey_resolution import (
    aggregate_jersey_evidence_for_final_tracks,
    partition_local_jersey_evidence_for_resolver,
)
from .jersey_ocr_fusion import detect_jersey_number_switches
from .reconstruction_identity_semantics import annotation_role, annotation_team
from .reconstruction_coordinate_policy import METRIC_REQUIRED
from .reconstruction_person_tracking import track_people
from .reconstruction_progress import ReconstructionProgress
from .reconstruction_scene_track_publisher import (
    publish_provisional_canonical_tracks,
    publish_scene_tracks,
)
from .reconstruction_team_classification import (
    include_reid_official_candidates,
    team_clusters,
)


@dataclass(frozen=True)
class IdentityPhaseResult:
    tracks: list[dict]
    canonical_people: list[dict]
    canonical_identity_diagnostics: dict
    track_projection_diagnostics: dict
    jersey_ocr_diagnostics: dict
    warnings: list[str]
    raw_track_count: int
    stable_track_count: int
    team_colors: dict[str, str]


def track_and_resolve_identity_phase(
    scene: dict,
    frames: list[tuple[Path, float]],
    person_frames: list[tuple[list[Detection], float]],
    frame_size: tuple[int, int],
    coordinate_mode: str,
    resolved_calibrations_by_sample: Mapping[int, PitchCalibration],
    calibration: PitchCalibration | None,
    progress: ReconstructionProgress,
    match_snapshot: Mapping[str, object] | None,
    identity_worker_diagnostics: dict,
    initial_warnings: list[str],
    *,
    jersey_ocr_profile: str = "automatic",
    tracking_coordinate_policy: str = METRIC_REQUIRED,
    image_fallback_sample_indices: Sequence[int] = (),
) -> IdentityPhaseResult:
    identity_warnings = list(initial_warnings)
    progress.update(
        "tracking",
        4,
        "Linking observations into tracks",
        f"Associating detections across {len(frames)} frames.",
        62,
        86,
        completed=0,
        total=4,
    )
    short_horizon_association: dict = {}
    local_tracks = apply_track_identity_corrections(
        track_people(
            person_frames,
            coordinate_policy=tracking_coordinate_policy,
            image_fallback_sample_indices=image_fallback_sample_indices,
            diagnostics=short_horizon_association,
        ),
        scene,
    )
    progress.update(
        "tracking",
        4,
        "Building local tracklets",
        f"Built {len(local_tracks)} local tracks; preparing team and role constraints.",
        62,
        86,
        completed=1,
        total=4,
    )
    minimum = max(5, round(len(frames) * 0.24))
    preliminary_stable_tracks = [
        track
        for track in local_tracks
        if len(track.points) >= minimum or track.positive_annotation_ids
    ]
    preliminary_cluster_tracks = [
        track
        for track in preliminary_stable_tracks
        if len(track.points) >= minimum
        and track.role != "referee"
        and annotation_role(track.manual_kind) != "referee"
    ]
    preliminary_mapping, _ = team_clusters(
        preliminary_cluster_tracks,
        frame_size[0],
    )
    for track in preliminary_stable_tracks:
        manual_team = annotation_team(track.manual_kind)
        if manual_team:
            preliminary_mapping[track.id] = manual_team

    def jersey_ocr_progress(completed: int, total: int, recognized: int) -> None:
        progress.update(
            "tracking",
            4,
            "Reading jersey numbers",
            (
                f"OCR crops {completed}/{total} · {recognized} readable "
                "shirt-number observations."
            ),
            62,
            86,
            completed=1,
            total=4,
            eta_padding=2.0,
        )

    if jersey_ocr_profile == "off":
        # The queued profile disabled OCR explicitly (manually bound roster):
        # cheaper runs at the cost of automatic shirt-number merge evidence.
        jersey_tracklet_evidence = {}
        jersey_ocr_diagnostics = {
            "status": "skipped-by-profile",
            "skippedByProfile": True,
            "profile": jersey_ocr_profile,
            "submittedCropCount": 0,
            "crops": [],
        }
        jersey_ocr_warnings = [
            "Jersey OCR was skipped by the analysis profile; automatic "
            "shirt-number merge evidence is absent for this run."
        ]
    else:
        (
            jersey_tracklet_evidence,
            jersey_ocr_diagnostics,
            jersey_ocr_warnings,
        ) = run_jersey_ocr_for_tracklets(
            local_tracks,
            frames,
            jersey_ocr_progress,
            scene=scene,
        )
        switch_suspects = detect_jersey_number_switches(
            jersey_ocr_diagnostics.get("crops") or []
        )
        jersey_ocr_diagnostics["numberSwitchSuspects"] = switch_suspects
        if switch_suspects:
            flagged = ", ".join(
                f"{item['trackletId']} ({item['fromNumber']}→{item['toNumber']} "
                f"at {item['switchTime']}s)"
                for item in switch_suspects[:5]
            )
            jersey_ocr_warnings = [
                *jersey_ocr_warnings,
                f"{len(switch_suspects)} tracklet(s) show a stable shirt-number "
                f"change mid-track (possible tracker ID switch): {flagged}. "
                "Review a manual Split at the flagged times.",
            ]
    identity_warnings.extend(jersey_ocr_warnings)
    partitioned_tracks, split_identity_diagnostics = apply_canonical_split_corrections(
        local_tracks,
        scene,
    )
    resolver_jersey_evidence = jersey_tracklet_evidence
    if split_identity_diagnostics["appliedCount"]:
        (
            resolver_jersey_evidence,
            split_jersey_mapping_diagnostics,
        ) = partition_local_jersey_evidence_for_resolver(
            partitioned_tracks,
            jersey_ocr_diagnostics,
        )
        jersey_ocr_diagnostics["preResolverSplitObservationMapping"] = (
            split_jersey_mapping_diagnostics
        )
    partitioned_mapping = {
        track.id: (
            annotation_team(track.manual_kind)
            or preliminary_mapping.get(track.id)
        )
        for track in partitioned_tracks
        if annotation_team(track.manual_kind)
        or preliminary_mapping.get(track.id)
    }
    canonical_tracks, identity_resolution_diagnostics = resolve_canonical_track_states(
        partitioned_tracks,
        partitioned_mapping,
        resolver_jersey_evidence,
    )
    identity_resolution_diagnostics["manualSplits"] = split_identity_diagnostics
    identity_resolution_diagnostics[
        "shortHorizonAssociation"
    ] = short_horizon_association
    identity_resolution_diagnostics["reid"] = deepcopy(identity_worker_diagnostics)
    identity_resolution_diagnostics["jerseyOcr"] = deepcopy(
        jersey_ocr_diagnostics
    )
    progress.update(
        "tracking",
        4,
        "Resolving canonical people",
        (
            f"Resolved {len(local_tracks)} local tracklets into "
            f"{len(canonical_tracks)} canonical people; ambiguous links remain provisional."
        ),
        62,
        86,
        completed=2,
        total=4,
    )
    stable_tracks = [
        track
        for track in canonical_tracks
        if len(track.points) >= minimum or track.positive_annotation_ids
    ]
    cluster_tracks = [
        track
        for track in stable_tracks
        if len(track.points) >= minimum
        and track.role != "referee"
        and annotation_role(track.manual_kind) != "referee"
    ]
    team_classification_diagnostics: dict = {}
    mapping, colors = team_clusters(
        cluster_tracks,
        frame_size[0],
        diagnostics=team_classification_diagnostics,
    )
    for track in stable_tracks:
        manual_team = annotation_team(track.manual_kind)
        if manual_team:
            mapping[track.id] = manual_team
    auto_official_tracklets = include_reid_official_candidates(
        stable_tracks,
        mapping,
    )
    identity_resolution_diagnostics["autoOfficials"] = {
        "count": len(auto_official_tracklets),
        "trackletIds": auto_official_tracklets,
        "source": "reid-role-votes",
    }
    identity_resolution_diagnostics[
        "teamClassification"
    ] = team_classification_diagnostics
    assign_persistent_canonical_person_ids(canonical_tracks, scene, mapping)
    try:
        (
            canonical_jersey_evidence,
            final_jersey_mapping_diagnostics,
        ) = aggregate_jersey_evidence_for_final_tracks(
            canonical_tracks,
            jersey_ocr_diagnostics,
        )
        jersey_ocr_diagnostics["canonicalAggregationStatus"] = "ready"
        jersey_ocr_diagnostics["finalObservationMapping"] = (
            final_jersey_mapping_diagnostics
        )
    except ValueError as exc:
        # Jersey OCR is optional identity evidence. A bad mapping remains
        # visible, but cannot make an otherwise valid reconstruction fail.
        canonical_jersey_evidence = {}
        jersey_ocr_diagnostics.update(
            {
                "canonicalAggregationStatus": "failed",
                "canonicalAggregationDetail": str(exc),
            }
        )
        identity_warnings.append(
            "Jersey OCR canonical aggregation failed; shirt numbers were omitted from this reconstruction."
        )
    jersey_ocr_diagnostics.update(
        {
            "canonicalPersonEvidence": {
                canonical_id: summary.to_payload()
                for canonical_id, summary in sorted(
                    canonical_jersey_evidence.items()
                )
            },
            "reliableCanonicalPersonCount": sum(
                summary.status == "reliable"
                for summary in canonical_jersey_evidence.values()
            ),
            "provisionalCanonicalPersonCount": sum(
                summary.status == "provisional"
                for summary in canonical_jersey_evidence.values()
            ),
            "conflictingCanonicalPersonCount": sum(
                summary.status == "conflict"
                for summary in canonical_jersey_evidence.values()
            ),
        }
    )
    identity_resolution_diagnostics["jerseyOcr"] = deepcopy(
        jersey_ocr_diagnostics
    )
    progress.update(
        "tracking",
        4,
        "Assigning teams and roles",
        (
            f"Kept {len(stable_tracks)} renderable identities and preserved "
            f"{len(canonical_tracks)} video identities."
        ),
        62,
        86,
        completed=4,
        total=4,
        eta_padding=3.0,
    )
    progress.update(
        "projection",
        5,
        "Building metric 3D trajectories",
        "Projecting foot points and the ball onto the pitch, rejecting geometric outliers.",
        86,
        96,
        completed=0,
        total=2,
    )
    track_projection_diagnostics: dict = {}
    track_projection_diagnostics["trackingAssociation"] = {
        key: deepcopy(value)
        for key, value in short_horizon_association.items()
        if key != "frames"
    }
    tracks = (
        publish_scene_tracks(
            canonical_tracks,
            mapping,
            colors,
            frame_size,
            scene,
            calibration,
            coordinate_mode=coordinate_mode,
            diagnostics=track_projection_diagnostics,
        )
        if coordinate_mode != "unavailable"
        else []
    )
    # A canonical person without a confident team/roster assignment is still a
    # real detector-backed actor. Publish it as provisional identity rather than
    # converting its whole trajectory into a translucent positional guess.
    if coordinate_mode != "unavailable":
        tracks = [
            *tracks,
            *publish_provisional_canonical_tracks(
                canonical_tracks,
                tracks,
                mapping,
                colors,
                frame_size,
                scene,
                calibration,
                coordinate_mode=coordinate_mode,
            ),
        ]
    tracks = apply_scene_track_identity_corrections(tracks, scene)
    canonical_people, canonical_identity_diagnostics = canonical_people_documents(
        canonical_tracks,
        mapping,
        tracks,
        scene,
        identity_resolution_diagnostics,
        canonical_jersey_evidence,
        match_snapshot,
    )
    return IdentityPhaseResult(
        tracks=tracks,
        canonical_people=canonical_people,
        canonical_identity_diagnostics=canonical_identity_diagnostics,
        track_projection_diagnostics=track_projection_diagnostics,
        jersey_ocr_diagnostics=jersey_ocr_diagnostics,
        warnings=identity_warnings,
        raw_track_count=len(local_tracks),
        stable_track_count=len(stable_tracks),
        team_colors=colors,
    )
