from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.identity_decisions import reject_roster_candidate
from app.jersey_ocr_fusion import aggregate_canonical_people
from app.jersey_ocr_worker_contract import JerseyOcrBatchResult, JerseyOcrWorkerError
from app.reconstruction_person_detection_contract import Detection
from app.reconstruction_track_state import TrackState
from app.track_observation_accumulator import append_track_observation
from app.reconstruction_identity_splitting import (
    apply_canonical_split_corrections as _apply_canonical_split_corrections,
)
from app.reconstruction_canonical_people_projection import (
    canonical_people_documents as _canonical_people_documents,
)
from app.reconstruction_identity_persistence import (
    assign_persistent_canonical_person_ids as _assign_persistent_canonical_person_ids,
)
from app.reconstruction_jersey_inference import (
    run_jersey_ocr_for_tracklets as _run_jersey_ocr_for_tracklets,
)
from app.reconstruction_jersey_policy import JERSEY_OCR_FUSION_CONFIG
from app.reconstruction_jersey_resolution import (
    aggregate_jersey_evidence_for_final_tracks as _aggregate_jersey_evidence_for_final_tracks,
    partition_local_jersey_evidence_for_resolver as _partition_local_jersey_evidence_for_resolver,
)
from app.reconstruction_reid_evidence import (
    capture_detection_observations as _capture_detection_observations,
)
from app.reconstruction_canonical_identity_resolution import (
    resolve_canonical_track_states as _resolve_canonical_track_states,
)


def _scene(
    *,
    players: list[dict] | None = None,
    automatic_identity_eligible: bool | None = None,
) -> dict:
    roster_players = players or []
    if automatic_identity_eligible is None:
        automatic_identity_eligible = bool(roster_players)
    return {
        "id": "jersey-scene",
        "duration": 4.0,
        "_testMatchSnapshot": {
            "homeTeam": {"id": "home-api", "name": "Home"},
            "awayTeam": {"id": "away-api", "name": "Away"},
            "roster": roster_players,
            "rosterQuality": {
                "status": (
                    "automatic-ready"
                    if automatic_identity_eligible
                    else "partial"
                ),
                "automaticIdentityEligible": automatic_identity_eligible,
                "manualIdentityEligible": bool(roster_players),
                "reasons": [],
            },
        },
        "payload": {
            "pitch": {"length": 105, "width": 68},
            "teams": [
                {"id": "home", "color": "#ffffff"},
                {"id": "away", "color": "#000000"},
            ],
            "tracks": [],
            "canonicalPeople": [],
            "ball": {"keyframes": []},
            "videoAsset": {
                "sourceStart": 0.0,
                "selectedSegmentId": "segment-1",
                "reconstruction": {"status": "processing"},
            },
        },
    }


def _tracks_and_frames(
    tmp_path: Path,
    *,
    track_count: int = 1,
    samples_per_track: int = 3,
) -> tuple[list[TrackState], list[tuple[Path, float]]]:
    source_frames = [1 + index * 10 for index in range(samples_per_track)]
    frames: list[tuple[Path, float]] = []
    for sample_index, source_frame in enumerate(source_frames):
        path = tmp_path / f"frame_{source_frame:05d}.jpg"
        # Textured crops avoid making this fixture depend on blur thresholds
        # when a test temporarily exercises the real client implementation.
        image = np.random.default_rng(source_frame).integers(
            0, 255, (240, 360, 3), dtype=np.uint8
        )
        assert cv2.imwrite(str(path), image)
        frames.append((path, float(sample_index)))

    tracks: list[TrackState] = []
    for track_index in range(track_count):
        track = TrackState(id=track_index + 1, role="player")
        for sample_index, source_frame in enumerate(source_frames):
            detection = Detection(
                x=100.0 + track_index * 90.0 + sample_index * 2.0,
                y=190.0,
                width=42.0,
                height=100.0,
                confidence=0.94,
                feature=np.ones(12, dtype=np.float32) * (track_index + 1) * 0.1,
                pitch_x=float(track_index * 8 + sample_index),
                pitch_z=0.0,
                position_uncertainty_metres=0.5,
            )
            _capture_detection_observations([detection], source_frame)
            append_track_observation(track, detection, sample_index, float(sample_index))
        tracks.append(track)
    return tracks, frames


def _recognized(number: int, confidence: float = 0.90) -> dict:
    return {
        "usable": True,
        "status": "recognized",
        "number": str(number),
        "confidence": confidence,
        "candidates": [{"number": str(number), "confidence": confidence}],
        "quality": {"cropWidth": 42, "cropHeight": 100, "sharpness": 80.0},
        "rejectionReasons": [],
        "decisionReasons": [],
        "provider": "fake-jersey-ocr",
        "modelVersion": "fake-v1",
    }


def _install_worker(monkeypatch, values: dict[tuple[str, int], dict]) -> list:
    submitted = []
    monkeypatch.setattr(
        "app.reconstruction_jersey_inference.jersey_ocr_worker_readiness",
        lambda **_: {
            "configured": True,
            "status": "ready",
            "backend": "fake-jersey-ocr",
            "modelVersion": "fake-v1",
            "inferenceScope": "crop",
        },
    )

    def analyze(requests, on_progress=None):
        submitted.extend(requests)
        assert all(request.path.is_file() for request in requests)
        result: dict[str, dict] = {}
        recognized = 0
        for request in requests:
            item = values[(str(request.tracklet_id), int(request.frame_index))]
            recognized += int(item["status"] == "recognized")
            result[request.crop_id] = {"cropId": request.crop_id, **item}
        if on_progress is not None:
            on_progress(len(requests), len(requests), recognized)
        return JerseyOcrBatchResult(items_by_crop_id=result)

    monkeypatch.setattr("app.reconstruction_jersey_inference.analyze_jersey_crops", analyze)
    return submitted


def _publish(
    tracks: list[TrackState],
    scene: dict,
    tracklet_evidence,
    ocr_diagnostics: dict,
):
    preliminary_mapping = {track.id: "home" for track in tracks}
    resolved, resolver_diagnostics = _resolve_canonical_track_states(
        tracks,
        preliminary_mapping,
        tracklet_evidence,
    )
    _assign_persistent_canonical_person_ids(resolved, scene, preliminary_mapping)
    tracklet_to_canonical = {
        tracklet_id: str(track.canonical_person_id)
        for track in resolved
        for tracklet_id in (track.source_tracklet_ids or {track.local_tracklet_id})
        if tracklet_id in tracklet_evidence
    }
    canonical_evidence = aggregate_canonical_people(
        tracklet_evidence,
        tracklet_to_canonical,
        config=JERSEY_OCR_FUSION_CONFIG,
    )
    resolver_diagnostics["jerseyOcr"] = ocr_diagnostics
    people, diagnostics = _canonical_people_documents(
        resolved,
        {track.id: "home" for track in resolved},
        [],
        scene,
        resolver_diagnostics,
        canonical_evidence,
        scene.get("_testMatchSnapshot"),
    )
    return resolved, people, diagnostics


def test_reliable_two_to_one_vote_publishes_jersey_and_review_only_roster_candidate(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path)
    values = {
        ("tracklet-0001", 1): _recognized(8),
        ("tracklet-0001", 11): _recognized(8),
        ("tracklet-0001", 21): _recognized(3),
    }
    submitted = _install_worker(monkeypatch, values)

    summaries, ocr_diagnostics, warnings = _run_jersey_ocr_for_tracklets(
        tracks,
        frames,
    )

    assert len(submitted) == 3
    assert warnings == []
    assert summaries["tracklet-0001"].status == "reliable"
    assert summaries["tracklet-0001"].jersey_number == "8"
    assert ocr_diagnostics["recognizedCropCount"] == 3

    scene = _scene(
        players=[
            {
                "id": "player-8",
                "name": "Player Eight",
                "team_id": "home-api",
                "number": "08",
                "position": "Forward",
            },
            {
                "id": "player-3",
                "name": "Player Three",
                "team_id": "home-api",
                "number": "3",
                "position": "Defender",
            },
        ]
    )
    _, people, diagnostics = _publish(tracks, scene, summaries, ocr_diagnostics)
    person = people[0]

    assert person["jerseyNumber"] == "8"
    assert person["candidateNumber"] == "8"
    assert person["externalPlayerId"] is None
    assert len(person["rosterCandidates"]) == 1
    candidate = person["rosterCandidates"][0]
    assert candidate["externalPlayerId"] == "player-8"
    assert candidate["name"] == "Player Eight"
    assert candidate["number"] == "8"
    assert candidate["teamId"] == "home-api"
    assert candidate["position"] == "Forward"
    assert candidate["proposalStatus"] == "selected"
    assert candidate["requiresManualConfirmation"] is True
    assert candidate["confidence"] == candidate["score"]
    assert candidate["identitySignalScore"] > 0.3
    assert {item["code"] for item in candidate["evidence"]} >= {
        "team-match",
        "jersey-number-match",
        "role-match",
    }
    assert person["rosterResolution"]["status"] == "suggested"
    assert person["externalPlayerId"] is None
    assert any(
        item["kind"] == "jersey-ocr" and item["status"] == "reliable"
        for item in person["evidence"]
    )
    assert diagnostics["rosterCandidateCount"] == 1
    assert diagnostics["closedSetRoster"]["automaticBindingCount"] == 0
    assert diagnostics["closedSetRoster"]["oneToOneSuggestions"] is True


def test_incomplete_roster_never_drives_automatic_name_candidates(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path, samples_per_track=2)
    values = {
        ("tracklet-0001", 1): _recognized(8),
        ("tracklet-0001", 11): _recognized(8),
    }
    _install_worker(monkeypatch, values)
    summaries, ocr_diagnostics, _ = _run_jersey_ocr_for_tracklets(tracks, frames)
    scene = _scene(
        players=[
            {
                "id": "player-8",
                "name": "Player Eight",
                "team_id": "home-api",
                "number": "8",
            }
        ],
        automatic_identity_eligible=False,
    )

    _, people, diagnostics = _publish(tracks, scene, summaries, ocr_diagnostics)

    assert people[0]["jerseyNumber"] == "8"
    assert people[0]["rosterCandidates"] == []
    assert diagnostics["rosterCandidateCount"] == 0
    assert diagnostics["rosterPrior"]["automaticIdentityEligible"] is False


def test_rejected_roster_hypothesis_is_not_republished_on_rebuild(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path, samples_per_track=2)
    _install_worker(
        monkeypatch,
        {
            ("tracklet-0001", 1): _recognized(8),
            ("tracklet-0001", 11): _recognized(8),
        },
    )
    summaries, ocr_diagnostics, _ = _run_jersey_ocr_for_tracklets(tracks, frames)
    scene = _scene(
        players=[
            {
                "id": "player-8",
                "name": "Player Eight",
                "team_id": "home-api",
                "number": "8",
            }
        ]
    )

    _, first_people, _ = _publish(tracks, scene, summaries, ocr_diagnostics)
    scene["payload"]["canonicalPeople"] = first_people
    canonical_id = first_people[0]["canonicalPersonId"]
    assert first_people[0]["rosterCandidates"][0]["externalPlayerId"] == "player-8"

    reject_roster_candidate(
        scene,
        canonical_id,
        "player-8",
        match_snapshot=scene["_testMatchSnapshot"],
    )
    _, rebuilt_people, diagnostics = _publish(
        tracks,
        scene,
        summaries,
        ocr_diagnostics,
    )

    assert rebuilt_people[0]["rosterCandidates"] == []
    assert rebuilt_people[0]["rosterResolution"]["status"] == "abstain"
    assert diagnostics["rosterCandidateCount"] == 0


def test_missing_raw_ocr_rows_do_not_reuse_pre_resolver_selection(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path)
    _install_worker(
        monkeypatch,
        {
            ("tracklet-0001", 1): _recognized(8),
            ("tracklet-0001", 11): _recognized(8),
            ("tracklet-0001", 21): _recognized(3),
        },
    )
    summaries, _, _ = _run_jersey_ocr_for_tracklets(tracks, frames)
    tracks[0].canonical_person_id = "canonical-only"

    canonical, mapping = _aggregate_jersey_evidence_for_final_tracks(
        tracks,
        {},
    )

    assert mapping["evidenceSource"] == "raw-crop-results"
    assert mapping["mappedRawCropCount"] == 0
    assert canonical == {}


def test_missing_raw_ocr_rows_abstain_after_a_source_tracklet_split(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path)
    _install_worker(
        monkeypatch,
        {
            ("tracklet-0001", 1): _recognized(8),
            ("tracklet-0001", 11): _recognized(8),
            ("tracklet-0001", 21): _recognized(8),
        },
    )
    summaries, _, _ = _run_jersey_ocr_for_tracklets(tracks, frames)
    remaining = deepcopy(tracks[0])
    split = deepcopy(tracks[0])
    remaining.canonical_person_id = "canonical-remaining"
    split.canonical_person_id = "canonical-split"
    remaining.points = remaining.points[:2]
    split.points = split.points[2:]

    canonical, mapping = _aggregate_jersey_evidence_for_final_tracks(
        [remaining, split],
        {},
    )

    assert canonical == {}
    assert mapping["mappedRawCropCount"] == 0
    assert mapping["ambiguousRawCropIds"] == []


def test_manual_roster_binding_survives_reliable_ocr_disagreement_with_explicit_conflict(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path)
    _install_worker(
        monkeypatch,
        {
            ("tracklet-0001", 1): _recognized(9),
            ("tracklet-0001", 11): _recognized(9),
            ("tracklet-0001", 21): _recognized(9),
        },
    )
    summaries, diagnostics, _ = _run_jersey_ocr_for_tracklets(tracks, frames)
    track = tracks[0]
    track.manual_external_player_id = "player-8"
    track.roster_binding_state = "bound"
    track.roster_binding_annotation_ids = {"bind-player-8"}
    track.annotation_ids = {"bind-player-8"}
    track.manual_kind = "home-player"
    scene = _scene(
        players=[
            {
                "id": "player-8",
                "name": "Player Eight",
                "team_id": "home-api",
                "number": "08",
            },
            {
                "id": "player-9",
                "name": "Player Nine",
                "team_id": "home-api",
                "number": "9",
            },
        ]
    )

    _, people, identity_diagnostics = _publish(
        tracks,
        scene,
        summaries,
        diagnostics,
    )

    person = people[0]
    assert person["externalPlayerId"] == "player-8"
    assert person["jerseyNumber"] == "9"
    assert person["rosterCandidates"] == []
    conflict = next(
        item
        for item in person["conflicts"]
        if item["code"] == "manual-roster-jersey-conflict"
    )
    assert conflict["expectedNumber"] == "8"
    assert conflict["observedNumber"] == "9"
    assert conflict["bindingAnnotationIds"] == ["bind-player-8"]
    assert identity_diagnostics["conflictPersonCount"] == 1
    assert identity_diagnostics["manualRosterJerseyConflictCount"] == 1


def test_missing_bound_player_in_replaced_roster_is_an_explicit_conflict(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path)
    _install_worker(
        monkeypatch,
        {
            ("tracklet-0001", 1): _recognized(8),
            ("tracklet-0001", 11): _recognized(8),
            ("tracklet-0001", 21): _recognized(8),
        },
    )
    summaries, diagnostics, _ = _run_jersey_ocr_for_tracklets(tracks, frames)
    track = tracks[0]
    track.manual_external_player_id = "removed-player-8"
    track.roster_binding_state = "bound"
    track.roster_binding_annotation_ids = {"bind-removed-player"}
    track.annotation_ids = {"bind-removed-player"}
    track.manual_kind = "home-player"
    scene = _scene(
        players=[
            {
                "id": "replacement-player-8",
                "name": "Replacement Eight",
                "team_id": "home-api",
                "number": "8",
            }
        ]
    )

    _, people, identity_diagnostics = _publish(
        tracks,
        scene,
        summaries,
        diagnostics,
    )

    person = people[0]
    assert person["externalPlayerId"] == "removed-player-8"
    assert person["rosterCandidates"] == []
    conflict = next(
        item
        for item in person["conflicts"]
        if item["code"] == "manual-roster-player-missing"
    )
    assert conflict["externalPlayerId"] == "removed-player-8"
    assert conflict["bindingAnnotationIds"] == ["bind-removed-player"]
    assert conflict["rosterStatus"] == "ready"
    assert identity_diagnostics["manualRosterMissingConflictCount"] == 1


def test_manual_split_reassigns_ocr_by_immutable_observation_not_old_tracklet(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path)
    _install_worker(
        monkeypatch,
        {
            ("tracklet-0001", 1): _recognized(8),
            ("tracklet-0001", 11): _recognized(9),
            ("tracklet-0001", 21): _recognized(9),
        },
    )
    summaries, diagnostics, _ = _run_jersey_ocr_for_tracklets(tracks, frames)
    assert summaries["tracklet-0001"].jersey_number == "9"

    remaining = tracks[0]
    split = deepcopy(remaining)
    remaining.canonical_person_id = "canonical-remaining"
    split.canonical_person_id = "canonical-split"
    remaining.points = remaining.points[1:]
    split.points = split.points[:1]

    canonical, mapping_diagnostics = _aggregate_jersey_evidence_for_final_tracks(
        [remaining, split],
        diagnostics,
    )

    assert canonical["canonical-remaining"].jersey_number == "9"
    assert canonical["canonical-remaining"].support_count == 2
    assert canonical["canonical-split"].jersey_number is None
    assert canonical["canonical-split"].candidate_number == "8"
    assert mapping_diagnostics["mappedRawCropCount"] == 3


def test_split_is_partitioned_before_ocr_can_stitch_a_continuation_to_wrong_side(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(
        tmp_path,
        track_count=2,
        samples_per_track=4,
    )
    source, continuation = tracks
    for point in continuation.points:
        point["t"] = float(point["t"]) + 4.0
    no_number = {
        "usable": True,
        "status": "no-number",
        "number": None,
        "confidence": 0.0,
        "candidates": [],
        "quality": {"cropWidth": 42, "cropHeight": 100, "sharpness": 80.0},
        "rejectionReasons": [],
        "decisionReasons": [],
        "provider": "fake-jersey-ocr",
        "modelVersion": "fake-v1",
    }
    _install_worker(
        monkeypatch,
        {
            (source.local_tracklet_id, 1): no_number,
            (source.local_tracklet_id, 11): _recognized(8),
            (source.local_tracklet_id, 21): _recognized(8),
            (source.local_tracklet_id, 31): no_number,
            **{
                (continuation.local_tracklet_id, frame_index): _recognized(8)
                for frame_index in (1, 11, 21, 31)
            },
        },
    )
    summaries, diagnostics, _ = _run_jersey_ocr_for_tracklets(tracks, frames)
    assert summaries[source.local_tracklet_id].jersey_number == "8"
    assert summaries[continuation.local_tracklet_id].jersey_number == "8"

    anchor = source.points[1]
    split = {
        "id": "split-source-middle",
        "kind": "home-player",
        "action": "split",
        "scope": "range",
        "canonicalPersonId": "canonical-source",
        "targetObservationId": anchor["observationId"],
        "targetObservation": {
            "observationId": anchor["observationId"],
            "frameIndex": anchor["frameIndex"],
            "sceneTime": anchor["t"],
            "bbox": deepcopy(anchor["bbox"]),
            "canonicalPersonId": "canonical-source",
        },
        "rangeStart": 1.0,
        "rangeEnd": 3.0,
        "splitCanonicalPersonId": "canonical-split-middle",
    }
    scene = {
        "duration": 8.0,
        "payload": {
            "videoAsset": {"reconstruction": {"frameAnnotations": [split]}},
        },
    }

    partitioned, _ = _apply_canonical_split_corrections(tracks, scene)
    partition_evidence, mapping = _partition_local_jersey_evidence_for_resolver(
        partitioned,
        diagnostics,
    )
    assert mapping["mappedRawCropCount"] == 8
    resolved, _ = _resolve_canonical_track_states(
        partitioned,
        {track.id: "home" for track in partitioned},
        partition_evidence,
    )

    remaining = next(
        track for track in resolved if "canonical-source" in track.manual_identity_owner_ids
    )
    split_with_continuation = next(
        track
        for track in resolved
        if "canonical-split-middle" in track.manual_identity_owner_ids
    )
    assert [point["t"] for point in remaining.points] == [0.0, 3.0]
    assert [point["t"] for point in split_with_continuation.points] == [
        1.0,
        2.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]


def test_manual_split_fuses_partition_local_raw_crops_excluded_from_pre_resolver_top_five(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path, samples_per_track=8)
    track = tracks[0]
    split_target = track.points[5]
    scene = _scene()
    scene["payload"]["canonicalPeople"] = [
        {
            "id": "canonical-source",
            "canonicalPersonId": "canonical-source",
            "sourceTrackletIds": [track.local_tracklet_id],
            "observations": [deepcopy(point) for point in track.points],
        }
    ]
    scene["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] = [
        {
            "id": "split-correction-1",
            "action": "split",
            "canonicalPersonId": "canonical-source",
            "targetObservationId": split_target["observationId"],
            "rangeStart": 5.0,
            "rangeEnd": 8.0,
            "splitCanonicalPersonId": "canonical-split",
        }
    ]
    _install_worker(
        monkeypatch,
        {
            (track.local_tracklet_id, int(point["frameIndex"])): _recognized(
                9 if float(point["t"]) < 5.0 else 8,
                0.95,
            )
            for point in track.points
        },
    )

    summaries, diagnostics, _ = _run_jersey_ocr_for_tracklets(
        tracks,
        frames,
        scene=scene,
    )

    # The resolver receives the same conservative top-five it did before the
    # split-aware pool was introduced.  The later range's three readings are
    # retained as raw worker results rather than influencing this early link.
    pre_resolver = summaries[track.local_tracklet_id]
    assert pre_resolver.jersey_number == "9"
    assert pre_resolver.selected_sample_count == 5
    assert diagnostics["candidateCropCount"] == 8
    assert diagnostics["selectedCropCount"] == 8
    assert diagnostics["preResolverSelectedCropCount"] == 5
    assert diagnostics["rawObservationCount"] == 8
    assert diagnostics["prospectiveSplitRangeCount"] == 1
    pre_resolver_crop_ids = {
        observation.id for observation in pre_resolver.selected_observations
    }

    remaining = deepcopy(track)
    split = deepcopy(track)
    remaining.canonical_person_id = "canonical-remaining"
    split.canonical_person_id = "canonical-split"
    remaining.points = remaining.points[:5]
    split.points = split.points[5:]
    canonical, mapping_diagnostics = _aggregate_jersey_evidence_for_final_tracks(
        [remaining, split],
        diagnostics,
    )

    assert canonical["canonical-remaining"].jersey_number == "9"
    assert canonical["canonical-remaining"].support_count == 5
    assert canonical["canonical-split"].jersey_number == "8"
    assert canonical["canonical-split"].support_count == 3
    assert not (
        pre_resolver_crop_ids
        & {
            observation.id
            for observation in canonical[
                "canonical-split"
            ].selected_observations
        }
    )
    assert mapping_diagnostics["evidenceSource"] == "raw-crop-results"
    assert mapping_diagnostics["mappedRawCropCount"] == 8
    assert mapping_diagnostics["finalSelectedCropCount"] == 8


def test_jersey_crop_submission_is_bounded_and_reports_candidate_cap(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path, samples_per_track=12)
    track = tracks[0]
    submitted = _install_worker(
        monkeypatch,
        {
            (track.local_tracklet_id, int(point["frameIndex"])): _recognized(8)
            for point in track.points
        },
    )

    _, diagnostics, _ = _run_jersey_ocr_for_tracklets(tracks, frames)

    assert diagnostics["candidateCropCount"] == 12
    assert diagnostics["selectionPartitionCount"] == 1
    assert diagnostics["selectedCropCount"] == 5
    assert diagnostics["submittedCropCount"] == 5
    assert len(submitted) == 5


@pytest.mark.parametrize(
    "worker_item",
    [
        _recognized(8, 0.95),
        {
            "usable": True,
            "status": "low-confidence",
            "number": None,
            "confidence": None,
            "candidates": [{"number": "8", "confidence": 0.40}],
            "quality": {"cropWidth": 42, "cropHeight": 100},
            "rejectionReasons": [],
            "decisionReasons": ["confidence-below-threshold"],
            "provider": "fake-jersey-ocr",
            "modelVersion": "fake-v1",
        },
    ],
    ids=["single-recognized", "low-confidence"],
)
def test_single_or_low_confidence_reading_remains_provisional(
    monkeypatch,
    tmp_path,
    worker_item,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path, samples_per_track=1)
    _install_worker(monkeypatch, {("tracklet-0001", 1): worker_item})

    summaries, diagnostics, warnings = _run_jersey_ocr_for_tracklets(tracks, frames)

    summary = summaries["tracklet-0001"]
    assert warnings == []
    assert summary.status == "provisional"
    assert summary.jersey_number is None
    assert summary.candidate_number == "8"
    _, people, _ = _publish(tracks, _scene(), summaries, diagnostics)
    assert people[0]["jerseyNumber"] is None
    assert people[0]["candidateNumber"] == "8"
    assert people[0]["rosterCandidates"] == []


def test_conflicting_jersey_readings_publish_review_conflict_without_number(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path, samples_per_track=2)
    _install_worker(
        monkeypatch,
        {
            ("tracklet-0001", 1): _recognized(8, 0.95),
            ("tracklet-0001", 11): _recognized(9, 0.95),
        },
    )

    summaries, diagnostics, _ = _run_jersey_ocr_for_tracklets(tracks, frames)
    summary = summaries["tracklet-0001"]
    assert summary.status == "conflict"
    assert summary.jersey_number is None
    assert summary.candidate_number is None

    _, people, _ = _publish(tracks, _scene(), summaries, diagnostics)
    person = people[0]
    assert person["jerseyNumber"] is None
    assert person["candidateNumber"] is None
    assert person["externalPlayerId"] is None
    assert any(item["code"] == "jersey-ocr-conflict" for item in person["conflicts"])


def test_reliable_tracklet_jerseys_reach_global_identity_resolver(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(
        tmp_path,
        track_count=2,
        samples_per_track=2,
    )
    # Make the tracklets non-overlapping so reliable jersey evidence may stitch
    # them. ReID is intentionally absent.
    for point in tracks[1].points:
        point["t"] = float(point["t"]) + 3.0
    values = {
        (track.local_tracklet_id, int(point["frameIndex"])): _recognized(8, 0.95)
        for track in tracks
        for point in track.points
    }
    _install_worker(monkeypatch, values)

    summaries, _, _ = _run_jersey_ocr_for_tracklets(tracks, frames)
    resolved, diagnostics = _resolve_canonical_track_states(
        tracks,
        {1: "home", 2: "home"},
        summaries,
    )

    assert len(resolved) == 1
    assert diagnostics["acceptedEdges"][0]["reasons"] == ("reliable-jersey-match",)
    assert diagnostics["jerseyReliableTrackletCount"] == 2


def test_ocr_outage_is_diagnostic_and_reconstruction_identity_remains_available(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path, samples_per_track=2)
    monkeypatch.setattr(
        "app.reconstruction_jersey_inference.jersey_ocr_worker_readiness",
        lambda **_: {
            "configured": True,
            "status": "ready",
            "backend": "fake-jersey-ocr",
        },
    )

    def outage(*_args, **_kwargs):
        raise JerseyOcrWorkerError("worker offline")

    monkeypatch.setattr("app.reconstruction_jersey_inference.analyze_jersey_crops", outage)

    summaries, diagnostics, warnings = _run_jersey_ocr_for_tracklets(tracks, frames)

    assert summaries == {}
    assert diagnostics["status"] == "failed"
    assert "worker offline" in diagnostics["detail"]
    assert warnings
    resolved, resolver_diagnostics = _resolve_canonical_track_states(
        tracks,
        {1: "home"},
        summaries,
    )
    assert len(resolved) == 1
    assert resolver_diagnostics["jerseyReliableTrackletCount"] == 0


def test_unavailable_worker_skips_crop_submission_and_remains_fail_open(
    monkeypatch,
    tmp_path,
) -> None:
    tracks, frames = _tracks_and_frames(tmp_path, samples_per_track=1)
    monkeypatch.setattr(
        "app.reconstruction_jersey_inference.jersey_ocr_worker_readiness",
        lambda **_: {
            "configured": True,
            "status": "unavailable",
            "backend": None,
            "detail": "connection refused",
        },
    )
    monkeypatch.setattr(
        "app.reconstruction_jersey_inference.analyze_jersey_crops",
        lambda *_args, **_kwargs: pytest.fail("client must not run while unavailable"),
    )

    summaries, diagnostics, warnings = _run_jersey_ocr_for_tracklets(tracks, frames)

    assert summaries == {}
    assert diagnostics["status"] == "unavailable"
    assert diagnostics["submittedCropCount"] == 0
    assert warnings == [
        "Jersey OCR is unavailable; reconstruction continued without shirt-number identity evidence."
    ]
