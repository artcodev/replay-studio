from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.reconstruction_person_detection_contract import Detection
from app.reconstruction_track_state import TrackState
from app.track_observation_accumulator import append_track_observation
from app.reconstruction_frame_analysis import analyze_scene_frame
from app.reconstruction_identity_annotation_draft import (
    draft_frame_person_annotation_upsert as upsert_frame_person_annotation,
)
from app.reconstruction_canonical_people_projection import (
    canonical_people_documents as _canonical_people_documents,
)
from app.reconstruction_identity_persistence import (
    assign_persistent_canonical_person_ids as _assign_persistent_canonical_person_ids,
)
from app.reconstruction_reid_evidence import (
    attach_identity_embeddings as _attach_identity_embeddings,
    capture_detection_observations as _capture_detection_observations,
    identity_embedding_requests as _identity_embedding_requests,
)
from app.reconstruction_canonical_identity_resolution import (
    resolve_canonical_track_states as _resolve_canonical_track_states,
)


def _scene(*, canonical_people=None, tracks=None) -> dict:
    return {
        "id": "identity-scene",
        "duration": 2.0,
        "payload": {
            "pitch": {"length": 105, "width": 68},
            "teams": [
                {"id": "home", "color": "#ffffff"},
                {"id": "away", "color": "#000000"},
            ],
            "canonicalPeople": canonical_people or [],
            "tracks": tracks or [],
            "ball": {"keyframes": []},
            "videoAsset": {
                "sourceStart": 3.0,
                "selectedSegmentId": "segment-1",
                "reconstruction": {
                    "status": "ready",
                    "model": "test-model",
                    "trackObservationSchemaVersion": 3,
                    "pitchCalibration": {"status": "fallback"},
                },
            },
        },
    }


def _detection(
    frame_index: int,
    time: float,
    x: float,
    embedding: np.ndarray | None = None,
) -> tuple[Detection, float]:
    detection = Detection(
        x=x,
        y=200.0,
        width=24.0,
        height=48.0,
        confidence=0.9,
        feature=np.ones(12, dtype=np.float32) * 0.1,
        reid_feature=embedding,
        pitch_x=x / 20.0,
        pitch_z=0.0,
        position_uncertainty_metres=0.5,
    )
    _capture_detection_observations([detection], frame_index)
    return detection, time


def test_identity_worker_results_stay_separate_from_hsv_and_keep_crop_qa():
    vector = np.zeros(256, dtype=np.float32)
    vector[5] = 1.0
    detection, _ = _detection(101, 0.0, 100.0)
    detection.crop_frame_sha256 = "ab" * 32
    detection.crop_sha256 = "cd" * 32
    path = Path("/tmp/frame_00101.jpg")

    requests, _local_items, _overlap = _identity_embedding_requests(
        [(path, 0.0)],
        [([detection], 0.0)],
    )
    assert requests[0][2][0]["observationId"] == detection.observation_id
    assert requests[0][2][0]["cropSha256"] == "cd" * 32

    diagnostics = _attach_identity_embeddings(
        [([detection], 0.0)],
        {
            str(detection.observation_id): {
                "usable": True,
                "embedding": vector.tolist(),
                "quality": {"sharpness": 44.0},
                "role": "player",
                "roleConfidence": 0.93,
                "provider": "prtreid-bpbreid-soccernet",
                "modelVersion": "weights-1",
            }
        },
    )

    assert detection.reid_feature is not None
    assert detection.reid_feature.shape == (256,)
    assert detection.feature.shape == (12,)
    assert detection.reid_quality == {"sharpness": 44.0}
    assert diagnostics["usableCropRatio"] == 1.0
    assert diagnostics["crops"] == [
        {
            "observationId": str(detection.observation_id),
            "frameIndex": None,
            "status": "usable",
            "usable": True,
            "quality": {"sharpness": 44.0},
            "rejectionReasons": [],
            "evidenceFingerprint": None,
            "provider": "prtreid-bpbreid-soccernet",
            "modelVersion": "weights-1",
            "role": "player",
            "roleConfidence": 0.93,
        }
    ]


def test_identity_worker_rejection_reasons_are_published_for_review():
    detection, _ = _detection(102, 0.1, 120.0)

    diagnostics = _attach_identity_embeddings(
        [([detection], 0.1)],
        {
            str(detection.observation_id): {
                "usable": False,
                "frameIndex": 102,
                "quality": {"sharpness": 2.0},
                "rejectionReasons": ["too-blurry", "too-small"],
                "evidenceFingerprint": "pixel-evidence-v1:abc",
                "provider": "prtreid-bpbreid-soccernet",
                "modelVersion": "weights-1",
            }
        },
    )

    assert detection.reid_feature is None
    assert diagnostics["rejectedObservationCount"] == 1
    assert diagnostics["crops"][0]["status"] == "rejected"
    assert diagnostics["crops"][0]["rejectionReasons"] == [
        "too-blurry",
        "too-small",
    ]


def test_offline_reid_stitch_publishes_one_canonical_person_without_3d_actor():
    embedding = np.zeros(256, dtype=np.float32)
    embedding[7] = 1.0
    first = TrackState(id=1)
    second = TrackState(id=2)
    for frame_index, time, x in ((101, 0.0, 100.0), (105, 0.5, 102.0)):
        detection, _ = _detection(frame_index, time, x, embedding)
        append_track_observation(first, detection, frame_index, time)
    for frame_index, time, x in ((110, 1.0, 108.0), (115, 1.5, 110.0)):
        detection, _ = _detection(frame_index, time, x, embedding)
        append_track_observation(second, detection, frame_index, time)

    resolved, resolver_diagnostics = _resolve_canonical_track_states(
        [first, second],
        {1: "home", 2: "home"},
    )
    assert len(resolved) == 1
    assert resolved[0].identity_status == "resolved"
    assert resolved[0].source_tracklet_ids == {"tracklet-0001", "tracklet-0002"}

    scene = _scene()
    _assign_persistent_canonical_person_ids(resolved, scene)
    people, diagnostics = _canonical_people_documents(
        resolved,
        {resolved[0].id: "home"},
        [],
        scene,
        resolver_diagnostics,
    )

    assert len(people) == 1
    assert people[0]["identityStatus"] == "resolved"
    assert people[0]["renderTrackId"] is None
    assert people[0]["memberTrackletIds"] == ["tracklet-0001", "tracklet-0002"]
    assert len(people[0]["observations"]) == 4
    assert diagnostics["canonicalPersonCount"] == 1
    assert diagnostics["reidCropCoverage"] == 1.0


def test_canonical_id_survives_rebuild_from_authoritative_bbox_overlap():
    track = TrackState(id=8)
    for frame_index, time, x in (
        (101, 0.0, 100.0),
        (102, 0.1, 102.0),
        (103, 0.2, 104.0),
        (104, 0.3, 106.0),
        (105, 0.4, 108.0),
    ):
        detection, _ = _detection(frame_index, time, x)
        append_track_observation(track, detection, frame_index, time)
    previous_observations = [
        {
            "frameIndex": point["frameIndex"],
            "sceneTime": point["t"],
            "bbox": point["bbox"],
            "confidence": point["confidence"],
        }
        for point in track.points
    ]
    scene = _scene(
        canonical_people=[
            {
                "canonicalPersonId": "canonical-kept",
                "observations": previous_observations,
            }
        ]
    )

    _assign_persistent_canonical_person_ids([track], scene)
    assert track.canonical_person_id == "canonical-kept"


def test_one_crossing_frame_cannot_transfer_previous_canonical_id():
    track = TrackState(id=9)
    detection, _ = _detection(101, 0.0, 100.0)
    append_track_observation(track, detection, 101, 0.0)
    previous = {
        "canonicalPersonId": "canonical-home-player",
        "teamId": "home",
        "role": "player",
        "observations": [
            {
                "frameIndex": 101,
                "bbox": track.points[0]["bbox"],
            }
        ],
    }
    scene = _scene(canonical_people=[previous])

    _assign_persistent_canonical_person_ids([track], scene, {9: "away"})

    assert track.canonical_person_id != "canonical-home-player"


def test_ambiguous_bidirectional_remap_keeps_previous_ids_reserved():
    tracks = [TrackState(id=10), TrackState(id=11)]
    for track in tracks:
        for frame_index, time, x in (
            (101, 0.0, 100.0),
            (102, 0.1, 102.0),
            (103, 0.2, 104.0),
            (104, 0.3, 106.0),
            (105, 0.4, 108.0),
        ):
            detection, _ = _detection(frame_index, time, x)
            append_track_observation(track, detection, frame_index, time)
    observations = [
        {"frameIndex": point["frameIndex"], "bbox": point["bbox"]}
        for point in tracks[0].points
    ]
    scene = _scene(
        canonical_people=[
            {
                "canonicalPersonId": "canonical-a",
                "teamId": "home",
                "role": "player",
                "observations": observations,
            },
            {
                "canonicalPersonId": "canonical-b",
                "teamId": "home",
                "role": "player",
                "observations": observations,
            },
        ]
    )

    _assign_persistent_canonical_person_ids(tracks, scene, {10: "home", 11: "home"})

    assert {track.canonical_person_id for track in tracks}.isdisjoint(
        {"canonical-a", "canonical-b"}
    )
    assert all(
        any(conflict["code"] == "canonical-id-remap-ambiguous" for conflict in track.identity_conflicts)
        for track in tracks
    )


def test_frame_analysis_links_canonical_person_even_without_render_track(monkeypatch):
    frame_path = Path("/tmp/frame_00101.jpg")
    observation = {
        "id": "obs-101",
        "observationId": "obs-101",
        "frameIndex": 101,
        "sceneTime": 0.0,
        "bbox": {"x": 100.0, "y": 152.0, "width": 24.0, "height": 48.0},
        "confidence": 0.9,
        "metricStatus": "unprojected",
        "metricReason": "metric-projection-unavailable",
        "canonicalPersonId": "canonical-video-only",
        "sourceTrackletId": "tracklet-0001",
    }
    scene = _scene(
        canonical_people=[
            {
                "canonicalPersonId": "canonical-video-only",
                "displayName": "Video-only person",
                "identityStatus": "provisional",
                "identityConfidence": 0.4,
                "identitySource": "tracker+trajectory",
                "teamId": None,
                "role": "player",
                "jerseyNumber": None,
                "externalPlayerId": None,
                "memberTrackletIds": ["tracklet-0001"],
                "observations": [observation],
                "evidence": [],
                "rosterCandidates": [],
                "conflicts": [],
            }
        ]
    )
    result = SimpleNamespace(orig_img=np.zeros((540, 960, 3), dtype=np.uint8))
    detection = Detection(112, 200, 24, 48, 0.9, np.zeros(12, dtype=np.float32))
    monkeypatch.setattr("app.reconstruction_frame_analysis.frame_paths", lambda _: [(frame_path, 0.0)])
    monkeypatch.setattr("app.reconstruction_frame_analysis.load_model", lambda _: object())
    monkeypatch.setattr("app.ultralytics_person_inference.predict_frame", lambda *_: result)
    monkeypatch.setattr(
        "app.ultralytics_person_inference.parse_person_detections",
        lambda _: ([detection], []),
    )

    analysis = analyze_scene_frame(scene, 0.0)
    person = analysis["people"][0]
    assert person["canonicalPersonId"] == "canonical-video-only"
    assert person["matchedTrackId"] is None
    assert person["displayName"] == "Video-only person"
    assert analysis["matchedCanonicalPeople"] == 1
    assert analysis["matchedTracks"] == 0


def test_annotation_accepts_canonical_person_separately_from_legacy_track(monkeypatch):
    frame_path = Path("/tmp/frame_00101.jpg")
    scene = _scene(
        canonical_people=[
            {
                "canonicalPersonId": "canonical-1",
                "displayName": "Person 1",
                "identityStatus": "provisional",
                "identityConfidence": 0.5,
                "identitySource": "tracker+trajectory",
                "teamId": "home",
                "role": "player",
                "externalPlayerId": None,
                "memberTrackletIds": ["tracklet-1"],
                "observations": [],
                "evidence": [],
                "rosterCandidates": [],
                "conflicts": [],
            }
        ]
    )
    monkeypatch.setattr(
        "app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame_path, 0.0)]
    )
    monkeypatch.setattr(
        "app.reconstruction_frame_annotation_target.cv2.imread",
        lambda _: np.zeros((540, 960, 3), dtype=np.uint8),
    )

    annotation = upsert_frame_person_annotation(
        scene,
        {
            "scene_time": 0.0,
            "bbox": {"x": 100, "y": 150, "width": 24, "height": 48},
            "kind": "home-player",
            "action": "confirm",
            "scope": "identity",
            "canonical_person_id": "canonical-1",
            "source_track_id": None,
        }
    )

    assert annotation["canonicalPersonId"] == "canonical-1"
    assert annotation["sourceTrackId"] is None
