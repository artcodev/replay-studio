from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.project_resource_access as resource_access
import app.scene_identity_routes as scene_identity_routes
from app.project_lifecycle_contract import ProjectCreate
from app.project_resource_repository import ProjectResourceRepository
from app.project_store import ProjectStore
from app.database import Base
from app.main import app
from app.reconstruction_errors import ReconstructionError, StaleReconstructionRun
from app.reconstruction_identity_annotation_draft import (
    draft_frame_person_annotation_delete as delete_frame_person_annotation,
    draft_frame_person_annotation_upsert as upsert_frame_person_annotation,
)
from app.reconstruction_identity_roster_binding_planning import (
    plan_canonical_roster_binding as _plan_canonical_roster_binding,
)
from app.reconstruction_identity_roster_clear_planning import (
    plan_clear_canonical_roster_binding as _plan_clear_canonical_roster_binding,
)
from app.reconstruction_queue import queue_reconstruction
from app.reconstruction_run_repository import ReconstructionRunRepository
from app.reconstruction_track_state import TrackState
from app.reconstruction_identity_merging import (
    merge_raw_track_states as _merge_raw_track_states,
)
from app.reconstruction_identity_splitting import (
    apply_canonical_split_corrections as _apply_canonical_split_corrections,
)
from app.reconstruction_canonical_people_projection import (
    canonical_people_documents as _canonical_people_documents,
)
from app.reconstruction_identity_persistence import (
    assign_persistent_canonical_person_ids as _assign_persistent_canonical_person_ids,
)
from app.reconstruction_canonical_identity_resolution import (
    resolve_canonical_track_states as _resolve_canonical_track_states,
)
from app.reconstruction_person_annotations import (
    apply_person_annotations as _apply_person_annotations,
    frame_annotations as _frame_annotations,
)
from app.scene_document import reconstruction_input_fingerprint
from app.scene_repository import SceneRepository


_MATCH_SNAPSHOT = {
    "homeTeam": {"id": "team-home", "name": "Home"},
    "awayTeam": {"id": "team-away", "name": "Away"},
    "roster": [
        {
            "id": "player-home-8",
            "name": "Home Eight",
            "teamId": "team-home",
            "number": "8",
        },
        {
            "id": "player-away-10",
            "name": "Away Ten",
            "teamId": "team-away",
            "number": "10",
        },
    ],
}


def set_canonical_roster_binding(*args, **kwargs):
    kwargs.setdefault("match_snapshot", _MATCH_SNAPSHOT)
    return _plan_canonical_roster_binding(*args, **kwargs)


def clear_canonical_roster_binding(*args, **kwargs):
    return _plan_clear_canonical_roster_binding(*args, **kwargs)


def _scene(*, status: str = "ready") -> dict:
    return {
        "id": "roster-binding-scene",
        "title": "Roster binding",
        "version": 1,
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "selectedSegmentId": "segment-1",
                "sourceStart": 20.0,
                "sourceEnd": 24.0,
                "analysisFps": 10.0,
                "reconstruction": {
                    "status": status,
                    "runId": "run-ready",
                    "runRevision": 3,
                    "model": "yolo26m.pt",
                    "frameAnnotations": [],
                },
            },
            "canonicalPeople": [
                {
                    "id": "canonical-offscreen",
                    "canonicalPersonId": "canonical-offscreen",
                    "displayName": "Home person 02",
                    "identityStatus": "provisional",
                    "identityConfidence": 0.72,
                    "identitySource": "tracker+trajectory",
                    "teamId": "home",
                    "role": "player",
                    "externalPlayerId": None,
                    "annotationIds": [],
                    # Intentionally no renderTrackId: this identity is currently
                    # off screen / rejected by metric projection.
                    "renderTrackId": None,
                    "observations": [
                        {
                            "id": "obs-low",
                            "observationId": "obs-low",
                            "frameIndex": 201,
                            "sceneTime": 0.0,
                            "sourceTime": 20.0,
                            "bbox": {"x": 20, "y": 30, "width": 18, "height": 42},
                            "confidence": 0.45,
                            "metricStatus": "unprojected",
                            "sourceTrackletId": "tracklet-1",
                        },
                        {
                            "id": "obs-best",
                            "observationId": "obs-best",
                            "frameIndex": 207,
                            "sceneTime": 0.6,
                            "sourceTime": 20.6,
                            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
                            "confidence": 0.94,
                            "metricStatus": "accepted",
                            "sourceTrackletId": "tracklet-1",
                        },
                    ],
                    "evidence": [],
                    "rosterCandidates": [],
                    "conflicts": [],
                }
            ],
            "tracks": [],
        },
    }


async def _async_request(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _request(method: str, path: str, **kwargs):
    def owned_scene(_project_id: str, scene_id: str):
        scene = resource_access.scenes.get(scene_id)
        if scene is None:
            raise HTTPException(status_code=404, detail="Scene not found in project")
        return scene

    with patch.object(
        resource_access,
        "project_scene_or_404",
        side_effect=owned_scene,
    ), patch.object(
        scene_identity_routes.project_matches,
        "current_snapshot",
        return_value=SimpleNamespace(payload=_MATCH_SNAPSHOT),
    ):
        return asyncio.run(_async_request(method, path, **kwargs))


@dataclass(frozen=True)
class Persistence:
    documents: SceneRepository
    runs: ReconstructionRunRepository
    sessions: sessionmaker


@pytest.fixture
def isolated_store(monkeypatch) -> Persistence:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    return Persistence(
        documents=SceneRepository(session_local),
        runs=ReconstructionRunRepository(session_local),
        sessions=session_local,
    )


def _append_second_canonical_person(scene: dict) -> None:
    second = deepcopy(scene["payload"]["canonicalPeople"][0])
    second.update(
        {
            "id": "canonical-second",
            "canonicalPersonId": "canonical-second",
            "displayName": "Home person 03",
            "annotationIds": [],
        }
    )
    for index, observation in enumerate(second["observations"]):
        observation["id"] = observation["observationId"] = f"second-{index}"
        observation["frameIndex"] += 1
        observation["sceneTime"] += 0.1
        observation["bbox"]["x"] += 100
    scene["payload"]["canonicalPeople"].append(second)


def test_split_undo_restores_and_rekeys_preexisting_unbind(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    original_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None
    )
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.6)])

    split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "split-with-preexisting-unbind",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-offscreen",
            "target_observation_id": "obs-best",
            "range_start": 0.5,
            "range_end": 1.0,
        }
    )

    stored = split["preSplitRosterCorrections"]
    assert len(stored) == 1
    assert stored[0]["id"] == original_unbind["id"]
    assert stored[0]["rosterDecisionOriginId"] == original_unbind["id"]

    # Model the completed split: the original Unbind anchor belongs to the new
    # range child, so the next explicit edit transactionally rekeys it there.
    original_person = deepcopy(scene["payload"]["canonicalPeople"][0])
    child_person = deepcopy(original_person)
    original_person["observations"] = [
        item
        for item in original_person["observations"]
        if item["observationId"] == "obs-low"
    ]
    original_person["annotationIds"] = []
    child_person.update(
        {
            "id": split["splitCanonicalPersonId"],
            "canonicalPersonId": split["splitCanonicalPersonId"],
            "observations": [
                item
                for item in child_person["observations"]
                if item["observationId"] == "obs-best"
            ],
            "annotationIds": [original_unbind["id"], split["id"]],
        }
    )
    scene["payload"]["canonicalPeople"] = [original_person, child_person]

    child_unbind = set_canonical_roster_binding(
        scene,
        split["splitCanonicalPersonId"],
        None
    )
    assert child_unbind["id"] != original_unbind["id"]
    assert child_unbind["rosterDecisionOriginId"] == original_unbind["id"]

    delete_frame_person_annotation(scene, split["id"])

    corrections = scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ]
    assert len(corrections) == 1
    restored = corrections[0]
    assert restored["id"] == original_unbind["id"]
    assert restored["canonicalPersonId"] == "canonical-offscreen"
    assert restored["rosterBindingState"] == "unbound"
    assert restored["targetObservation"]["canonicalPersonId"] == "canonical-offscreen"
    assert restored["targetObservation"]["annotationId"] == original_unbind["id"]


def test_clear_split_unbind_removes_pre_split_resurrection_path(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    original_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None
    )
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.6)])
    split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "split-before-clear",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-offscreen",
            "target_observation_id": "obs-best",
            "range_start": 0.5,
            "range_end": 1.0,
        }
    )

    original_person = deepcopy(scene["payload"]["canonicalPeople"][0])
    child_person = deepcopy(original_person)
    original_person["observations"] = [
        item
        for item in original_person["observations"]
        if item["observationId"] == "obs-low"
    ]
    original_person["annotationIds"] = []
    child_person.update(
        {
            "id": split["splitCanonicalPersonId"],
            "canonicalPersonId": split["splitCanonicalPersonId"],
            "observations": [
                item
                for item in child_person["observations"]
                if item["observationId"] == "obs-best"
            ],
            "annotationIds": [original_unbind["id"], split["id"]],
        }
    )
    scene["payload"]["canonicalPeople"] = [original_person, child_person]
    child_unbind = set_canonical_roster_binding(
        scene,
        split["splitCanonicalPersonId"],
        None
    )
    assert child_unbind["rosterDecisionOriginId"] == original_unbind["id"]

    clear_canonical_roster_binding(
        scene,
        split["splitCanonicalPersonId"]
    )
    remaining = scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ]
    assert [item["id"] for item in remaining] == [split["id"]]
    assert remaining[0]["preSplitRosterCorrections"] == []

    delete_frame_person_annotation(scene, split["id"])
    assert scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ] == []


def test_post_merge_unbind_blocks_undo_until_explicit_clear(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    _append_second_canonical_person(scene)
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.6)])

    merge = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "merge-before-new-unbind",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "merge",
            "scope": "identity",
            "canonical_person_id": "canonical-offscreen",
            "merge_target_id": "canonical-second",
        }
    )

    source, target = scene["payload"]["canonicalPeople"]
    merged = deepcopy(target)
    merged["observations"] = deepcopy(source["observations"] + target["observations"])
    merged["annotationIds"] = [merge["id"]]
    scene["payload"]["canonicalPeople"] = [merged]

    post_merge_unbind = set_canonical_roster_binding(
        scene, "canonical-second", None
    )
    assert post_merge_unbind["identityCorrectionDependencies"] == [merge["id"]]

    with pytest.raises(
        ReconstructionError,
        match="created or changed after this merge",
    ):
        delete_frame_person_annotation(scene, merge["id"])

    with pytest.raises(ReconstructionError, match="Bind / Unbind / Clear"):
        delete_frame_person_annotation(scene, post_merge_unbind["id"])

    cleared = clear_canonical_roster_binding(
        scene, "canonical-second"
    )
    assert cleared["id"] == post_merge_unbind["id"]
    delete_frame_person_annotation(scene, merge["id"])
    assert scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ] == []


def test_merge_consolidates_two_unbind_tombstones_and_delete_restores_them(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    _append_second_canonical_person(scene)
    first_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None
    )
    second_unbind = set_canonical_roster_binding(
        scene, "canonical-second", None
    )
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.6)])

    merge = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "merge-first-into-second",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "merge",
            "scope": "identity",
            "canonical_person_id": "canonical-offscreen",
            "merge_target_id": "canonical-second",
        }
    )

    annotations = scene["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"]
    assert merge["consolidatedRosterCorrectionIds"] == [first_unbind["id"]]
    assert merge["consolidatedRosterCorrections"][0]["id"] == first_unbind["id"]
    assert {item["id"] for item in annotations} == {
        "merge-first-into-second",
        second_unbind["id"],
    }

    delete_frame_person_annotation(scene, "merge-first-into-second")
    restored = scene["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"]
    assert {item["id"] for item in restored} == {
        first_unbind["id"],
        second_unbind["id"],
    }
    assert all(item["rosterBindingState"] == "unbound" for item in restored)


def test_clear_merged_unbind_removes_consolidated_lineage_before_undo(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    _append_second_canonical_person(scene)
    first_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None
    )
    second_unbind = set_canonical_roster_binding(
        scene, "canonical-second", None
    )
    assert first_unbind["rosterDecisionOriginId"] != second_unbind[
        "rosterDecisionOriginId"
    ]
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.6)])

    merge = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "merge-before-clear",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "merge",
            "scope": "identity",
            "canonical_person_id": "canonical-offscreen",
            "merge_target_id": "canonical-second",
        }
    )
    assert merge["consolidatedRosterCorrectionIds"] == [first_unbind["id"]]

    # Model the completed rebuild: the published target represents both source
    # identities and owns the one visible compatible Unbind correction.
    source, target = scene["payload"]["canonicalPeople"]
    merged = deepcopy(target)
    merged["observations"] = deepcopy(
        source["observations"] + target["observations"]
    )
    merged["annotationIds"] = [merge["id"], second_unbind["id"]]
    scene["payload"]["canonicalPeople"] = [merged]

    cleared = clear_canonical_roster_binding(
        scene, "canonical-second"
    )
    assert cleared["id"] == second_unbind["id"]
    remaining = scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ]
    assert [item["id"] for item in remaining] == [merge["id"]]
    assert "consolidatedRosterCorrectionIds" not in remaining[0]
    assert "consolidatedRosterCorrections" not in remaining[0]

    delete_frame_person_annotation(scene, merge["id"])
    assert scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ] == []


def test_clear_merged_unbind_rejects_unrelated_valid_snapshot_atomically(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    _append_second_canonical_person(scene)
    first_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None
    )
    second_unbind = set_canonical_roster_binding(
        scene, "canonical-second", None
    )
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.6)])
    merge = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "merge-with-corrupt-lineage",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "merge",
            "scope": "identity",
            "canonical_person_id": "canonical-offscreen",
            "merge_target_id": "canonical-second",
        }
    )
    unrelated = deepcopy(first_unbind)
    unrelated.update(
        {
            "id": "unrelated-hidden-unbind",
            "canonicalPersonId": "canonical-unrelated",
            "rosterDecisionOriginId": "unrelated-hidden-unbind",
        }
    )
    unrelated["targetObservation"][
        "canonicalPersonId"
    ] = "canonical-unrelated"
    merge["consolidatedRosterCorrectionIds"] = [unrelated["id"]]
    merge["consolidatedRosterCorrections"] = [unrelated]

    source, target = scene["payload"]["canonicalPeople"]
    merged = deepcopy(target)
    merged["observations"] = deepcopy(
        source["observations"] + target["observations"]
    )
    merged["annotationIds"] = [merge["id"], second_unbind["id"]]
    scene["payload"]["canonicalPeople"] = [merged]
    before = deepcopy(scene)

    with pytest.raises(ReconstructionError, match="belongs to another identity"):
        clear_canonical_roster_binding(
            scene, "canonical-second"
        )
    assert scene == before


def test_clear_unbind_preserves_unrelated_malformed_split_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    _append_second_canonical_person(scene)
    unbound = set_canonical_roster_binding(
        scene, "canonical-offscreen", None
    )
    frame = tmp_path / "frame_00208.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.7)])
    split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "unrelated-split",
            "scene_time": 0.7,
            "bbox": {"x": 220, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-second",
            "target_observation_id": "second-1",
            "range_start": 0.6,
            "range_end": 1.1,
        }
    )
    split["preSplitRosterCorrections"] = "legacy-corrupt-but-unrelated"

    cleared = clear_canonical_roster_binding(
        scene, "canonical-offscreen"
    )
    assert cleared["id"] == unbound["id"]
    remaining = scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ]
    assert [item["id"] for item in remaining] == [split["id"]]
    assert (
        remaining[0]["preSplitRosterCorrections"]
        == "legacy-corrupt-but-unrelated"
    )


def test_clear_unbind_rejects_related_malformed_split_metadata_atomically(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    set_canonical_roster_binding(
        scene, "canonical-offscreen", None
    )
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.6)])
    split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "related-corrupt-split",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-offscreen",
            "target_observation_id": "obs-best",
            "range_start": 0.5,
            "range_end": 1.0,
        }
    )
    split["preSplitRosterCorrections"][0]["correctionKind"] = "legacy-corrupt"
    before = deepcopy(scene)

    with pytest.raises(ReconstructionError, match="unsafe roster undo metadata"):
        clear_canonical_roster_binding(
            scene, "canonical-offscreen"
        )
    assert scene == before


def test_clear_nested_split_merge_lineage_prevents_ordered_undo_resurrection(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    original_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None
    )
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.6)])
    split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "nested-split",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-offscreen",
            "target_observation_id": "obs-best",
            "range_start": 0.5,
            "range_end": 1.0,
        }
    )

    source_person = deepcopy(scene["payload"]["canonicalPeople"][0])
    child_person = deepcopy(source_person)
    source_person["observations"] = [
        item
        for item in source_person["observations"]
        if item["observationId"] == "obs-low"
    ]
    source_person["annotationIds"] = []
    child_person.update(
        {
            "id": split["splitCanonicalPersonId"],
            "canonicalPersonId": split["splitCanonicalPersonId"],
            "observations": [
                item
                for item in child_person["observations"]
                if item["observationId"] == "obs-best"
            ],
            "annotationIds": [original_unbind["id"], split["id"]],
        }
    )
    scene["payload"]["canonicalPeople"] = [source_person, child_person]
    child_unbind = set_canonical_roster_binding(
        scene,
        split["splitCanonicalPersonId"],
        None
    )
    assert child_unbind["rosterDecisionOriginId"] == original_unbind["id"]

    _append_second_canonical_person(scene)
    target_person = scene["payload"]["canonicalPeople"][-1]
    target_unbind = set_canonical_roster_binding(
        scene, "canonical-second", None
    )
    merge = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "nested-merge",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "merge",
            "scope": "identity",
            "canonical_person_id": split["splitCanonicalPersonId"],
            "merge_target_id": "canonical-second",
        }
    )
    assert merge["consolidatedRosterCorrectionIds"] == [child_unbind["id"]]

    merged_target = deepcopy(target_person)
    merged_target["observations"] = deepcopy(
        child_person["observations"] + target_person["observations"]
    )
    merged_target["annotationIds"] = [
        split["id"],
        merge["id"],
        target_unbind["id"],
    ]
    scene["payload"]["canonicalPeople"] = [source_person, merged_target]

    clear_canonical_roster_binding(
        scene, "canonical-second"
    )
    remaining = {
        item["id"]: item
        for item in scene["payload"]["videoAsset"]["reconstruction"][
            "frameAnnotations"
        ]
    }
    assert set(remaining) == {split["id"], merge["id"]}
    assert remaining[split["id"]]["preSplitRosterCorrections"] == []
    assert "consolidatedRosterCorrections" not in remaining[merge["id"]]

    delete_frame_person_annotation(scene, merge["id"])
    child_after_undo = deepcopy(child_person)
    child_after_undo["annotationIds"] = [split["id"]]
    target_after_undo = deepcopy(target_person)
    target_after_undo["annotationIds"] = []
    scene["payload"]["canonicalPeople"] = [
        source_person,
        child_after_undo,
        target_after_undo,
    ]
    delete_frame_person_annotation(scene, split["id"])
    assert scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ] == []


def test_clear_nested_split_ancestry_prevents_parent_undo_resurrection(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    best = scene["payload"]["canonicalPeople"][0]["observations"][1]
    middle = deepcopy(best)
    middle.update(
        {
            "id": "obs-middle",
            "observationId": "obs-middle",
            "frameIndex": 209,
            "sceneTime": 0.8,
            "sourceTime": 20.8,
            "confidence": 0.99,
        }
    )
    middle["bbox"] = {
        "x": 150,
        "y": 80,
        "width": 24,
        "height": 58,
    }
    scene["payload"]["canonicalPeople"][0]["observations"].append(middle)
    original_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None
    )
    assert original_unbind["targetObservationId"] == "obs-middle"
    parent_frame = tmp_path / "frame_00207.jpg"
    child_frame = tmp_path / "frame_00209.jpg"
    assert cv2.imwrite(
        str(parent_frame), np.zeros((240, 360, 3), dtype=np.uint8)
    )
    assert cv2.imwrite(
        str(child_frame), np.zeros((240, 360, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        "app.reconstruction_frame_annotation_target.frame_paths",
        lambda _: [(parent_frame, 0.6), (child_frame, 0.8)],
    )
    parent_split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "parent-split",
            "scene_time": 0.6,
            "bbox": best["bbox"],
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-offscreen",
            "target_observation_id": "obs-best",
            "range_start": 0.5,
            "range_end": 1.0,
        }
    )

    original_person = deepcopy(scene["payload"]["canonicalPeople"][0])
    source_a = deepcopy(original_person)
    source_a["observations"] = [
        item
        for item in source_a["observations"]
        if item["observationId"] == "obs-low"
    ]
    source_a["annotationIds"] = []
    child_b = deepcopy(original_person)
    child_b.update(
        {
            "id": parent_split["splitCanonicalPersonId"],
            "canonicalPersonId": parent_split["splitCanonicalPersonId"],
            "observations": [
                item
                for item in child_b["observations"]
                if item["observationId"] in {"obs-best", "obs-middle"}
            ],
            "annotationIds": [original_unbind["id"], parent_split["id"]],
        }
    )
    scene["payload"]["canonicalPeople"] = [source_a, child_b]
    child_b_unbind = set_canonical_roster_binding(
        scene,
        parent_split["splitCanonicalPersonId"],
        None
    )
    child_split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "child-split",
            "scene_time": 0.8,
            "bbox": middle["bbox"],
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": parent_split["splitCanonicalPersonId"],
            "target_observation_id": "obs-middle",
            "range_start": 0.75,
            "range_end": 0.9,
        }
    )
    assert child_split["preSplitRosterCorrections"][0]["id"] == child_b_unbind["id"]

    remaining_b = deepcopy(child_b)
    remaining_b["observations"] = [
        item
        for item in remaining_b["observations"]
        if item["observationId"] == "obs-best"
    ]
    remaining_b["annotationIds"] = [parent_split["id"]]
    child_c = deepcopy(child_b)
    child_c.update(
        {
            "id": child_split["splitCanonicalPersonId"],
            "canonicalPersonId": child_split["splitCanonicalPersonId"],
            "observations": [
                item
                for item in child_c["observations"]
                if item["observationId"] == "obs-middle"
            ],
            # Deliberately omit the parent id: Clear must discover it through
            # the B -> C split's transitive source ancestry.
            "annotationIds": [child_b_unbind["id"], child_split["id"]],
        }
    )
    scene["payload"]["canonicalPeople"] = [source_a, remaining_b, child_c]
    child_c_unbind = set_canonical_roster_binding(
        scene,
        child_split["splitCanonicalPersonId"],
        None
    )
    assert child_c_unbind["rosterDecisionOriginId"] == original_unbind["id"]

    clear_canonical_roster_binding(
        scene,
        child_split["splitCanonicalPersonId"]
    )
    remaining = {
        item["id"]: item
        for item in scene["payload"]["videoAsset"]["reconstruction"][
            "frameAnnotations"
        ]
    }
    assert remaining[parent_split["id"]]["preSplitRosterCorrections"] == []
    assert remaining[child_split["id"]]["preSplitRosterCorrections"] == []

    delete_frame_person_annotation(scene, child_split["id"])
    recombined_b = deepcopy(child_b)
    recombined_b["annotationIds"] = [parent_split["id"]]
    scene["payload"]["canonicalPeople"] = [source_a, recombined_b]
    delete_frame_person_annotation(scene, parent_split["id"])
    assert scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ] == []


def test_clear_unbind_is_explicit_idempotent_and_preserves_generic_correction() -> None:
    scene = _scene()
    generic = {
        "id": "reviewed-role-only",
        "frameIndex": 207,
        "sceneTime": 0.6,
        "bbox": {"x": 120.0, "y": 80.0, "width": 24.0, "height": 58.0},
        "kind": "home-goalkeeper",
        "label": "Reviewed goalkeeper",
        "externalPlayerId": None,
        "action": "confirm",
        "scope": "identity",
        "canonicalPersonId": "canonical-offscreen",
        "updatedAt": "2026-07-17T00:00:00+00:00",
    }
    scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ].append(generic)
    bound = set_canonical_roster_binding(
        scene, "canonical-offscreen", "player-home-8"
    )

    with pytest.raises(ReconstructionError, match="Unbind the roster player"):
        clear_canonical_roster_binding(
            scene, "canonical-offscreen"
        )
    assert bound in scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ]

    unbound = set_canonical_roster_binding(
        scene, "canonical-offscreen", None
    )
    cleared = clear_canonical_roster_binding(
        scene, "canonical-offscreen"
    )
    assert cleared["id"] == unbound["id"]
    assert scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ] == [generic]

    with pytest.raises(ReconstructionError, match="no roster decision to clear"):
        clear_canonical_roster_binding(
            scene, "canonical-offscreen"
        )


def test_merge_rejects_bound_and_unbound_dedicated_decisions(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    _append_second_canonical_person(scene)
    set_canonical_roster_binding(
        scene, "canonical-offscreen", "player-home-8"
    )
    set_canonical_roster_binding(scene, "canonical-second", None)
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.6)])

    with pytest.raises(
        ReconstructionError,
        match="different dedicated Bind / Unbind decisions",
    ):
        upsert_frame_person_annotation(
            scene,
            {
                "annotation_id": "merge-first-into-second",
                "scene_time": 0.6,
                "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
                "kind": "home-player",
                "action": "merge",
                "scope": "identity",
                "canonical_person_id": "canonical-offscreen",
                "merge_target_id": "canonical-second",
            }
        )
