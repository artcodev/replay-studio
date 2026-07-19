import asyncio
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from app.main import app
from app.artifact_store import FilesystemArtifactStore, ReconstructionArtifactError
from app.reconstruction_artifact_hydration import (
    hydrate_scene_reconstruction,
    load_dense_reconstruction_artifacts,
)
from app.reconstruction_artifact_manifest import artifact_references
from app.reconstruction_artifact_publication import (
    publish_dense_reconstruction_artifacts,
)
from app.reconstruction_ball_artifacts import publish_ball_trajectory_artifact
from app.reconstruction_identity_artifacts import (
    load_identity_diagnostics,
    publish_identity_diagnostics,
)
from app.reconstruction_series import (
    ReconstructionSeriesWindowError,
    reconstruction_series_window,
)


def _identity_diagnostics(crop_count: int = 40) -> dict:
    return {
        "sourceTrackletCount": 31,
        "canonicalPersonCount": 24,
        "resolvedPersonCount": 12,
        "provisionalPersonCount": 12,
        "associationConfidenceP10": 0.62,
        "reid": {
            "crops": [
                {
                    "observationId": f"observation-{index}",
                    "status": "usable",
                    "rejectionReasons": [],
                    "feature": [round((index + offset) / 1000, 4) for offset in range(96)],
                }
                for index in range(crop_count)
            ]
        },
        "jerseyOcr": {
            "crops": [
                {
                    "observationId": f"observation-{index}",
                    "status": "recognized",
                    "number": str(index % 30),
                }
                for index in range(crop_count)
            ]
        },
    }


def _calibration_evidence(frame_count: int = 66) -> list[dict]:
    return [
        {
            "sourceFrameIndex": index * 5,
            "sceneTime": round(index / 10, 3),
            "status": "accepted",
            "imageToPitch": [
                [1.0, 0.0, float(index)],
                [0.0, 1.0, float(index) / 2],
                [0.0, 0.0, 1.0],
            ],
            "keypoints": [
                {"id": point, "x": 100 + point, "y": 200 + point, "confidence": 0.8}
                for point in range(24)
            ],
        }
        for index in range(frame_count)
    ]


def test_content_addressed_artifact_round_trip_is_idempotent(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)
    diagnostics = _identity_diagnostics(3)

    first_manifest, first_summary = publish_identity_diagnostics(diagnostics, store=store)
    second_manifest, second_summary = publish_identity_diagnostics(diagnostics, store=store)

    assert first_manifest == second_manifest
    assert first_summary == second_summary
    assert first_summary["canonicalPersonCount"] == 24
    assert "reid" not in first_summary
    assert load_identity_diagnostics(
        {"artifactManifest": first_manifest}, store=store
    ) == diagnostics
    assert len(list(tmp_path.rglob("*.json"))) == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_artifact_publish_does_not_leave_partial_file_when_replace_fails(
    monkeypatch, tmp_path: Path
):
    store = FilesystemArtifactStore(tmp_path)

    def fail_replace(_source, _target):
        raise OSError("disk unavailable")

    monkeypatch.setattr("app.artifact_store.os.replace", fail_replace)

    with pytest.raises(ReconstructionArtifactError, match="atomically"):
        publish_identity_diagnostics(_identity_diagnostics(1), store=store)

    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.tmp"))


def test_missing_or_corrupt_artifact_fails_closed(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)
    manifest, _ = publish_identity_diagnostics(_identity_diagnostics(2), store=store)
    reference = manifest["artifacts"]["identityDiagnostics"]
    artifact_path = next(tmp_path.rglob("*.json"))
    artifact_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ReconstructionArtifactError, match="checksum"):
        load_identity_diagnostics({"artifactManifest": manifest}, store=store)

    artifact_path.unlink()
    with pytest.raises(ReconstructionArtifactError, match="missing"):
        load_identity_diagnostics({"artifactManifest": manifest}, store=store)

    malformed = json.loads(json.dumps(manifest))
    malformed["artifacts"]["identityDiagnostics"]["schemaVersion"] = 9
    with pytest.raises(ReconstructionArtifactError, match="schema version"):
        load_identity_diagnostics({"artifactManifest": malformed}, store=store)
    assert reference["uri"].startswith("artifact://sha256/")


@pytest.mark.parametrize("manifest", ("not-an-object", ["not-an-object"]))
def test_non_object_artifact_manifest_fails_closed(manifest: object):
    with pytest.raises(ReconstructionArtifactError, match="manifest is malformed"):
        artifact_references({"artifactManifest": manifest})


def test_absent_or_null_artifact_manifest_means_no_artifacts():
    assert artifact_references({}) == {}
    assert artifact_references({"artifactManifest": None}) == {}


def test_target_scene_contract_removes_dense_exact_duplicates(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)
    identity = _identity_diagnostics(160)
    calibration = _calibration_evidence()
    manifest, summary = publish_identity_diagnostics(identity, store=store)
    old_reconstruction = {
        "calibration": {"schemaVersion": 2, "frameEvidence": calibration},
        "calibrationFrames": calibration,
        "diagnostics": {
            "identity": identity,
            "identityResolver": identity,
        },
    }
    target_reconstruction = {
        "artifactManifest": manifest,
        "calibration": {"schemaVersion": 2, "frameEvidence": calibration},
        "diagnostics": {"identity": summary},
    }

    old_size = len(json.dumps(old_reconstruction, separators=(",", ":")).encode())
    target_size = len(
        json.dumps(target_reconstruction, separators=(",", ":")).encode()
    )

    assert "calibrationFrames" not in target_reconstruction
    assert "identityResolver" not in target_reconstruction["diagnostics"]
    assert "reid" not in target_reconstruction["diagnostics"]["identity"]
    assert target_size < old_size * 0.35


def _dense_scene(frame_count: int = 180) -> dict:
    keyframes = [
        {
            "id": f"point-{index}",
            "t": round(index / 10, 3),
            "x": round(index / 20, 3),
            "z": round(index / 30, 3),
            "confidence": 0.9,
        }
        for index in range(frame_count)
    ]
    observations = [
        {
            "observationId": f"observation-{index}",
            "frameIndex": index,
            "sourceFrameIndex": index,
            "sceneTime": round(index / 10, 3),
            "bbox": {"x": index, "y": 2, "width": 20, "height": 40},
            "confidence": 0.8,
        }
        for index in range(frame_count)
    ]
    frame_evidence = [
        {
            "sourceFrameIndex": index,
            "sampleIndex": index,
            "sceneTime": round(index / 10, 3),
            "sourceTime": round(index / 10, 3),
            "status": "accepted",
            "source": "automatic",
            "projectionSource": "direct",
            "imageToPitch": [[1, 0, index], [0, 1, index], [0, 0, 1]],
        }
        for index in range(frame_count)
    ]
    return {
        "id": "dense-scene",
        "title": "Dense scene",
        "version": 1,
        "revision": 1,
        "duration": 18.0,
        "payload": {
            "videoAsset": {
                "id": "video-1",
                "reconstruction": {
                    "status": "ready",
                    "calibration": {
                        "schemaVersion": 2,
                        "summary": {},
                        "frameEvidence": frame_evidence,
                    },
                    "ballDetection": {
                        "schemaVersion": 1,
                        "frames": [
                            {"frameIndex": index, "t": round(index / 10, 3)}
                            for index in range(frame_count)
                        ],
                    },
                },
            },
            "tracks": [
                {
                    "id": "track-1",
                    "label": "Player",
                    "keyframes": keyframes,
                    "observations": observations,
                }
            ],
            "canonicalPeople": [
                {
                    "id": "person-1",
                    "canonicalPersonId": "person-1",
                    "displayName": "Player",
                    "observations": observations,
                }
            ],
            "ball": {
                "mode": "automatic",
                "keyframes": keyframes,
                "automaticKeyframes": keyframes,
                "manualKeyframes": [],
                "diagnostics": {"status": "ready"},
            },
        },
    }


def test_dense_series_are_artifact_only_and_hydrate_from_checksums(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)
    scene = _dense_scene()
    original = deepcopy(scene)
    original_size = len(json.dumps(original, separators=(",", ":")).encode())

    manifest = publish_dense_reconstruction_artifacts(scene, store=store)
    compact = deepcopy(scene)
    compact_size = len(json.dumps(scene, separators=(",", ":")).encode())

    assert set(manifest["artifacts"]) == {
        "identityTimeline",
        "ballTrajectory",
        "calibrationFrames",
    }
    assert "keyframes" not in scene["payload"]["tracks"][0]
    assert "observations" not in scene["payload"]["canonicalPeople"][0]
    assert "keyframes" not in scene["payload"]["ball"]
    assert "frameEvidence" not in scene["payload"]["videoAsset"]["reconstruction"]["calibration"]
    assert compact_size < original_size * 0.08

    # A compact database document may cross the publication boundary again
    # during a guarded state update. It must retain its immutable references,
    # rather than replacing the artifact contents with empty arrays.
    assert publish_dense_reconstruction_artifacts(scene, store=store) == manifest
    assert scene == compact

    loaded = load_dense_reconstruction_artifacts(
        scene["payload"]["videoAsset"]["reconstruction"], store=store
    )
    assert len(loaded["identityTimeline"]["tracks"][0]["keyframes"]) == 180
    hydrate_scene_reconstruction(scene, store=store)
    assert scene["payload"]["tracks"][0]["keyframes"] == original["payload"]["tracks"][0]["keyframes"]
    assert scene["payload"]["canonicalPeople"][0]["observations"] == original["payload"]["canonicalPeople"][0]["observations"]
    assert scene["payload"]["ball"]["keyframes"] == original["payload"]["ball"]["keyframes"]
    assert "keyframeCount" not in scene["payload"]["tracks"][0]
    assert "observationCount" not in scene["payload"]["tracks"][0]
    assert "observationCount" not in scene["payload"]["canonicalPeople"][0]
    assert "keyframeCount" not in scene["payload"]["ball"]
    assert (
        "frameEvidenceCount"
        not in scene["payload"]["videoAsset"]["reconstruction"]["calibration"]
    )

    # Hydration is a mutable view; publication recomputes compact summaries
    # and returns the exact same content identities when nothing was changed.
    assert publish_dense_reconstruction_artifacts(scene, store=store) == manifest
    assert scene == compact
    assert len(list(tmp_path.rglob("*.json"))) == 3


def test_hydration_fails_closed_when_compact_count_disagrees_with_artifact(
    tmp_path: Path,
):
    store = FilesystemArtifactStore(tmp_path)
    scene = _dense_scene(3)
    publish_dense_reconstruction_artifacts(scene, store=store)
    scene["payload"]["tracks"][0]["keyframeCount"] = 99

    with pytest.raises(ReconstructionArtifactError, match="keyframeCount"):
        hydrate_scene_reconstruction(scene, store=store)


def test_materialized_empty_identity_series_replaces_previous_artifact(
    tmp_path: Path,
):
    store = FilesystemArtifactStore(tmp_path)
    scene = _dense_scene(3)
    first = deepcopy(publish_dense_reconstruction_artifacts(scene, store=store))

    hydrate_scene_reconstruction(scene, names=("identityTimeline",), store=store)
    scene["payload"]["tracks"] = []
    scene["payload"]["canonicalPeople"] = []
    second = publish_dense_reconstruction_artifacts(scene, store=store)

    assert second["artifacts"]["identityTimeline"] != first["artifacts"]["identityTimeline"]
    assert second["artifacts"]["ballTrajectory"] == first["artifacts"]["ballTrajectory"]
    loaded = load_dense_reconstruction_artifacts(
        scene["payload"]["videoAsset"]["reconstruction"],
        names=("identityTimeline",),
        store=store,
    )
    assert loaded["identityTimeline"]["tracks"] == []
    assert loaded["identityTimeline"]["canonicalPeople"] == []


def test_bulk_publication_does_not_partially_compact_scene_on_store_failure(
    tmp_path: Path,
):
    delegate = FilesystemArtifactStore(tmp_path)

    class FailingBallStore:
        def put_json(self, *, kind, schema_version, payload):
            if kind == "reconstruction.ball-trajectory":
                raise ReconstructionArtifactError("ball store unavailable")
            return delegate.put_json(
                kind=kind,
                schema_version=schema_version,
                payload=payload,
            )

        def get_json(self, reference, *, expected_kind, expected_schema_version):
            return delegate.get_json(
                reference,
                expected_kind=expected_kind,
                expected_schema_version=expected_schema_version,
            )

    scene = _dense_scene(3)
    original = deepcopy(scene)

    with pytest.raises(ReconstructionArtifactError, match="ball store unavailable"):
        publish_dense_reconstruction_artifacts(scene, store=FailingBallStore())

    assert scene == original


def test_ball_only_publication_preserves_other_materialization_markers(
    tmp_path: Path,
):
    store = FilesystemArtifactStore(tmp_path)
    scene = _dense_scene(3)
    original_manifest = deepcopy(
        publish_dense_reconstruction_artifacts(scene, store=store)
    )
    hydrate_scene_reconstruction(
        scene,
        names=("identityTimeline", "ballTrajectory"),
        store=store,
    )
    scene["payload"]["ball"]["keyframes"][0]["x"] = 42.0

    publish_ball_trajectory_artifact(scene, store=store)

    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["_materializedArtifactNames"] == ["identityTimeline"]
    assert (
        reconstruction["artifactManifest"]["artifacts"]["identityTimeline"]
        == original_manifest["artifacts"]["identityTimeline"]
    )
    assert (
        reconstruction["artifactManifest"]["artifacts"]["calibrationFrames"]
        == original_manifest["artifacts"]["calibrationFrames"]
    )
    assert (
        reconstruction["artifactManifest"]["artifacts"]["ballTrajectory"]
        != original_manifest["artifacts"]["ballTrajectory"]
    )


def test_reconstruction_series_accessor_is_bounded_and_filters_all_series(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)
    scene = _dense_scene()
    publish_dense_reconstruction_artifacts(scene, store=store)

    window = reconstruction_series_window(
        scene,
        start=2.0,
        end=3.0,
        frame_start=20,
        frame_end=30,
        store=store,
    )
    assert [item["t"] for item in window["tracks"][0]["keyframes"]] == [
        round(index / 10, 3) for index in range(20, 31)
    ]
    assert len(window["canonicalPeople"][0]["observations"]) == 11
    assert len(window["ball"]["keyframes"]) == 11
    assert len(window["calibration"]["frameEvidence"]) == 11
    assert len(window["ballDetection"]["frames"]) == 11

    with pytest.raises(ReconstructionSeriesWindowError, match="30 seconds"):
        reconstruction_series_window(scene, start=0.0, end=31.0, store=store)


def test_reconstruction_series_route_fails_explicitly_for_missing_artifact(
    monkeypatch,
    tmp_path: Path,
):
    store = FilesystemArtifactStore(tmp_path)
    scene = _dense_scene()
    publish_dense_reconstruction_artifacts(scene, store=store)
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _scene_id: scene)
    monkeypatch.setattr(
        "app.project_resource_access.project_resources.scene_owner",
        lambda scene_id: "project-test" if scene_id == "dense-scene" else None,
    )
    monkeypatch.setattr(
        "app.project_resource_access.project_matches.current_summary",
        lambda _project_id: None,
    )
    monkeypatch.setattr(
        "app.reconstruction_artifact_hydration.reconstruction_artifact_store",
        lambda: store,
    )

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            success = await client.get(
                "/api/projects/project-test/scenes/dense-scene/reconstruction-series?start=0&end=3"
            )
            next(tmp_path.rglob("*.json")).unlink()
            missing = await client.get(
                "/api/projects/project-test/scenes/dense-scene/reconstruction-series?start=0&end=3"
            )
            return success, missing

    success, missing = asyncio.run(request())
    assert success.status_code == 200
    assert missing.status_code == 409
    assert "unavailable or corrupt" in missing.json()["detail"]
