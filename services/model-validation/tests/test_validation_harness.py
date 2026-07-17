from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from validation_harness import (
    IDENTITY_DIMENSION,
    MANIFEST_SCHEMA_VERSION,
    REPORT_VERSION,
    ManifestError,
    build_report,
    evaluate_identity,
    evaluate_jersey_ocr,
    load_manifest,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
MANIFEST = FIXTURES / "fake-manifest.json"


def _provider() -> dict:
    return {
        "backend": "prtreid-bpbreid-soccernet",
        "dimension": IDENTITY_DIMENSION,
        "normalized": True,
        "modelVersion": "aaaaaaaaaaaaaaaa-sn-cccccccccccc",
        "checkpointSha256": "a" * 64,
        "hrnetCheckpointSha256": "b" * 64,
        "soccerNetCommit": "c" * 40,
    }


def _vector(*values: float) -> list[float]:
    result = np.zeros(IDENTITY_DIMENSION, dtype=np.float64)
    result[: len(values)] = values
    result /= np.linalg.norm(result)
    return result.tolist()


def _identity_predictions() -> dict[str, dict]:
    return {
        "p8-a": {"usable": True, "embedding": _vector(1.0, 0.0, 0.0), "role": "player"},
        "p8-b": {"usable": True, "embedding": _vector(0.999, 0.02, 0.0), "role": "player"},
        "gk1": {"usable": True, "embedding": _vector(0.0, 1.0, 0.0), "role": "goalkeeper"},
        "ref-hidden": {"usable": True, "embedding": _vector(0.0, 0.0, 1.0), "role": "referee"},
    }


def _ocr_provider() -> dict:
    return {
        "backend": "mmocr-dbnet18-sar",
        "providerVersion": "1.0.1",
        "modelVersion": "mmocr-1.0.1/dbnet/SAR",
        "contractVersion": "jersey-ocr.v1",
        "inferenceScope": "crop",
    }


def _ocr_predictions() -> dict[str, dict]:
    return {
        "p8-a": {"usable": True, "status": "recognized", "number": "8"},
        "p8-b": {"usable": True, "status": "recognized", "number": "8"},
        "gk1": {"usable": True, "status": "recognized", "number": "1"},
        "ref-hidden": {"usable": True, "status": "no-number", "number": None},
    }


def test_fake_fixture_exercises_identity_distributions_roles_and_provenance():
    manifest = load_manifest(MANIFEST)

    result = evaluate_identity(manifest, _provider(), _identity_predictions())

    assert result["status"] == "pass"
    assert result["metrics"]["embeddingContract"]["dimension"] == 256
    assert result["metrics"]["samePersonDistance"]["count"] == 1
    assert result["metrics"]["differentPersonDistance"]["count"] == 2
    assert result["metrics"]["medianDistanceSeparation"] > 0.9
    assert result["metrics"]["roleConfusion"]["accuracy"] == 1.0
    assert all(check["passed"] for check in result["checks"])


def test_identity_contract_fails_wrong_dimension_or_checkpoint_provenance():
    manifest = load_manifest(MANIFEST)
    provider = _provider()
    provider["checkpointSha256"] = None
    predictions = _identity_predictions()
    predictions["p8-b"]["embedding"] = [1.0, 0.0]

    result = evaluate_identity(manifest, provider, predictions)

    assert result["status"] == "fail"
    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert "identity-provider-provenance" in failed
    assert "identity-embedding-contract" in failed
    assert result["metrics"]["pairCoverage"] < 1.0


def test_identity_contract_rejects_non_normalized_vector_and_fake_backend():
    manifest = load_manifest(MANIFEST)
    provider = _provider()
    provider["backend"] = "fake-provider"
    predictions = _identity_predictions()
    predictions["p8-a"]["embedding"] = [2.0, *([0.0] * (IDENTITY_DIMENSION - 1))]

    result = evaluate_identity(manifest, provider, predictions)

    assert result["status"] == "fail"
    row = next(item for item in result["samples"]["crops"] if item["cropId"] == "p8-a")
    assert "embedding-not-normalized" in row["reasons"]
    assert not next(
        check
        for check in result["checks"]
        if check["id"] == "identity-provider-provenance"
    )["passed"]


def test_fake_fixture_exercises_ocr_exact_abstention_and_conflict_metrics():
    manifest = load_manifest(MANIFEST)

    result = evaluate_jersey_ocr(manifest, _ocr_provider(), _ocr_predictions())

    assert result["status"] == "pass"
    assert result["metrics"]["readableExactAccuracy"] == 1.0
    assert result["metrics"]["expectedAbstentionAccuracy"] == 1.0
    assert result["metrics"]["conflictGroupRate"] == 0.0
    assert result["metrics"]["abstentionPrecision"] == 1.0


def test_ocr_substitution_and_same_person_number_conflict_fail_closed():
    manifest = load_manifest(MANIFEST)
    predictions = _ocr_predictions()
    predictions["p8-b"] = {"usable": True, "status": "recognized", "number": "9"}

    result = evaluate_jersey_ocr(manifest, _ocr_provider(), predictions)

    assert result["status"] == "fail"
    assert result["metrics"]["substitutionCount"] == 1
    assert result["metrics"]["conflictGroupCount"] == 1
    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert "jersey-readable-exact-accuracy" in failed
    assert "jersey-substitution-rate" in failed
    assert "jersey-conflict-group-rate" in failed


def test_ocr_provenance_rejects_an_injected_fake_backend():
    manifest = load_manifest(MANIFEST)
    provider = _ocr_provider()
    provider["backend"] = "fake-ocr"

    result = evaluate_jersey_ocr(manifest, provider, _ocr_predictions())

    assert result["status"] == "fail"
    provenance = next(
        check for check in result["checks"] if check["id"] == "jersey-provider-provenance"
    )
    assert provenance["passed"] is False


def test_dataset_fingerprint_covers_labels_and_crop_bytes_but_not_thresholds(tmp_path):
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURES, copied)
    original = load_manifest(copied / "fake-manifest.json")

    raw = json.loads((copied / "fake-manifest.json").read_text(encoding="utf-8"))
    raw["thresholds"]["identity"]["minimumRoleAccuracy"] = 0.5
    (copied / "fake-manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    threshold_change = load_manifest(copied / "fake-manifest.json")
    assert threshold_change.fingerprint == original.fingerprint

    crop = copied / "crops" / "player-8-a.ppm"
    crop.write_bytes(crop.read_bytes() + b"\n")
    byte_change = load_manifest(copied / "fake-manifest.json")
    assert byte_change.fingerprint != original.fingerprint


def test_manifest_rejects_pair_label_inconsistency(tmp_path):
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURES, copied)
    path = copied / "fake-manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["identityPairs"][0]["samePerson"] = False
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match="conflicts with the two personId labels"):
        load_manifest(path)


def test_combined_report_is_versioned_and_binds_manifest_fingerprint():
    manifest = load_manifest(MANIFEST)
    identity = evaluate_identity(manifest, _provider(), _identity_predictions())
    ocr = evaluate_jersey_ocr(manifest, _ocr_provider(), _ocr_predictions())

    report = build_report(manifest, identity=identity, jersey_ocr=ocr)

    assert report["reportVersion"] == REPORT_VERSION
    assert report["manifestSchemaVersion"] == MANIFEST_SCHEMA_VERSION
    assert report["dataset"]["fingerprint"].startswith("sha256:")
    assert report["status"] == "pass"
    assert report["thresholds"] == manifest.thresholds


def test_json_schema_resource_is_well_formed():
    schema = json.loads((FIXTURES.parents[1] / "manifest.schema.json").read_text(encoding="utf-8"))

    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["schemaVersion"]["const"] == MANIFEST_SCHEMA_VERSION
