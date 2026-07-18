from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
import shutil

import pytest

from model_validation.manifest_contract import MANIFEST_SCHEMA_VERSION, ManifestError
from model_validation.manifest_crop_labels import MAX_CROP_COUNT, parse_crop_labels
from model_validation.manifest_identity_pairs import (
    MAX_IDENTITY_PAIR_COUNT,
    parse_identity_pairs,
)
from model_validation.manifest_loader import load_manifest


FIXTURES = Path(__file__).resolve().parent / "fixtures"
MANIFEST = FIXTURES / "fake-manifest.json"


def _copy_manifest(tmp_path: Path) -> Path:
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURES, copied)
    return copied / "fake-manifest.json"


def test_dataset_fingerprint_covers_labels_and_crop_bytes_but_not_thresholds(tmp_path):
    path = _copy_manifest(tmp_path)
    copied = path.parent
    original = load_manifest(path)

    assert (
        original.fingerprint
        == "sha256:a4ae56c13375b20d4da5500fee7bc57565ceb55280fa41d3e3280fa49ac41e56"
    )

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["thresholds"]["identity"]["minimumRoleAccuracy"] = 0.5
    path.write_text(json.dumps(raw), encoding="utf-8")
    threshold_change = load_manifest(path)
    assert threshold_change.fingerprint == original.fingerprint

    crop = copied / "crops" / "player-8-a.ppm"
    crop.write_bytes(crop.read_bytes() + b"\n")
    byte_change = load_manifest(path)
    assert byte_change.fingerprint != original.fingerprint


def test_manifest_rejects_pair_label_inconsistency(tmp_path):
    path = _copy_manifest(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["identityPairs"][0]["samePerson"] = False
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match="conflicts with the two personId labels"):
        load_manifest(path)


def test_json_schema_resource_is_well_formed():
    schema = json.loads((FIXTURES.parents[1] / "manifest.schema.json").read_text(encoding="utf-8"))

    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["schemaVersion"]["const"] == MANIFEST_SCHEMA_VERSION


def test_manifest_fixture_is_valid_and_exposes_crop_index():
    manifest = load_manifest(MANIFEST)

    assert manifest.crop_by_id["p8-a"].person_id == "person-8"
    assert manifest.fingerprint.startswith("sha256:")


def test_loaded_manifest_is_deeply_immutable():
    manifest = load_manifest(MANIFEST)

    with pytest.raises(TypeError):
        manifest.dataset["name"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest.thresholds["identity"]["minimumPairCoverage"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest.crop_by_id["new"] = manifest.crops[0]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        manifest.crops[0].role = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutate", "error"),
    (
        (lambda raw: raw.update({"unexpected": True}), "manifest has unsupported fields"),
        (
            lambda raw: raw["crops"][0].update({"confidence": 1.0}),
            r"crops\[0\] has unsupported fields",
        ),
        (
            lambda raw: raw["thresholds"]["identity"].update({"newGate": 1.0}),
            "thresholds.identity has unsupported fields",
        ),
    ),
)
def test_manifest_rejects_unknown_fields_at_every_contract_boundary(
    tmp_path,
    mutate,
    error,
):
    path = _copy_manifest(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match=error):
        load_manifest(path)


def test_manifest_rejects_existing_file_outside_its_directory(tmp_path):
    path = _copy_manifest(tmp_path)
    (path.parent.parent / "outside.ppm").write_bytes(b"P3\n1 1\n255\n0 0 0\n")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["crops"][0]["path"] = "../outside.ppm"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match="outside the manifest directory"):
        load_manifest(path)


def test_crop_and_identity_pair_safety_limits_fail_before_item_parsing():
    with pytest.raises(ManifestError, match="5000-item safety limit"):
        parse_crop_labels([None] * (MAX_CROP_COUNT + 1), MANIFEST.parent)
    with pytest.raises(ManifestError, match="100000-item safety limit"):
        parse_identity_pairs([None] * (MAX_IDENTITY_PAIR_COUNT + 1), {})
