"""Composition loader for the versioned model-validation manifest."""

from __future__ import annotations

from pathlib import Path

from .manifest_contract import MANIFEST_SCHEMA_VERSION, ManifestError, ValidationManifest
from .manifest_crop_labels import parse_crop_labels
from .manifest_dataset import parse_dataset
from .manifest_fingerprint import dataset_fingerprint
from .manifest_identity_pairs import parse_identity_pairs
from .manifest_parsing import load_json_object, reject_unknown
from .manifest_thresholds import parse_thresholds


MANIFEST_FIELDS = frozenset(
    {"$schema", "schemaVersion", "dataset", "crops", "identityPairs", "thresholds"}
)


def load_manifest(path: str | Path) -> ValidationManifest:
    source_path, raw = load_json_object(path)
    reject_unknown(raw, MANIFEST_FIELDS, "manifest")
    if raw.get("schemaVersion") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"schemaVersion must equal {MANIFEST_SCHEMA_VERSION!r}")

    dataset = parse_dataset(raw.get("dataset"))
    crops = parse_crop_labels(raw.get("crops"), source_path.parent.resolve())
    crop_by_id = {crop.crop_id: crop for crop in crops}
    identity_pairs = parse_identity_pairs(raw.get("identityPairs"), crop_by_id)
    thresholds = parse_thresholds(raw.get("thresholds"))
    return ValidationManifest(
        source_path=source_path,
        dataset=dataset,
        crops=crops,
        identity_pairs=identity_pairs,
        thresholds=thresholds,
        fingerprint=dataset_fingerprint(
            MANIFEST_SCHEMA_VERSION,
            dataset,
            crops,
            identity_pairs,
        ),
    )

