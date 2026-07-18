"""Content fingerprint for labelled datasets and immutable crop bytes."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

from .manifest_contract import CropLabel, IdentityPair


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(
    schema_version: str,
    dataset: Mapping[str, str],
    crops: Sequence[CropLabel],
    pairs: Sequence[IdentityPair],
) -> str:
    # Thresholds are intentionally excluded: this identifies labels and bytes,
    # while every report records the independently selected acceptance gates.
    payload = {
        "schemaVersion": schema_version,
        "dataset": dict(dataset),
        "crops": [
            {
                "id": crop.crop_id,
                "path": crop.relative_path,
                "personId": crop.person_id,
                "role": crop.role,
                "jerseyLabel": {
                    "readable": crop.jersey_readable,
                    "number": crop.jersey_number,
                },
                "byteLength": crop.path.stat().st_size,
                "sha256": _file_sha256(crop.path),
            }
            for crop in sorted(crops, key=lambda item: item.crop_id)
        ],
        "identityPairs": [
            {
                "id": pair.pair_id,
                "leftCropId": pair.left_crop_id,
                "rightCropId": pair.right_crop_id,
                "samePerson": pair.same_person,
            }
            for pair in sorted(pairs, key=lambda item: item.pair_id)
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{sha256(canonical).hexdigest()}"

