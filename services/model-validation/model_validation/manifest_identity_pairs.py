"""Identity-pair parsing and consistency with crop ground truth."""

from __future__ import annotations

from typing import Any, Mapping

from .manifest_contract import CropLabel, IdentityPair, ManifestError
from .manifest_parsing import reject_unknown, required_string


MAX_IDENTITY_PAIR_COUNT = 100_000
IDENTITY_PAIR_FIELDS = frozenset(
    {"id", "leftCropId", "rightCropId", "samePerson"}
)


def parse_identity_pairs(
    value: Any,
    crop_by_id: Mapping[str, CropLabel],
) -> tuple[IdentityPair, ...]:
    if not isinstance(value, list) or not value:
        raise ManifestError("identityPairs must be a non-empty array")
    if len(value) > MAX_IDENTITY_PAIR_COUNT:
        raise ManifestError("identityPairs exceeds the 100000-item safety limit")

    pairs: list[IdentityPair] = []
    pair_ids: set[str] = set()
    same_count = 0
    different_count = 0
    for index, raw_pair in enumerate(value):
        label = f"identityPairs[{index}]"
        if not isinstance(raw_pair, dict):
            raise ManifestError(f"{label} must be an object")
        reject_unknown(raw_pair, IDENTITY_PAIR_FIELDS, label)
        pair_id = required_string(raw_pair.get("id"), f"{label}.id")
        if pair_id in pair_ids:
            raise ManifestError(f"Duplicate identity pair id: {pair_id}")
        pair_ids.add(pair_id)
        left_id = required_string(raw_pair.get("leftCropId"), f"{label}.leftCropId")
        right_id = required_string(raw_pair.get("rightCropId"), f"{label}.rightCropId")
        if left_id == right_id:
            raise ManifestError(f"{label} must reference two different crops")
        if left_id not in crop_by_id or right_id not in crop_by_id:
            raise ManifestError(f"{label} references an unknown crop")
        same_person = raw_pair.get("samePerson")
        if not isinstance(same_person, bool):
            raise ManifestError(f"{label}.samePerson must be boolean")
        ground_truth_same = crop_by_id[left_id].person_id == crop_by_id[right_id].person_id
        if same_person != ground_truth_same:
            raise ManifestError(
                f"{label}.samePerson conflicts with the two personId labels"
            )
        same_count += int(same_person)
        different_count += int(not same_person)
        pairs.append(IdentityPair(pair_id, left_id, right_id, same_person))

    if same_count == 0 or different_count == 0:
        raise ManifestError(
            "identityPairs must contain at least one same-person and one different-person pair"
        )
    return tuple(pairs)

