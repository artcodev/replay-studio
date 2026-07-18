"""Crop-label parsing and cross-crop ground-truth invariants."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .manifest_contract import CropLabel, ManifestError, ROLES
from .manifest_parsing import reject_unknown, required_string, safe_relative_file


MAX_CROP_COUNT = 5_000
CROP_FIELDS = frozenset({"id", "path", "personId", "role", "jerseyLabel"})
JERSEY_FIELDS = frozenset({"readable", "number"})


def parse_crop_labels(value: Any, manifest_directory: Path) -> tuple[CropLabel, ...]:
    if not isinstance(value, list) or not value:
        raise ManifestError("crops must be a non-empty array")
    if len(value) > MAX_CROP_COUNT:
        raise ManifestError("crops exceeds the 5000-item safety limit")

    crops: list[CropLabel] = []
    crop_ids: set[str] = set()
    readable_count = 0
    unreadable_count = 0
    expected_numbers_by_person: dict[str, set[str]] = {}
    for index, raw_crop in enumerate(value):
        label = f"crops[{index}]"
        if not isinstance(raw_crop, dict):
            raise ManifestError(f"{label} must be an object")
        reject_unknown(raw_crop, CROP_FIELDS, label)
        crop_id = required_string(raw_crop.get("id"), f"{label}.id")
        if crop_id in crop_ids:
            raise ManifestError(f"Duplicate crop id: {crop_id}")
        crop_ids.add(crop_id)
        relative_path = required_string(raw_crop.get("path"), f"{label}.path")
        person_id = required_string(raw_crop.get("personId"), f"{label}.personId")
        role = required_string(raw_crop.get("role"), f"{label}.role")
        if role not in ROLES:
            raise ManifestError(f"{label}.role must be one of {', '.join(ROLES)}")

        jersey = raw_crop.get("jerseyLabel")
        if not isinstance(jersey, dict) or not isinstance(jersey.get("readable"), bool):
            raise ManifestError(f"{label}.jerseyLabel.readable must be boolean")
        reject_unknown(jersey, JERSEY_FIELDS, f"{label}.jerseyLabel")
        readable = jersey["readable"]
        number = jersey.get("number")
        if readable:
            if not isinstance(number, str) or re.fullmatch(r"[0-9]{1,2}", number) is None:
                raise ManifestError(
                    f"{label}.jerseyLabel.number must be one or two ASCII digits when readable"
                )
            readable_count += 1
            expected_numbers_by_person.setdefault(person_id, set()).add(number)
        else:
            if number is not None:
                raise ManifestError(
                    f"{label}.jerseyLabel.number must be null when unreadable"
                )
            unreadable_count += 1

        crops.append(
            CropLabel(
                crop_id=crop_id,
                path=safe_relative_file(
                    manifest_directory,
                    relative_path,
                    f"{label}.path",
                ),
                relative_path=relative_path,
                person_id=person_id,
                role=role,
                jersey_readable=readable,
                jersey_number=number,
            )
        )

    if readable_count == 0 or unreadable_count == 0:
        raise ManifestError(
            "crops must contain readable jersey labels and expected-abstention labels"
        )
    inconsistent_people = sorted(
        person_id
        for person_id, numbers in expected_numbers_by_person.items()
        if len(numbers) > 1
    )
    if inconsistent_people:
        raise ManifestError(
            "A person has conflicting readable ground-truth jersey labels: "
            + ", ".join(inconsistent_people)
        )
    return tuple(crops)

