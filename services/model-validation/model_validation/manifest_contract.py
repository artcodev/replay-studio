"""Immutable domain contract for a labelled model-validation dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


MANIFEST_SCHEMA_VERSION = "football-model-validation-manifest.v1"
ROLES = ("ball", "goalkeeper", "other", "player", "referee")

IDENTITY_THRESHOLD_KEYS = (
    "normalizationTolerance",
    "minimumUsableCropRatio",
    "minimumPairCoverage",
    "maximumSamePersonDistanceP95",
    "minimumDifferentPersonDistanceP05",
    "minimumMedianDistanceSeparation",
    "minimumRoleAccuracy",
)
OCR_THRESHOLD_KEYS = (
    "minimumUsableCropRatio",
    "minimumReadableExactAccuracy",
    "minimumExpectedAbstentionAccuracy",
    "maximumReadableAbstentionRate",
    "maximumSubstitutionRate",
    "maximumConflictGroupRate",
)


class ManifestError(ValueError):
    """The labelled validation manifest is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class CropLabel:
    crop_id: str
    path: Path
    relative_path: str
    person_id: str
    role: str
    jersey_readable: bool
    jersey_number: str | None


@dataclass(frozen=True, slots=True)
class IdentityPair:
    pair_id: str
    left_crop_id: str
    right_crop_id: str
    same_person: bool


@dataclass(frozen=True, slots=True)
class ValidationManifest:
    source_path: Path
    dataset: Mapping[str, str]
    crops: tuple[CropLabel, ...]
    identity_pairs: tuple[IdentityPair, ...]
    thresholds: Mapping[str, Mapping[str, float]]
    fingerprint: str

    @property
    def crop_by_id(self) -> Mapping[str, CropLabel]:
        return MappingProxyType({crop.crop_id: crop for crop in self.crops})

