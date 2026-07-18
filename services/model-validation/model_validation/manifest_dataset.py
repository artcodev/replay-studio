"""Strict parser for immutable validation-dataset provenance."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from .manifest_parsing import reject_unknown, required_object, required_string


DATASET_FIELDS = frozenset({"name", "version", "license", "source"})


def parse_dataset(value: Any) -> Mapping[str, str]:
    raw = required_object(value, "dataset")
    reject_unknown(raw, DATASET_FIELDS, "dataset")
    return MappingProxyType(
        {
            key: required_string(raw.get(key), f"dataset.{key}")
            for key in ("name", "version", "license", "source")
        }
    )

