from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from .calibration_contract import FrameCalibration


@dataclass(frozen=True, slots=True)
class CacheLookup:
    hit: bool
    result: FrameCalibration | None


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    created_at: float
    result: FrameCalibration | None


class CalibrationResultCache:
    """Process-local bounded LRU for immutable model-versioned results."""

    def __init__(self, *, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max(0, max_entries)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()

    @staticmethod
    def key(model_version: str, content_sha256: str) -> str:
        return f"{model_version}:{content_sha256}"

    def get(self, key: str, now: float) -> CacheLookup:
        if self.max_entries <= 0:
            return CacheLookup(hit=False, result=None)
        entry = self._entries.get(key)
        if entry is None:
            return CacheLookup(hit=False, result=None)
        if self.ttl_seconds > 0.0 and now - entry.created_at > self.ttl_seconds:
            self._entries.pop(key, None)
            return CacheLookup(hit=False, result=None)
        self._entries.move_to_end(key)
        return CacheLookup(hit=True, result=entry.result)

    def put(
        self,
        key: str,
        result: FrameCalibration | None,
        now: float,
    ) -> None:
        if self.max_entries <= 0:
            return
        self._entries[key] = _CacheEntry(created_at=now, result=result)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
