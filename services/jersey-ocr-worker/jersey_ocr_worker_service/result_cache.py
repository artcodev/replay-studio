from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from time import monotonic
from typing import Sequence

from .provider_contract import RawTextCandidate


class OcrResultCache:
    """Bounded in-memory cache for deterministic crop-level providers."""

    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max(0, int(max_entries))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._items: OrderedDict[
            str, tuple[float, tuple[RawTextCandidate, ...]]
        ] = OrderedDict()
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_seconds > 0

    def get(self, key: str) -> tuple[RawTextCandidate, ...] | None:
        if not self.enabled:
            return None
        now = monotonic()
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            created_at, candidates = value
            if now - created_at > self.ttl_seconds:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return candidates

    def put(self, key: str, candidates: Sequence[RawTextCandidate]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._items[key] = (monotonic(), tuple(candidates))
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)
