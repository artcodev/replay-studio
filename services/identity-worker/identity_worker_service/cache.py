from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from math import isfinite, sqrt
from threading import Event, Lock
from time import monotonic
from typing import Any, Iterable, Mapping


CACHE_SCHEMA_VERSION = "identity-embedding-cache.v1"


@dataclass(frozen=True, slots=True)
class IdentityCacheEntry:
    usable: bool
    quality: dict[str, Any]
    rejection_reasons: tuple[str, ...]
    embedding: tuple[float, ...] | None = None
    visibility_scores: tuple[float, ...] | None = None
    role: str | None = None
    role_confidence: float | None = None


@dataclass(slots=True)
class _StoredEntry:
    created_at: float
    value: object


@dataclass(slots=True)
class _Flight:
    event: Event = field(default_factory=Event)
    result: object | None = None
    failed: bool = False


@dataclass(frozen=True, slots=True)
class CacheReservation:
    hits: Mapping[str, IdentityCacheEntry]
    owners: tuple[str, ...]
    waiters: Mapping[str, _Flight]
    corrupt_misses: int
    expired_misses: int


def validate_cache_entry(value: object, *, dimension: int) -> IdentityCacheEntry:
    """Return a safe immutable entry or raise for stale/corrupt cache data."""

    if not isinstance(value, IdentityCacheEntry):
        raise ValueError("cache entry type is invalid")
    if not isinstance(value.usable, bool):
        raise ValueError("cache usable flag is invalid")
    if not isinstance(value.quality, dict):
        raise ValueError("cache quality is invalid")
    for key in ("cropWidth", "cropHeight"):
        item = value.quality.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or item < 0:
            raise ValueError(f"cache quality.{key} is invalid")
    for key in ("sourceBoxWidth", "sourceBoxHeight", "sharpness"):
        item = value.quality.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not isfinite(float(item)):
            raise ValueError(f"cache quality.{key} is invalid")
    if not all(isinstance(item, str) and item for item in value.rejection_reasons):
        raise ValueError("cache rejection reasons are invalid")
    if value.usable:
        if value.rejection_reasons:
            raise ValueError("usable cache entry contains rejection reasons")
        if not isinstance(value.embedding, tuple) or len(value.embedding) != dimension:
            raise ValueError("cache embedding dimension is invalid")
        vector = [float(item) for item in value.embedding]
        if not all(isfinite(item) for item in vector):
            raise ValueError("cache embedding is non-finite")
        norm = sqrt(sum(item * item for item in vector))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError("cache embedding is not normalized")
    else:
        if not value.rejection_reasons:
            raise ValueError("rejected cache entry has no reason")
        if value.embedding is not None:
            raise ValueError("rejected cache entry contains an embedding")
    if value.visibility_scores is not None:
        if not isinstance(value.visibility_scores, tuple) or not all(
            isfinite(float(item)) for item in value.visibility_scores
        ):
            raise ValueError("cache visibility scores are invalid")
    if value.role is not None and (not isinstance(value.role, str) or not value.role):
        raise ValueError("cache role is invalid")
    if value.role_confidence is not None:
        confidence = float(value.role_confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("cache role confidence is invalid")
    return value


class IdentityEmbeddingCache:
    """Bounded TTL-LRU plus single-flight coordination for embedding crops.

    Persistent entries are process-local by design. `_Flight` objects also
    carry a completed result directly, so concurrent request deduplication
    remains correct when persistent caching is intentionally disabled.
    """

    def __init__(
        self,
        *,
        dimension: int,
        max_entries: int = 4096,
        ttl_seconds: float = 86_400,
        wait_timeout_seconds: float = 900,
        configuration_error: str | None = None,
    ) -> None:
        self.dimension = int(dimension)
        self.max_entries = max(0, int(max_entries))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.wait_timeout_seconds = max(0.1, float(wait_timeout_seconds))
        self.configuration_error = configuration_error
        self._entries: OrderedDict[str, _StoredEntry] = OrderedDict()
        self._inflight: dict[str, _Flight] = {}
        self._lock = Lock()
        self._totals = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "expirations": 0,
            "corruptMisses": 0,
            "inRequestDeduplicated": 0,
            "concurrentDeduplicated": 0,
            "waitTimeouts": 0,
            "providerFailures": 0,
        }

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_seconds > 0

    @classmethod
    def from_environment(cls, *, dimension: int, environment: Mapping[str, str]) -> "IdentityEmbeddingCache":
        try:
            return cls(
                dimension=dimension,
                max_entries=int(environment.get("REID_CACHE_MAX_ENTRIES", "4096")),
                ttl_seconds=float(environment.get("REID_CACHE_TTL_SECONDS", "86400")),
                wait_timeout_seconds=float(
                    environment.get("REID_CACHE_WAIT_TIMEOUT_SECONDS", "900")
                ),
            )
        except (TypeError, ValueError) as exc:
            return cls(
                dimension=dimension,
                max_entries=0,
                ttl_seconds=0,
                configuration_error=f"Invalid ReID cache configuration: {exc}",
            )

    def note_in_request_deduplicated(self, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            self._totals["inRequestDeduplicated"] += int(count)

    def _lookup_locked(
        self,
        key: str,
        now: float,
    ) -> tuple[IdentityCacheEntry | None, bool, bool]:
        stored = self._entries.get(key)
        if stored is None:
            return None, False, False
        if now - stored.created_at > self.ttl_seconds:
            self._entries.pop(key, None)
            self._totals["expirations"] += 1
            return None, False, True
        try:
            value = validate_cache_entry(stored.value, dimension=self.dimension)
        except (TypeError, ValueError, OverflowError):
            self._entries.pop(key, None)
            self._totals["corruptMisses"] += 1
            return None, True, False
        self._entries.move_to_end(key)
        return value, False, False

    def reserve_many(self, keys: Iterable[str]) -> CacheReservation:
        hits: dict[str, IdentityCacheEntry] = {}
        owners: list[str] = []
        waiters: dict[str, _Flight] = {}
        corrupt = expired = 0
        now = monotonic()
        with self._lock:
            for key in keys:
                value, was_corrupt, was_expired = self._lookup_locked(key, now)
                corrupt += int(was_corrupt)
                expired += int(was_expired)
                if value is not None:
                    hits[key] = value
                    self._totals["hits"] += 1
                    continue
                self._totals["misses"] += 1
                flight = self._inflight.get(key)
                if flight is not None:
                    waiters[key] = flight
                    self._totals["concurrentDeduplicated"] += 1
                    continue
                self._inflight[key] = _Flight()
                owners.append(key)
        return CacheReservation(
            hits=hits,
            owners=tuple(owners),
            waiters=waiters,
            corrupt_misses=corrupt,
            expired_misses=expired,
        )

    def publish(self, values: Mapping[str, IdentityCacheEntry]) -> None:
        now = monotonic()
        with self._lock:
            for key, raw_value in values.items():
                value = validate_cache_entry(raw_value, dimension=self.dimension)
                flight = self._inflight.pop(key, None)
                if self.enabled:
                    self._entries[key] = _StoredEntry(now, value)
                    self._entries.move_to_end(key)
                    self._totals["stores"] += 1
                    while len(self._entries) > self.max_entries:
                        self._entries.popitem(last=False)
                        self._totals["evictions"] += 1
                if flight is not None:
                    flight.result = value
                    flight.event.set()

    def fail(self, keys: Iterable[str]) -> None:
        with self._lock:
            for key in keys:
                flight = self._inflight.pop(key, None)
                if flight is None:
                    continue
                flight.failed = True
                flight.event.set()
                self._totals["providerFailures"] += 1

    def wait_many(self, waiters: Mapping[str, _Flight]) -> dict[str, IdentityCacheEntry]:
        results: dict[str, IdentityCacheEntry] = {}
        for key, flight in waiters.items():
            if not flight.event.wait(self.wait_timeout_seconds):
                with self._lock:
                    self._totals["waitTimeouts"] += 1
                continue
            if flight.failed:
                continue
            try:
                value = validate_cache_entry(flight.result, dimension=self.dimension)
            except (TypeError, ValueError, OverflowError):
                with self._lock:
                    self._totals["corruptMisses"] += 1
                continue
            results[key] = value
        return results

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schemaVersion": CACHE_SCHEMA_VERSION,
                "enabled": self.enabled,
                "maxEntries": self.max_entries,
                "ttlSeconds": self.ttl_seconds,
                "waitTimeoutSeconds": self.wait_timeout_seconds,
                "size": len(self._entries),
                "inFlight": len(self._inflight),
                "configurationError": self.configuration_error,
                **self._totals,
            }
