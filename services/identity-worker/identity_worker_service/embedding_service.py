from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .cache import IdentityCacheEntry, IdentityEmbeddingCache
from .evidence import (
    EVIDENCE_FINGERPRINT_VERSION,
    apply_cache_entry,
    cache_key,
    decode_image,
    evidence_fingerprint,
    provider_cache_entry,
)
from .provider_contract import EmbeddingSample, IdentityEmbeddingProvider, ProviderUnavailable
from .request_contract import parse_manifest


class IdentityInferenceError(RuntimeError):
    """The provider or cache could not complete a validated request."""


@dataclass(slots=True)
class _RequestGroup:
    key: str
    crop: np.ndarray
    quality: dict
    representative_observation_id: str
    response_indices: list[int]


class IdentityEmbeddingService:
    """Coordinate cache single-flight and provider inference over crops.

    Contract v2: the API already cut and QA-gated every crop, so the service
    is a pure embedder — it decodes crop bytes, deduplicates identical
    pixels, and never rejects an observation itself.
    """

    def __init__(
        self,
        provider: IdentityEmbeddingProvider,
        cache: IdentityEmbeddingCache,
    ) -> None:
        self.provider = provider
        self.cache = cache

    def process(self, crop_bytes: list[bytes], manifest: str) -> dict[str, Any]:
        request_started = perf_counter()
        provider_inference_seconds = 0.0
        provider_call_count = 0
        crop_items = parse_manifest(manifest, len(crop_bytes))
        provider_info = self.provider.info()
        responses: list[dict] = []
        groups: dict[str, _RequestGroup] = {}
        in_request_deduplicated = 0
        for item in crop_items:
            file_index = int(item["fileIndex"])
            crop = decode_image(crop_bytes[file_index], file_index)
            response = {
                "observationId": item["observationId"],
                "frameIndex": int(item.get("frameIndex") or 0),
                "usable": True,
                "quality": dict(item.get("quality") or {}),
                "rejectionReasons": [],
                "embedding": None,
                "visibilityScores": None,
                "role": None,
                "roleConfidence": None,
                "evidenceFingerprint": evidence_fingerprint(crop),
                "cacheHit": False,
                "cacheSource": "pending",
            }
            responses.append(response)
            key = cache_key(crop, provider_info)
            group = groups.get(key)
            if group is not None:
                group.response_indices.append(len(responses) - 1)
                in_request_deduplicated += 1
                continue
            groups[key] = _RequestGroup(
                key=key,
                crop=crop,
                quality=response["quality"],
                representative_observation_id=str(item["observationId"]),
                response_indices=[len(responses) - 1],
            )

        self.cache.note_in_request_deduplicated(in_request_deduplicated)
        unresolved = list(groups)
        request_hits = 0
        request_misses = 0
        concurrent_deduplicated = 0
        corrupt_misses = 0
        expired_misses = 0
        provider_inference_count = 0
        resolution_round = 0
        while unresolved:
            resolution_round += 1
            if resolution_round > 3:
                raise IdentityInferenceError(
                    "Identity cache single-flight could not resolve a concurrent miss"
                )
            reservation = self.cache.reserve_many(unresolved)
            corrupt_misses += reservation.corrupt_misses
            expired_misses += reservation.expired_misses
            next_unresolved: list[str] = []

            for key, entry in reservation.hits.items():
                request_hits += 1
                for response_index in groups[key].response_indices:
                    apply_cache_entry(
                        responses[response_index], entry, cache_source="cache-hit"
                    )

            request_misses += len(reservation.owners)
            provider_keys: list[str] = []
            samples: list[EmbeddingSample] = []
            for key in reservation.owners:
                group = groups[key]
                provider_keys.append(key)
                samples.append(
                    EmbeddingSample(
                        observation_id=group.representative_observation_id,
                        image_rgb=group.crop,
                    )
                )

            if samples:
                provider_inference_count += len(samples)
                provider_call_count += 1
                provider_started = perf_counter()
                try:
                    embedded = self.provider.embed(samples)
                    if len(embedded) != len(samples):
                        raise ProviderUnavailable(
                            "Provider returned an incomplete batch"
                        )
                    provider_entries: dict[str, IdentityCacheEntry] = {}
                    for key, sample, item in zip(provider_keys, samples, embedded):
                        if item.observation_id != sample.observation_id:
                            raise ProviderUnavailable(
                                "Provider changed observation order"
                            )
                        provider_entries[key] = provider_cache_entry(
                            item, groups[key].quality
                        )
                    self.cache.publish(provider_entries)
                except ProviderUnavailable as exc:
                    self.cache.fail(provider_keys)
                    raise IdentityInferenceError(str(exc)) from exc
                except Exception as exc:
                    self.cache.fail(provider_keys)
                    raise IdentityInferenceError(
                        f"Identity provider failed: {exc}"
                    ) from exc
                finally:
                    provider_inference_seconds += perf_counter() - provider_started
                for key, entry in provider_entries.items():
                    for response_index in groups[key].response_indices:
                        apply_cache_entry(
                            responses[response_index], entry, cache_source="provider"
                        )

            if reservation.waiters:
                concurrent_deduplicated += len(reservation.waiters)
                waited = self.cache.wait_many(reservation.waiters)
                for key, entry in waited.items():
                    for response_index in groups[key].response_indices:
                        apply_cache_entry(
                            responses[response_index],
                            entry,
                            cache_source="concurrent-deduplicated",
                        )
                next_unresolved.extend(
                    key for key in reservation.waiters if key not in waited
                )
            unresolved = next_unresolved

        usable_count = sum(item["usable"] is True for item in responses)
        rejected_count = len(responses) - usable_count
        usable_fingerprints = [
            str(item["evidenceFingerprint"])
            for item in responses
            if item["usable"] is True
        ]
        unique_usable_fingerprints = set(usable_fingerprints)
        return {
            **provider_info,
            "evidenceFingerprintVersion": EVIDENCE_FINGERPRINT_VERSION,
            "items": responses,
            "diagnostics": {
                "requestedObservationCount": len(responses),
                "usableObservationCount": usable_count,
                "rejectedObservationCount": rejected_count,
                "cacheHitCount": request_hits,
                "cacheMissCount": request_misses,
                "deduplicatedObservationCount": in_request_deduplicated,
                "concurrentDeduplicatedCount": concurrent_deduplicated,
                "providerInferenceCount": provider_inference_count,
                "providerCallCount": provider_call_count,
                "providerInferenceSeconds": round(
                    provider_inference_seconds, 6
                ),
                "requestSeconds": round(perf_counter() - request_started, 6),
                "corruptCacheMissCount": corrupt_misses,
                "expiredCacheMissCount": expired_misses,
                "uniqueEvidenceFingerprintCount": len(unique_usable_fingerprints),
                "duplicateEvidenceFingerprintCount": (
                    len(usable_fingerprints) - len(unique_usable_fingerprints)
                ),
                "cache": self.cache.stats(),
            },
        }
