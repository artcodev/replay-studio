from __future__ import annotations

from typing import Any

from .analysis_policy import (
    QualityPolicy,
    assess_quality,
    cache_key,
    decide_number,
    evidence_fingerprint,
)
from .provider_contract import JerseyOcrProvider, OcrResult, OcrSample, ProviderUnavailable
from .request_contract import CONTRACT_VERSION, capabilities, decode_image, parse_manifest
from .result_cache import OcrResultCache


class JerseyInferenceError(RuntimeError):
    """The OCR provider violated its contract or could not infer a batch."""


def provider_info(provider: JerseyOcrProvider) -> dict[str, Any]:
    info = provider.info()
    if not isinstance(info.get("backend"), str) or not info["backend"]:
        raise ProviderUnavailable("OCR provider did not report a backend")
    if not isinstance(info.get("modelVersion"), str) or not info["modelVersion"]:
        raise ProviderUnavailable("OCR provider did not report modelVersion")
    return info


class JerseyOcrService:
    """Apply crop QA, deterministic caching, and OCR decision policy."""

    def __init__(
        self,
        provider: JerseyOcrProvider,
        policy: QualityPolicy,
        result_cache: OcrResultCache,
    ) -> None:
        self.provider = provider
        self.policy = policy
        self.result_cache = result_cache

    def analyze(
        self,
        crop_bytes: list[bytes],
        manifest: str,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        images = [
            decode_image(
                data,
                file_index,
                max_crop_bytes=self.policy.max_crop_bytes,
                max_crop_pixels=self.policy.max_crop_pixels,
            )
            for file_index, data in enumerate(crop_bytes)
        ]
        items = parse_manifest(
            manifest,
            len(images),
            max_batch_size=self.policy.max_batch_size,
        )
        responses: list[dict[str, Any]] = []
        samples: list[OcrSample] = []
        sample_cache_keys: list[str | None] = []
        sample_response_indices: list[list[int]] = []
        pending_by_cache_key: dict[str, int] = {}
        cache_hits = 0
        request_deduplicated = 0
        use_crop_cache = info.get("inferenceScope", "crop") == "crop"
        for item in items:
            image = images[int(item["fileIndex"])]
            quality, rejection_reasons = assess_quality(image, self.policy)
            response = {
                "cropId": item["cropId"],
                "observationId": item["observationId"],
                "trackletId": item["trackletId"],
                "frameIndex": item["frameIndex"],
                "timestamp": item["timestamp"],
                "evidenceFingerprint": evidence_fingerprint(image),
                "usable": not rejection_reasons,
                "status": "rejected" if rejection_reasons else "pending",
                "number": None,
                "confidence": None,
                "candidates": [],
                "quality": quality,
                "rejectionReasons": rejection_reasons,
                "decisionReasons": [],
            }
            responses.append(response)
            if rejection_reasons:
                continue
            item_cache_key = cache_key(image, info, self.policy) if use_crop_cache else None
            cached = (
                self.result_cache.get(item_cache_key)
                if item_cache_key is not None
                else None
            )
            if cached is not None:
                status, number, confidence, candidates, reasons = decide_number(
                    cached, self.policy
                )
                responses[-1].update(
                    {
                        "status": status,
                        "number": number,
                        "confidence": confidence,
                        "candidates": candidates,
                        "decisionReasons": reasons,
                        "cacheHit": True,
                    }
                )
                cache_hits += 1
                continue
            if item_cache_key is not None and item_cache_key in pending_by_cache_key:
                sample_response_indices[pending_by_cache_key[item_cache_key]].append(
                    len(responses) - 1
                )
                request_deduplicated += 1
                continue
            samples.append(
                OcrSample(
                    crop_id=item["cropId"],
                    image_rgb=image,
                    tracklet_id=item["trackletId"],
                    observation_id=item["observationId"],
                    frame_index=item["frameIndex"],
                    timestamp=item["timestamp"],
                )
            )
            sample_cache_keys.append(item_cache_key)
            sample_response_indices.append([len(responses) - 1])
            if item_cache_key is not None:
                pending_by_cache_key[item_cache_key] = len(samples) - 1

        try:
            recognized: list[OcrResult] = self.provider.recognize(samples)
        except ProviderUnavailable as exc:
            raise JerseyInferenceError(str(exc)) from exc
        except Exception as exc:
            raise JerseyInferenceError(f"OCR provider failed: {exc}") from exc
        if len(recognized) != len(samples):
            raise JerseyInferenceError("OCR provider returned an incomplete batch")
        for response_indices, item_cache_key, sample, result in zip(
            sample_response_indices, sample_cache_keys, samples, recognized
        ):
            if result.crop_id != sample.crop_id:
                raise JerseyInferenceError("OCR provider changed crop order")
            status, number, confidence, candidates, reasons = decide_number(
                result.candidates, self.policy
            )
            if item_cache_key is not None:
                self.result_cache.put(item_cache_key, result.candidates)
            for response_index in response_indices:
                responses[response_index].update(
                    {
                        "status": status,
                        "number": number,
                        "confidence": confidence,
                        "candidates": candidates,
                        "decisionReasons": reasons,
                        "cacheHit": False,
                    }
                )
        usable_fingerprints = [
            str(item["evidenceFingerprint"])
            for item in responses
            if item["usable"] is True
        ]
        unique_usable_fingerprints = set(usable_fingerprints)
        return {
            "contractVersion": CONTRACT_VERSION,
            "capabilities": capabilities(),
            **info,
            "items": responses,
            "diagnostics": {
                "requestedCropCount": len(responses),
                "usableCropCount": sum(item["usable"] for item in responses),
                "recognizedCropCount": sum(
                    item["status"] == "recognized" for item in responses
                ),
                "ambiguousCropCount": sum(
                    item["status"] == "ambiguous" for item in responses
                ),
                "rejectedCropCount": sum(not item["usable"] for item in responses),
                "providerInferenceCropCount": len(samples),
                "cacheHitCount": cache_hits,
                "requestDeduplicatedCount": request_deduplicated,
                "uniqueEvidenceFingerprintCount": len(unique_usable_fingerprints),
                "duplicateEvidenceFingerprintCount": (
                    len(usable_fingerprints) - len(unique_usable_fingerprints)
                ),
                "cacheEnabled": self.result_cache.enabled and use_crop_cache,
            },
        }
