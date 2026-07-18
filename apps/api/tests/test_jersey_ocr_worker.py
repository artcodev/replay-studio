import json
from types import SimpleNamespace

import httpx
import pytest

from app.jersey_ocr_worker_batch_validation import validate_analysis_payload
from app.jersey_ocr_worker_client import (
    analyze_jersey_crops,
    jersey_ocr_worker_readiness,
)
from app.jersey_ocr_worker_contract import JerseyCropRequest, JerseyOcrWorkerError
from app.jersey_ocr_worker_item_validation import validate_ocr_item
from app.jersey_ocr_worker_model_contract import validate_readiness_payload


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _settings(url="http://jersey-ocr-worker:8093", batch_size=2):
    return SimpleNamespace(
        jersey_ocr_worker_url=url,
        jersey_ocr_worker_timeout=900,
        jersey_ocr_worker_batch_size=batch_size,
    )


def _base_payload(**values):
    return {
        "contractVersion": "jersey-ocr.v1",
        "capabilities": {
            "digitsOnly": True,
            "maxDigits": 2,
            "evidenceFingerprintVersion": "pixel-evidence-v1",
        },
        "backend": "fake-ocr",
        "providerVersion": "test",
        "modelVersion": "fake-v1",
        **values,
    }


def _quality():
    return {"cropWidth": 80, "cropHeight": 120, "sharpness": 42.0, "contrast": 20.0}


def _fingerprint(identifier: str = "crop") -> str:
    return f"pixel-evidence-v1:{identifier}"


def _response_identity(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "fileIndex"}


def _recognized_item(crop_id: str = "crop") -> dict:
    return {
        "cropId": crop_id,
        "usable": True,
        "status": "recognized",
        "number": "8",
        "confidence": 0.9,
        "candidates": [
            {
                "number": "8",
                "confidence": 0.9,
                "rawText": "8",
                "polygon": None,
            }
        ],
        "quality": _quality(),
        "rejectionReasons": [],
        "decisionReasons": [],
        "evidenceFingerprint": _fingerprint(crop_id),
    }


def test_contract_layers_reject_unknown_wire_fields() -> None:
    with pytest.raises(JerseyOcrWorkerError, match="unsupported fields"):
        validate_readiness_payload(
            _base_payload(status="ready", unversionedModelMetadata=True)
        )

    readiness = _base_payload(status="ready")
    readiness["capabilities"]["providerHint"] = "private"
    with pytest.raises(JerseyOcrWorkerError, match="unsupported fields"):
        validate_readiness_payload(readiness)

    item = _recognized_item()
    item["candidates"][0]["providerLogit"] = 0.9
    with pytest.raises(JerseyOcrWorkerError, match="unsupported fields"):
        validate_ocr_item(item)

    item = _recognized_item()
    item["quality"] = {**_quality(), "providerScore": 0.9}
    with pytest.raises(JerseyOcrWorkerError, match="unsupported fields"):
        validate_ocr_item(item)

    with pytest.raises(JerseyOcrWorkerError, match="unsupported fields"):
        validate_analysis_payload(
            _base_payload(
                items=[_recognized_item()],
                diagnostics={"unversionedCount": 1},
            ),
            {"crop"},
        )


def test_readiness_is_disabled_without_url(monkeypatch):
    monkeypatch.setattr("app.jersey_ocr_worker_client.get_settings", lambda: _settings(url=""))
    assert jersey_ocr_worker_readiness() == {
        "configured": False,
        "status": "disabled",
        "backend": None,
    }


def test_readiness_accepts_provider_neutral_contract(monkeypatch):
    monkeypatch.setattr("app.jersey_ocr_worker_client.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.jersey_ocr_worker_transport.httpx.get",
        lambda url, timeout: FakeResponse(
            _base_payload(
                status="ready",
                device="cpu",
                batchSize=32,
                modelLoadSeconds=3.5,
                inferenceScope="crop",
            )
        ),
    )
    assert jersey_ocr_worker_readiness(timeout=1.5) == {
        "configured": True,
        "status": "ready",
        "backend": "fake-ocr",
        "providerVersion": "test",
        "modelVersion": "fake-v1",
        "device": "cpu",
        "batchSize": 32,
        "modelLoadSeconds": 3.5,
        "contractVersion": "jersey-ocr.v1",
        "inferenceScope": "crop",
    }


def test_readiness_is_nonfatal_when_worker_is_offline(monkeypatch):
    monkeypatch.setattr("app.jersey_ocr_worker_client.get_settings", lambda: _settings())

    def offline(*_args, **_kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("app.jersey_ocr_worker_transport.httpx.get", offline)
    result = jersey_ocr_worker_readiness()
    assert result["status"] == "unavailable"
    assert "offline" in result["detail"]


def test_client_batches_crops_and_preserves_unaccepted_evidence(monkeypatch, tmp_path):
    crops = []
    for index in range(3):
        path = tmp_path / f"crop-{index}.jpg"
        path.write_bytes(b"jpeg")
        crops.append(
            JerseyCropRequest(
                crop_id=f"crop-{index}",
                path=path,
                observation_id=f"obs-{index}",
                tracklet_id="tracklet-1",
                frame_index=index,
                timestamp=index / 10,
            )
        )
    requests = []

    def fake_post(url, files, data, timeout):
        manifest = json.loads(data["manifest"])
        requests.append((url, files, manifest, timeout))
        items = []
        for item in manifest["items"]:
            recognized = item["cropId"] != "crop-1"
            candidates = [
                {"number": "12", "confidence": 0.9, "rawText": "12", "polygon": None}
            ]
            if not recognized:
                candidates.append(
                    {"number": "11", "confidence": 0.87, "rawText": "11", "polygon": None}
                )
            items.append(
                {
                    **_response_identity(item),
                    "usable": True,
                    "status": "recognized" if recognized else "ambiguous",
                    "number": "12" if recognized else None,
                    "confidence": 0.9 if recognized else None,
                    "candidates": candidates,
                    "quality": _quality(),
                    "rejectionReasons": [],
                    "decisionReasons": [] if recognized else ["competing-numbers"],
                    "evidenceFingerprint": _fingerprint(item["cropId"]),
                }
            )
        return FakeResponse(
            _base_payload(
                items=items,
                diagnostics={
                    "requestedCropCount": len(items),
                    "usableCropCount": len(items),
                    "recognizedCropCount": sum(
                        item["status"] == "recognized" for item in items
                    ),
                    "ambiguousCropCount": sum(
                        item["status"] == "ambiguous" for item in items
                    ),
                    "rejectedCropCount": 0,
                    "providerInferenceCropCount": max(0, len(items) - 1),
                    "cacheHitCount": 1,
                    "requestDeduplicatedCount": 0,
                    "cacheEnabled": True,
                },
            )
        )

    progress = []
    monkeypatch.setattr(
        "app.jersey_ocr_worker_client.get_settings", lambda: _settings(batch_size=2)
    )
    monkeypatch.setattr("app.jersey_ocr_worker_transport.httpx.post", fake_post)
    result = analyze_jersey_crops(crops, lambda *values: progress.append(values))
    assert len(requests) == 2
    assert requests[0][2]["contractVersion"] == "jersey-ocr.v1"
    assert requests[0][2]["items"][0]["trackletId"] == "tracklet-1"
    assert result.items_by_crop_id["crop-0"]["number"] == "12"
    assert result.items_by_crop_id["crop-1"]["status"] == "ambiguous"
    assert result.items_by_crop_id["crop-1"]["number"] is None
    assert result.diagnostics["requestedCropCount"] == 3
    assert result.diagnostics["cacheHitCount"] == 2
    assert result.diagnostics["providerInferenceCropCount"] == 1
    assert result.diagnostics["modelContract"] == {
        "contractVersion": "jersey-ocr.v1",
        "backend": "fake-ocr",
        "providerVersion": "test",
        "modelVersion": "fake-v1",
        "inferenceScope": "crop",
        "digitsOnly": True,
        "maxDigits": 2,
        "evidenceFingerprintVersion": "pixel-evidence-v1",
    }
    assert progress == [(2, 3, 1), (3, 3, 2)]


def test_client_rejects_model_version_change_between_http_batches(
    monkeypatch, tmp_path
):
    crops = []
    for index in range(2):
        path = tmp_path / f"crop-{index}.jpg"
        path.write_bytes(b"jpeg")
        crops.append(JerseyCropRequest(crop_id=f"crop-{index}", path=path))

    call_count = 0

    def fake_post(_url, data, **_kwargs):
        nonlocal call_count
        call_count += 1
        item = json.loads(data["manifest"])["items"][0]
        return FakeResponse(
            _base_payload(
                modelVersion=f"fake-v{call_count}",
                items=[
                    {
                        **_response_identity(item),
                        "usable": True,
                        "status": "recognized",
                        "number": "8",
                        "confidence": 0.9,
                        "candidates": [
                            {
                                "number": "8",
                                "confidence": 0.9,
                                "rawText": "8",
                                "polygon": None,
                            }
                        ],
                        "quality": _quality(),
                        "rejectionReasons": [],
                        "decisionReasons": [],
                        "evidenceFingerprint": _fingerprint(item["cropId"]),
                    }
                ],
            )
        )

    monkeypatch.setattr(
        "app.jersey_ocr_worker_client.get_settings", lambda: _settings(batch_size=1)
    )
    monkeypatch.setattr("app.jersey_ocr_worker_transport.httpx.post", fake_post)

    with pytest.raises(
        JerseyOcrWorkerError,
        match="changed model contract between batches: modelVersion",
    ):
        analyze_jersey_crops(crops)

    assert call_count == 2


def test_client_rejects_non_digit_accepted_number(monkeypatch, tmp_path):
    path = tmp_path / "crop.jpg"
    path.write_bytes(b"jpeg")
    monkeypatch.setattr("app.jersey_ocr_worker_client.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.jersey_ocr_worker_transport.httpx.post",
        lambda *_args, **_kwargs: FakeResponse(
            _base_payload(
                items=[
                    {
                        "cropId": "crop",
                        "usable": True,
                        "status": "recognized",
                        "number": "A8",
                        "confidence": 0.9,
                        "candidates": [],
                        "quality": _quality(),
                        "rejectionReasons": [],
                        "decisionReasons": [],
                        "evidenceFingerprint": _fingerprint(),
                    }
                ]
            )
        ),
    )
    with pytest.raises(JerseyOcrWorkerError, match="invalid number"):
        analyze_jersey_crops([JerseyCropRequest("crop", path)])


def test_client_rejects_unknown_worker_item_fields(monkeypatch, tmp_path):
    path = tmp_path / "crop.jpg"
    path.write_bytes(b"jpeg")
    monkeypatch.setattr("app.jersey_ocr_worker_client.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.jersey_ocr_worker_transport.httpx.post",
        lambda *_args, **_kwargs: FakeResponse(
            _base_payload(
                items=[
                    {
                        "cropId": "crop",
                        "usable": True,
                        "status": "recognized",
                        "number": "8",
                        "confidence": 0.9,
                        "candidates": [
                            {
                                "number": "8",
                                "confidence": 0.9,
                                "rawText": "8",
                                "polygon": None,
                            }
                        ],
                        "quality": _quality(),
                        "rejectionReasons": [],
                        "decisionReasons": [],
                        "evidenceFingerprint": _fingerprint(),
                        "unversionedProviderPayload": True,
                    }
                ]
            )
        ),
    )

    with pytest.raises(JerseyOcrWorkerError, match="unsupported fields"):
        analyze_jersey_crops([JerseyCropRequest("crop", path)])


def test_readiness_rejects_non_object_json_without_raising(monkeypatch):
    monkeypatch.setattr("app.jersey_ocr_worker_client.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.jersey_ocr_worker_transport.httpx.get", lambda *_args, **_kwargs: FakeResponse([])
    )

    assert jersey_ocr_worker_readiness()["status"] == "invalid-response"


def test_client_rejects_inconsistent_usable_status(monkeypatch, tmp_path):
    path = tmp_path / "crop.jpg"
    path.write_bytes(b"jpeg")
    monkeypatch.setattr("app.jersey_ocr_worker_client.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.jersey_ocr_worker_transport.httpx.post",
        lambda *_args, **_kwargs: FakeResponse(
            _base_payload(
                items=[
                    {
                        "cropId": "crop",
                        "usable": False,
                        "status": "recognized",
                        "number": "8",
                        "confidence": 0.9,
                        "candidates": [
                            {"number": "8", "confidence": 0.9, "rawText": "8", "polygon": None}
                        ],
                        "quality": _quality(),
                        "rejectionReasons": ["crop-too-small"],
                        "decisionReasons": [],
                        "evidenceFingerprint": _fingerprint(),
                    }
                ]
            )
        ),
    )

    with pytest.raises(JerseyOcrWorkerError, match="usable state"):
        analyze_jersey_crops([JerseyCropRequest("crop", path)])


def test_client_rejects_non_object_analysis_response(monkeypatch, tmp_path):
    path = tmp_path / "crop.jpg"
    path.write_bytes(b"jpeg")
    monkeypatch.setattr("app.jersey_ocr_worker_client.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.jersey_ocr_worker_transport.httpx.post", lambda *_args, **_kwargs: FakeResponse([])
    )

    with pytest.raises(JerseyOcrWorkerError, match="malformed JSON"):
        analyze_jersey_crops([JerseyCropRequest("crop", path)])
