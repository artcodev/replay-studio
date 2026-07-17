import asyncio
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from app.identity_review import (
    IdentityReviewError,
    build_identity_review,
    identity_observation_crop,
)
from app.identity_review_routes import router as identity_review_router


async def _async_request(application: FastAPI, method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _request(application: FastAPI, method: str, path: str, **kwargs):
    return asyncio.run(_async_request(application, method, path, **kwargs))


def scene() -> dict:
    return {
        "id": "scene-identity",
        "revision": 7,
        "payload": {
            "matchBinding": {
                "source": "thesportsdb",
                "eventId": "event-1",
                "players": [{"id": "p1"}] * 5,
                "warnings": ["The free source returned only the first five lineup entries."],
            },
            "videoAsset": {
                "id": "asset-1",
                "reconstruction": {
                    "diagnostics": {
                        "identity": {
                            "reid": {
                                "crops": [
                                    {
                                        "observationId": "obs-1",
                                        "usable": False,
                                        "status": "rejected",
                                        "rejectionReasons": ["too-small"],
                                    }
                                ]
                            },
                            "jerseyOcr": {
                                "crops": [
                                    {
                                        "observationId": "obs-2",
                                        "status": "recognized",
                                        "number": "8",
                                    }
                                ]
                            },
                        }
                    }
                },
            },
            "canonicalPeople": [
                {
                    "canonicalPersonId": "canonical-1",
                    "displayName": "Away person",
                    "identityStatus": "provisional",
                    "teamId": "away",
                    "observationCount": 2,
                    "observations": [
                        {
                            "observationId": "obs-1",
                            "frameIndex": 1,
                            "sceneTime": 0.0,
                            "confidence": 0.8,
                            "bbox": {"x": 10, "y": 10, "width": 20, "height": 40},
                        },
                        {
                            "observationId": "obs-2",
                            "frameIndex": 5,
                            "sceneTime": 0.5,
                            "confidence": 0.9,
                            "bbox": {"x": 30, "y": 8, "width": 30, "height": 60},
                        },
                    ],
                    "rosterCandidates": [
                        {"externalPlayerId": "p8", "rank": 1, "score": 0.91}
                    ],
                    "conflicts": [],
                },
                {
                    "canonicalPersonId": "canonical-2",
                    "identityStatus": "provisional",
                    "observations": [],
                    "rosterCandidates": [],
                    "conflicts": [{"code": "jersey-ocr-conflict"}],
                },
            ],
        },
    }


def test_build_identity_review_prioritizes_conflicts_and_keeps_crop_evidence():
    result = build_identity_review(
        scene(),
        worker_health={"identity": {"status": "ready"}},
    )

    assert result["summary"] == {
        "canonicalPersonCount": 2,
        "boundCount": 0,
        "suggestedCount": 1,
        "conflictCount": 1,
        "anonymousCount": 0,
        "excludedCount": 0,
    }
    assert result["matchBinding"]["roster"]["status"] == "incomplete"
    assert result["items"][0]["canonicalPersonId"] == "canonical-2"
    suggested = result["items"][1]
    assert suggested["resolutionState"] == "suggested"
    assert suggested["representativeObservations"][0]["reid"]["rejectionReasons"] == [
        "too-small"
    ]
    assert suggested["representativeObservations"][1]["jerseyOcr"]["number"] == "8"
    assert suggested["representativeObservations"][1]["cropUrl"].endswith(
        "/obs-2/crop"
    )


def test_identity_observation_crop_uses_persisted_bbox(tmp_path: Path):
    frames = tmp_path / "asset-1" / "frames"
    frames.mkdir(parents=True)
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[8:80, 8:75] = (10, 180, 240)
    assert cv2.imwrite(str(frames / "frame_00001.jpg"), image)

    content = identity_observation_crop(scene(), "obs-1", media_root=tmp_path)
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded is not None
    assert decoded.shape[0] > 40
    assert decoded.shape[1] > 20


def test_identity_observation_crop_rejects_unknown_observation(tmp_path: Path):
    with pytest.raises(IdentityReviewError, match="not found"):
        identity_observation_crop(scene(), "missing", media_root=tmp_path)


def test_identity_review_route_exposes_worker_readiness(monkeypatch):
    application = FastAPI()
    application.include_router(identity_review_router)
    monkeypatch.setattr("app.identity_review_routes.scene_store.get", lambda _: scene())
    monkeypatch.setattr(
        "app.identity_review_routes.identity_worker_readiness",
        lambda **_: {"configured": True, "status": "ready", "backend": "reid"},
    )
    monkeypatch.setattr(
        "app.identity_review_routes.jersey_ocr_worker_readiness",
        lambda **_: {"configured": True, "status": "unavailable", "backend": None},
    )

    response = _request(
        application,
        "GET",
        "/api/scenes/scene-identity/identity-review",
    )

    assert response.status_code == 200
    assert response.json()["workers"] == {
        "reid": {"configured": True, "status": "ready", "backend": "reid"},
        "jerseyOcr": {
            "configured": True,
            "status": "unavailable",
            "backend": None,
        },
    }
