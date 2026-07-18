from fastapi import APIRouter

from .ball_worker import ball_worker_readiness
from .calibration_worker import calibration_worker_readiness
from .identity_worker_client import identity_worker_readiness
from .jersey_ocr_worker_client import jersey_ocr_worker_readiness
from .providers.registry import sports_provider


router = APIRouter(tags=["health"])


@router.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "replay-studio-api",
        "provider": sports_provider.default_provider,
        "match_data": sports_provider.descriptors(),
        "video_pipeline": "ffmpeg",
        "calibration_worker": calibration_worker_readiness(),
        "identity_worker": identity_worker_readiness(),
        "jersey_ocr_worker": jersey_ocr_worker_readiness(),
        "ball_worker": ball_worker_readiness(),
    }
