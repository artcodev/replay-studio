from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Replay Studio API"
    database_url: str = "sqlite:///./replay-studio.db"
    redis_url: str | None = None
    match_data_provider: str = "api-football"
    api_football_api_key: str | None = None
    api_football_base_url: str = "https://v3.football.api-sports.io"
    sportsdb_api_key: str = "123"
    sportsdb_base_url: str = "https://www.thesportsdb.com/api/v1/json"
    cors_origins: str = "http://localhost:5188,http://127.0.0.1:5188"
    media_root: str = "./data/media"
    max_video_bytes: int = 262_144_000
    max_video_duration: float = 60.0
    analysis_frame_rate: float = 10.0
    # Ten observations per second is the minimum useful cadence for linking
    # crossing football players. Five FPS produced 400 ms association gaps and
    # systematically fragmented short replay tracks.
    reconstruction_frame_rate: float = 10.0
    reconstruction_model: str = "yolo26m.pt"
    reconstruction_device: str = "cpu"
    # A database-backed fencing lease prevents two API processes from running
    # the same reconstruction. Heartbeats live outside the scene JSON so they
    # do not invalidate the worker's document revision. The monitor also
    # reclaims legacy ``processing`` scenes that have no lease.
    reconstruction_lease_ttl_seconds: float = 180.0
    reconstruction_lease_heartbeat_seconds: float = 15.0
    reconstruction_recovery_poll_seconds: float = 5.0
    reconstruction_recovery_max_workers: int = 2
    # The ball is a tiny, fast target and must not share the generic COCO
    # person detector's class/filter contract.  The bundled Roboflow Sports
    # checkpoint is the accuracy-first local default; WASB runs in its own
    # legacy worker and can be selected without changing the scene schema.
    ball_detection_backend: str = "dedicated-ultralytics"
    ball_detection_model: str = str(
        Path(__file__).resolve().parent.parent
        / "models"
        / "football-ball-detection.pt"
    )
    ball_detection_confidence: float = 0.05
    ball_detection_image_size: int = 640
    ball_detection_tile_size: int = 640
    ball_detection_tile_overlap: float = 0.20
    ball_detection_inference_batch_size: int = 8
    ball_detection_nms_iou: float = 0.10
    ball_detection_max_candidates: int = 12
    # Dense source-rate sampling is cached per scene range. Player detection
    # and calibration remain on their existing 10 FPS cadence.
    ball_analysis_frame_rate: float = 25.0
    ball_wasb_worker_url: str | None = "http://127.0.0.1:8092/detect"
    ball_wasb_timeout: float = 120.0
    ball_detection_failure_policy: str = "fallback"
    # Accuracy-first local default. Docker Compose overrides this with the
    # service-network hostname; set an empty value explicitly to opt into the
    # smaller local keypoint fallback.
    calibration_worker_url: str | None = "http://127.0.0.1:8090"
    calibration_worker_timeout: float = 900.0
    # A single editor preview should tolerate a cold CPU PnLCalib worker, but it
    # must not inherit the 15-minute background-job timeout.
    calibration_frame_worker_timeout: float = 60.0
    calibration_worker_batch_size: int = 2
    # PRTReID lives in its own legacy PyTorch runtime. Keep the URL optional so
    # reconstruction can report missing identity evidence without pretending a
    # generic image embedding is an equivalent fallback.
    identity_worker_url: str | None = "http://127.0.0.1:8091"
    identity_worker_timeout: float = 900.0
    identity_worker_batch_size: int = 4
    # Jersey OCR has an intentionally provider-neutral HTTP contract. MMOCR,
    # EasyOCR and a future tracklet-level PARSeq provider live outside the API
    # runtime and must fail explicitly instead of fabricating shirt numbers.
    jersey_ocr_worker_url: str | None = "http://127.0.0.1:8093"
    jersey_ocr_worker_timeout: float = 900.0
    jersey_ocr_worker_batch_size: int = 32
    pitch_keypoint_model: str = str(
        Path(__file__).resolve().parent.parent / "models" / "football-pitch-detection.pt"
    )
    # None selects the native image size stored in the Ultralytics checkpoint.
    # The bundled Roboflow model was trained at 640; forcing the 960px source
    # width causes it to drop otherwise visible penalty/goal-area landmarks.
    pitch_keypoint_image_size: int | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def origins(self) -> list[str]:
        return [value.strip() for value in self.cors_origins.split(",") if value.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
