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
    # Player/calibration frames are always materialized at the source video's
    # nominal cadence. A reduced cadence is selected explicitly per scene and
    # fingerprinted into the calibration/reconstruction job; there is no
    # process-global hidden FPS cap.
    reconstruction_model: str = "yolo26m.pt"
    reconstruction_device: str = "cpu"
    # When configured, sampled person detection is delegated to one strict
    # binary-media worker. A configured but unavailable worker fails closed;
    # there is no silent switch back to the in-process CPU detector.
    person_detection_worker_url: str | None = None
    person_detection_worker_timeout: float = 120.0
    # A database-backed fencing lease prevents two runner processes from owning
    # the same reconstruction. Heartbeats live outside the scene JSON so they
    # do not invalidate the worker's document revision. Recovery discovers
    # only compact current-job rows and reclaims processing jobs after expiry.
    reconstruction_lease_ttl_seconds: float = 180.0
    reconstruction_lease_heartbeat_seconds: float = 15.0
    reconstruction_recovery_poll_seconds: float = 5.0
    reconstruction_recovery_max_workers: int = 1
    # Video ingest and multi-pass orchestration run in a separate durable
    # process pool.  Their compact queue/leases are independent from both
    # dense Scene documents and AnalysisRun telemetry.
    pipeline_lease_ttl_seconds: float = 120.0
    pipeline_lease_heartbeat_seconds: float = 10.0
    pipeline_recovery_poll_seconds: float = 2.0
    pipeline_recovery_max_workers: int = 2
    pipeline_dependency_poll_seconds: float = 2.0
    # The ball is a tiny, fast target and must not share the generic COCO
    # person detector's class/filter contract.  The bundled Roboflow Sports
    # checkpoint is the accuracy-first local default; WASB runs in its own
    # isolated pinned-runtime worker behind the same detector contract.
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
    # Persist a clean detector prefix periodically so cancellation/restart does
    # not discard many minutes of tiled CPU inference.
    ball_detection_checkpoint_interval: int = 4
    # Dedicated Ultralytics performs a full tiled reacquisition periodically
    # and follows recent candidates in substantially cheaper full-resolution
    # crops between those scans. A miss is retried with a global scan on the
    # same frame, so this optimization keeps every source timestamp in the
    # evidence stream and prevents a clean miss from making a lost seed sticky.
    ball_detection_full_scan_interval: int = 5
    ball_detection_roi_region_count: int = 3
    ball_detection_roi_padding: int = 320
    ball_detection_nms_iou: float = 0.10
    ball_detection_max_candidates: int = 12
    # Dense ball decoding has its own explicit detector contract. Player and
    # calibration sampling is selected per scene, independently of this cap.
    ball_analysis_frame_rate: float = 25.0
    ball_wasb_worker_url: str | None = "http://127.0.0.1:8092/v1/detections"
    ball_wasb_timeout: float = 120.0
    # "per-frame-window" (default) keeps the symmetric centered window per
    # dense frame. "batched-sequence" opts into one multipart request per run
    # of frames — ~3x fewer uploads/inferences at the cost of the worker's
    # fixed window tiling at run boundaries. The value enters the queued
    # detector input and therefore the cache contract and input fingerprint.
    ball_wasb_transport: str = "per-frame-window"
    # Frames per batched-sequence request; must stay at or below the worker's
    # WASB_MAX_BATCH_FRAMES (96 by default) and is a multiple of the model's
    # 3-frame window so tiling never pads mid-run.
    ball_wasb_batch_size: int = 9
    ball_detection_failure_policy: str = "fallback"
    # After a primary-detector failure the circuit serves the fallback for
    # this many dense frames, then half-opens and retries the primary once.
    # A transient worker outage therefore cannot degrade the rest of a clip.
    ball_detection_circuit_retry_interval: int = 25
    # Dense phases emit one progress tick per frame; each durable write is a
    # full lease-fenced transaction. Quiet same-phase ticks are coalesced to
    # this interval, while phase transitions and terminal ticks write always.
    reconstruction_progress_write_interval_seconds: float = 1.0
    # Every run appends a JSONL journal (one event per pipeline step and
    # phase summary) so a finished analysis can be inspected independently.
    # Journal ticks are local file appends and are never throttled.
    analysis_run_log_enabled: bool = True
    analysis_run_log_directory: str = "./logs/analysis-runs"
    # A deterministically crashing child (for example OOM under CPU
    # inference) is bounded: after this many claims of the same run the job
    # is terminally invalidated with its last error instead of looping.
    reconstruction_max_attempts: int = 5
    # PnLCalib is the only automatic calibration backend. Docker Compose
    # overrides this with the service-network hostname; an empty value makes
    # automatic Calibration fail closed.
    calibration_worker_url: str | None = "http://127.0.0.1:8090"
    calibration_worker_timeout: float = 900.0
    # A single editor preview should tolerate a cold CPU PnLCalib worker, but it
    # must not inherit the 15-minute background-job timeout.
    calibration_frame_worker_timeout: float = 60.0
    calibration_worker_batch_size: int = 6
    # A direct candidate that is missing or rejected by frame-local QA gets up
    # to two fresh attempts. A retry round batches every pending frame in one
    # API operation (the worker may chunk by its safe inference batch size),
    # while acceptance and attempt audit remain strictly frame-local.
    calibration_pnlcalib_retry_count: int = 2
    # Direct-anchor sampling is a persisted per-scene calibration input. It is
    # intentionally not configurable through process-global settings: every
    # frame is the product default and sparse execution requires an explicit
    # operator choice in the calibration workspace.
    # PnLCalib anchor results are memoized on disk by exact frame bytes and
    # the worker's model identity, so warm rebuilds skip re-uploading and
    # re-inferring anchors that the worker already solved (or already
    # declared unsolvable) for this model.
    calibration_anchor_cache_enabled: bool = True
    # The observed pitch-line mask is a pure function of the frame bytes;
    # caching it removes temporal validation's full second decode pass.
    pitch_line_mask_cache_enabled: bool = True
    # PRTReID lives in its own pinned PyTorch runtime. Keep the URL optional so
    # reconstruction can report missing identity evidence without pretending a
    # generic image embedding is an equivalent fallback.
    identity_worker_url: str | None = "http://127.0.0.1:8091"
    identity_worker_timeout: float = 900.0
    identity_worker_batch_size: int = 4
    # Per-observation embeddings are memoized on disk by the exact crop bytes
    # and the worker's model contract, so warm rebuilds survive worker
    # restarts without re-embedding crops.
    identity_embedding_cache_enabled: bool = True
    # Person crops are cut once in the detection pass (the only frame decode
    # boundary) with the ReID padding policy; ReID and jersey OCR read crop
    # bytes from the store instead of decoding frames again.
    person_crop_padding_ratio: float = 0.08
    person_crop_minimum_width: int = 16
    person_crop_minimum_height: int = 30
    person_crop_minimum_sharpness: float = 12.0
    # A direct calibration anchor whose line-residual tail (p95) is an
    # outlier against the best quartile of accepted anchors is demoted: its
    # frames re-solve temporally from healthier neighbours. Manual anchors
    # are never demoted; ratio<=0 disables the gate.
    calibration_anchor_p95_demotion_enabled: bool = True
    calibration_anchor_p95_demotion_floor: float = 6.5
    calibration_anchor_p95_demotion_ratio: float = 1.6
    # Explicit location of the owner-supplied football.pt person detector;
    # when unset, the repository root and ./models/ are searched (the api
    # image bakes ./models in, mirroring the dedicated ball weights).
    football_detector_weights: str | None = None
    # The "pose-feet" contact-point profile projects RTMPose feet evidence
    # from stored person crops instead of the bbox bottom-centre. The ONNX
    # checkpoint is fetched lazily by rtmlib on first use; a missing runtime
    # degrades to bbox with explicit per-observation diagnostics.
    pose_contact_model_url: str = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip"
    )
    pose_contact_model_input_size: tuple[int, int] = (192, 256)
    pose_contact_device: str = "cpu"
    pose_contact_min_crop_height: int = 48
    pose_contact_min_keypoint_score: float = 0.35
    # A pose-derived contact point deviating from the bbox bottom-centre by
    # more than this fraction of the bbox size is treated as a broken pose.
    pose_contact_max_bbox_deviation_ratio: float = 0.35
    # A crop whose bbox is overlapped by another detection above this IoU
    # contains two players: its embedding is noise for the tracker's ReID
    # gate and is skipped explicitly (0 disables the filter).
    identity_crop_overlap_iou_threshold: float = 0.45
    # One transient transport error must not discard the embeddings that
    # earlier batches already produced: retry the failed batch, then return
    # the partial result with an explicit partialFailure diagnostic.
    identity_worker_batch_retry_count: int = 1
    # Jersey OCR has an intentionally provider-neutral HTTP contract. MMOCR,
    # EasyOCR and a future tracklet-level PARSeq provider live outside the API
    # runtime and must fail explicitly instead of fabricating shirt numbers.
    jersey_ocr_worker_url: str | None = "http://127.0.0.1:8093"
    jersey_ocr_worker_timeout: float = 900.0
    jersey_ocr_worker_batch_size: int = 32
    jersey_ocr_worker_batch_retry_count: int = 1
    jersey_ocr_cache_enabled: bool = True
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def origins(self) -> list[str]:
        return [value.strip() for value in self.cors_origins.split(",") if value.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
