from __future__ import annotations

from collections import OrderedDict
from contextlib import asynccontextmanager
from copy import deepcopy
from hashlib import sha256
import io
import json
import os
import sys
from dataclasses import dataclass
from math import exp
from pathlib import Path
from threading import Lock
from time import monotonic, perf_counter

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from starlette.concurrency import run_in_threadpool


INPUT_WIDTH = 960
INPUT_HEIGHT = 540
KEYPOINT_THRESHOLD = 0.1611
LINE_THRESHOLD = 0.3434
CACHE_SCHEMA_VERSION = "pnlcalib-points-lines-v2"

# Official PnLCalib line-channel order.  The network has one additional
# background channel which is removed before ``coords_to_dict``.  Goal-frame
# segments are useful visual evidence but are not coplanar with the grass, so
# they are exposed to callers while remaining excluded from the ground-plane
# homography fit.
SEMANTIC_LINE_NAMES = (
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Goal left crossbar",
    "Goal left post left",
    "Goal left post right",
    "Goal right crossbar",
    "Goal right post left",
    "Goal right post right",
    "Middle line",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Side line top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
)
GOAL_FRAME_LINE_IDS = frozenset(range(7, 13))


class AttrDict(dict):
    """Small yacs-compatible subset used by the official HRNet code."""

    __getattr__ = dict.__getitem__


def _attr_dict(value):
    if isinstance(value, dict):
        return AttrDict({key: _attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_attr_dict(item) for item in value]
    return value


SOCCERNET_ROOT = Path(os.environ.get("SOCCERNET_ROOT", "/opt/sn-gamestate"))
PLUGIN_ROOT = SOCCERNET_ROOT / "plugins" / "calibration"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from pnlcalib.model.cls_hrnet import get_cls_net  # noqa: E402
from pnlcalib.model.cls_hrnet_l import get_cls_net as get_cls_net_l  # noqa: E402
from pnlcalib.utils.utils_calib import FramebyFrameCalib  # noqa: E402
from pnlcalib.utils.utils_heatmap import (  # noqa: E402
    complete_keypoints,
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)


@dataclass
class DecodedFrame:
    frame_index: int
    width: int
    height: int
    tensor: torch.Tensor
    content_sha256: str


@dataclass
class _CacheEntry:
    created_at: float
    result: dict | None


class PnLCalibEngine:
    def __init__(self) -> None:
        started = perf_counter()
        requested_device = os.environ.get("PNLCALIB_DEVICE", "cpu")
        self.device = torch.device(requested_device)
        self.batch_size = max(1, int(os.environ.get("PNLCALIB_BATCH_SIZE", "2")))
        self.cache_max_entries = max(0, int(os.environ.get("PNLCALIB_CACHE_MAX_ENTRIES", "512")))
        self.cache_ttl_seconds = max(
            0.0,
            float(os.environ.get("PNLCALIB_CACHE_TTL_SECONDS", "3600")),
        )
        model_root = PLUGIN_ROOT / "pnlcalib"
        keypoint_weights = Path(
            os.environ.get("PNLCALIB_KEYPOINT_WEIGHTS", "/models/pnl_SV_kp")
        )
        line_weights = Path(
            os.environ.get("PNLCALIB_LINE_WEIGHTS", "/models/pnl_SV_lines")
        )
        self.model_version = self._model_version(keypoint_weights, line_weights)
        self.keypoint_model = self._load_model(
            model_root / "config" / "hrnetv2_w48.yaml",
            keypoint_weights,
            get_cls_net,
        )
        self.line_model = self._load_model(
            model_root / "config" / "hrnetv2_w48_l.yaml",
            line_weights,
            get_cls_net_l,
        )
        self.lock = Lock()
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self.model_load_seconds = perf_counter() - started

    @staticmethod
    def _model_version(*weights_paths: Path) -> str:
        """Return a process-local model/config identity without rereading 500 MB."""

        identity = [CACHE_SCHEMA_VERSION, str(INPUT_WIDTH), str(INPUT_HEIGHT)]
        for path in weights_paths:
            stat = path.stat()
            identity.extend(
                [str(path.resolve()), str(stat.st_size), str(stat.st_mtime_ns)]
            )
        return sha256("\n".join(identity).encode("utf-8")).hexdigest()[:16]

    def _load_model(self, config_path: Path, weights_path: Path, factory):
        if not weights_path.is_file():
            raise RuntimeError(f"PnLCalib weights are missing: {weights_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = _attr_dict(yaml.safe_load(handle))
        model = factory(config)
        try:
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def decode(frame_index: int, data: bytes) -> DecodedFrame:
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Frame {frame_index} is not a readable image") from exc
        width, height = image.size
        resized = (
            image
            if image.size == (INPUT_WIDTH, INPUT_HEIGHT)
            else image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.BILINEAR)
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return DecodedFrame(frame_index, width, height, tensor, sha256(data).hexdigest())

    @staticmethod
    def _side(keypoints: dict[int, dict]) -> str | None:
        visible = [
            (float(item.get("xw", item.get("x"))), float(item.get("p", 1.0)))
            for item in keypoints.values()
            if ("xw" in item or "x" in item) and float(item.get("p", 1.0)) > 0.0
        ]
        if not visible:
            return None
        weighted_x = sum(x * weight for x, weight in visible) / sum(weight for _, weight in visible)
        if abs(weighted_x) < 4.0:
            return None
        return "left" if weighted_x < 0.0 else "right"

    @staticmethod
    def _original_homography(homography: np.ndarray, width: int, height: int) -> np.ndarray:
        original_to_resized = np.array(
            [
                [INPUT_WIDTH / width, 0.0, 0.0],
                [0.0, INPUT_HEIGHT / height, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        result = np.asarray(homography, dtype=np.float64) @ original_to_resized
        result /= result[2, 2]
        return result

    @staticmethod
    def _raw_line_evidence(
        frame: DecodedFrame,
        detected_lines: dict[int, dict],
    ) -> list[dict]:
        """Return immutable, original-frame semantic line observations.

        ``complete_keypoints(..., normalize=True)`` mutates its line dictionary
        in place.  This snapshot is therefore created before calibration and
        converts the 960x540 network coordinates directly to the source frame.
        """

        scale_x = frame.width / INPUT_WIDTH
        scale_y = frame.height / INPUT_HEIGHT
        evidence = []
        for line_id, item in sorted(detected_lines.items()):
            if not 1 <= int(line_id) <= len(SEMANTIC_LINE_NAMES):
                continue
            confidence = min(
                float(item.get("p_1", 0.0)),
                float(item.get("p_2", 0.0)),
            )
            evidence.append(
                {
                    "id": int(line_id),
                    "name": SEMANTIC_LINE_NAMES[int(line_id) - 1],
                    "start": {
                        "x": round(float(item["x_1"]) * scale_x, 3),
                        "y": round(float(item["y_1"]) * scale_y, 3),
                    },
                    "end": {
                        "x": round(float(item["x_2"]) * scale_x, 3),
                        "y": round(float(item["y_2"]) * scale_y, 3),
                    },
                    # Both endpoint heatmaps must clear the detector threshold;
                    # the weaker endpoint is the conservative segment score.
                    "confidence": round(confidence, 5),
                    "groundPlane": int(line_id) not in GOAL_FRAME_LINE_IDS,
                }
            )
        return evidence

    def _calibrate_one(
        self,
        frame: DecodedFrame,
        keypoints: dict[int, dict],
        lines: dict[int, dict],
        raw_lines: list[dict] | None = None,
    ) -> dict | None:
        completed_keypoints, normalized_lines = complete_keypoints(
            keypoints,
            lines,
            w=INPUT_WIDTH,
            h=INPUT_HEIGHT,
            normalize=True,
        )
        camera = FramebyFrameCalib(INPUT_WIDTH, INPUT_HEIGHT, denormalize=True)
        camera.update(completed_keypoints, normalized_lines)
        result = camera.heuristic_voting_ground(refine_lines=True)
        if result is None or result.get("homography") is None:
            return None
        matrix = self._original_homography(result["homography"], frame.width, frame.height)
        if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-10:
            return None
        raw_error = result.get("rep_err")
        error = float(raw_error) if raw_error is not None else 999.0
        if not np.isfinite(error):
            return None
        ground = camera.subsets.get("ground_plane") or {}
        completed_keypoint_count = len(ground)
        # `complete_keypoints` is useful for camera fitting, but inferred points
        # are not independent evidence.  Quality and inlier counts must be based
        # on semantic points that were actually emitted by the network.
        raw_ground = {
            key: ground[key]
            for key in keypoints
            if key in ground
            and all(name in ground[key] for name in ("xi", "yi", "xw", "yw"))
        }
        detected_count = len(raw_ground)
        if detected_count < 6 or error > 18.0:
            return None
        image_points = np.array(
            [[float(item["xi"]), float(item["yi"]), 1.0] for item in raw_ground.values()],
            dtype=np.float64,
        )
        world_points = np.array(
            [[float(item["xw"]), float(item["yw"])] for item in raw_ground.values()],
            dtype=np.float64,
        )
        projected = image_points @ np.asarray(result["homography"], dtype=np.float64).T
        valid = np.abs(projected[:, 2]) > 1e-8
        world_error = np.full(len(projected), np.inf, dtype=np.float64)
        world_error[valid] = np.linalg.norm(
            projected[valid, :2] / projected[valid, 2:3] - world_points[valid],
            axis=1,
        )
        inlier_count = int((world_error <= 1.5).sum())
        if inlier_count < 6 or inlier_count / detected_count < 0.65:
            return None
        line_count = len(lines)
        inlier_ratio = inlier_count / detected_count
        finite_world_error = world_error[np.isfinite(world_error)]
        ground_error_p50 = float(np.median(finite_world_error))
        ground_error_p95 = float(np.percentile(finite_world_error, 95))
        point_coverage = min(1.0, detected_count / 10.0)
        line_coverage = min(1.0, line_count / 8.0)
        error_score = exp(-error / 8.0)
        confidence = float(
            np.clip(
                0.25 * point_coverage
                + 0.25 * inlier_ratio
                + 0.35 * error_score
                + 0.15 * line_coverage,
                0.0,
                0.99,
            )
        )
        scale_x = frame.width / INPUT_WIDTH
        scale_y = frame.height / INPUT_HEIGHT
        raw_keypoints = []
        for (key, item), residual in zip(raw_ground.items(), world_error):
            raw_keypoints.append(
                {
                    "id": int(key),
                    "image": {
                        "x": round(float(item["xi"]) * scale_x, 3),
                        "y": round(float(item["yi"]) * scale_y, 3),
                    },
                    "pitch": {
                        "x": round(float(item["xw"]), 4),
                        "z": round(float(item["yw"]), 4),
                    },
                    "confidence": round(float(keypoints[key].get("p", 1.0)), 5),
                    "inlier": bool(np.isfinite(residual) and residual <= 1.5),
                    "groundResidualMetres": (
                        round(float(residual), 5) if np.isfinite(residual) else None
                    ),
                }
            )
        return {
            "frameIndex": frame.frame_index,
            "method": "pnlcalib-points-lines",
            "confidence": round(confidence, 5),
            "confidenceKind": "heuristic-quality-score",
            "keypointCount": detected_count,
            "detectedKeypointCount": detected_count,
            "completedKeypointCount": completed_keypoint_count,
            "inlierCount": inlier_count,
            "inlierRatio": round(inlier_ratio, 5),
            "lineCount": line_count,
            "detectedLineCount": len(raw_lines or ()),
            "rawLines": list(raw_lines or ()),
            "matchedCurves": sum(key in keypoints for key in range(31, 58)),
            "completedCurveCount": sum(
                key in completed_keypoints for key in range(31, 58)
            ),
            "reprojectionError": round(error, 5),
            "groundErrorP50Metres": round(ground_error_p50, 5),
            "groundErrorP95Metres": round(ground_error_p95, 5),
            "pitchSide": self._side(raw_ground),
            "rawKeypoints": raw_keypoints,
            "imageToPitch": [[round(float(value), 10) for value in row] for row in matrix],
        }

    def _cache_key(self, frame: DecodedFrame) -> str:
        return f"{self.model_version}:{frame.content_sha256}"

    def _cache_get(self, key: str, now: float) -> tuple[bool, dict | None]:
        if self.cache_max_entries <= 0:
            return False, None
        entry = self._cache.get(key)
        if entry is None:
            return False, None
        if self.cache_ttl_seconds > 0.0 and now - entry.created_at > self.cache_ttl_seconds:
            self._cache.pop(key, None)
            return False, None
        self._cache.move_to_end(key)
        return True, deepcopy(entry.result)

    def _cache_put(self, key: str, result: dict | None, now: float) -> None:
        if self.cache_max_entries <= 0:
            return
        cached = deepcopy(result)
        if cached is not None:
            cached.pop("frameIndex", None)
        self._cache[key] = _CacheEntry(created_at=now, result=cached)
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_max_entries:
            self._cache.popitem(last=False)

    @staticmethod
    def _result_for_frame(result: dict | None, frame_index: int) -> dict | None:
        if result is None:
            return None
        resolved = deepcopy(result)
        resolved["frameIndex"] = frame_index
        return resolved

    def _infer_batch(self, frames: list[DecodedFrame], timings: dict[str, float]) -> list[dict | None]:
        started = perf_counter()
        batch = torch.stack([frame.tensor for frame in frames])
        if self.device.type != "cpu":
            batch = batch.to(self.device)
        timings["tensorAssemblySeconds"] += perf_counter() - started

        started = perf_counter()
        heatmaps = self.keypoint_model(batch)
        timings["keypointInferenceSeconds"] += perf_counter() - started

        started = perf_counter()
        line_heatmaps = self.line_model(batch)
        timings["lineInferenceSeconds"] += perf_counter() - started

        started = perf_counter()
        keypoint_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(line_heatmaps[:, :-1])
        keypoint_items = coords_to_dict(
            keypoint_coords,
            threshold=KEYPOINT_THRESHOLD,
            ground_plane_only=True,
        )
        line_items = coords_to_dict(
            line_coords,
            threshold=LINE_THRESHOLD,
            ground_plane_only=False,
        )
        timings["heatmapDecodeSeconds"] += perf_counter() - started

        started = perf_counter()
        output: list[dict | None] = []
        for frame, keypoints, detected_lines in zip(frames, keypoint_items, line_items):
            raw_lines = self._raw_line_evidence(frame, detected_lines)
            ground_lines = {
                key: value
                for key, value in detected_lines.items()
                if key not in GOAL_FRAME_LINE_IDS
            }
            output.append(
                self._calibrate_one(
                    frame,
                    keypoints,
                    ground_lines,
                    raw_lines=raw_lines,
                )
            )
        timings["geometrySeconds"] += perf_counter() - started
        return output

    def calibrate(
        self,
        frames: list[DecodedFrame],
        diagnostics: dict | None = None,
    ) -> list[dict]:
        engine_started = perf_counter()
        timings = {
            "tensorAssemblySeconds": 0.0,
            "keypointInferenceSeconds": 0.0,
            "lineInferenceSeconds": 0.0,
            "heatmapDecodeSeconds": 0.0,
            "geometrySeconds": 0.0,
        }
        lock_started = perf_counter()
        with self.lock, torch.inference_mode():
            lock_wait = perf_counter() - lock_started
            now = monotonic()
            resolved: list[dict | None] = [None] * len(frames)
            misses: OrderedDict[str, list[int]] = OrderedDict()
            cache_hit_count = 0
            for index, frame in enumerate(frames):
                key = self._cache_key(frame)
                hit, cached = self._cache_get(key, now)
                if hit:
                    cache_hit_count += 1
                    resolved[index] = self._result_for_frame(cached, frame.frame_index)
                else:
                    misses.setdefault(key, []).append(index)

            miss_items = list(misses.items())
            for start in range(0, len(miss_items), self.batch_size):
                batch_items = miss_items[start : start + self.batch_size]
                batch_frames = [frames[indices[0]] for _, indices in batch_items]
                batch_results = self._infer_batch(batch_frames, timings)
                for (key, indices), result in zip(batch_items, batch_results):
                    self._cache_put(key, result, monotonic())
                    for index in indices:
                        resolved[index] = self._result_for_frame(
                            result,
                            frames[index].frame_index,
                        )

            if diagnostics is not None:
                diagnostics.clear()
                diagnostics.update(
                    {
                        "modelVersion": self.model_version,
                        "requestedFrameCount": len(frames),
                        "uniqueFrameCount": len({self._cache_key(frame) for frame in frames}),
                        "cacheHitCount": cache_hit_count,
                        "cacheMissCount": len(miss_items),
                        "deduplicatedFrameCount": sum(
                            max(0, len(indices) - 1) for _, indices in miss_items
                        ),
                        "inferenceBatchCount": (
                            (len(miss_items) + self.batch_size - 1) // self.batch_size
                            if miss_items
                            else 0
                        ),
                        "cacheEntryCount": len(self._cache),
                        "lockWaitSeconds": lock_wait,
                        **timings,
                    }
                )
                diagnostics["modelInferenceSeconds"] = (
                    diagnostics["keypointInferenceSeconds"]
                    + diagnostics["lineInferenceSeconds"]
                )
                diagnostics["engineSeconds"] = perf_counter() - engine_started
                for key, value in tuple(diagnostics.items()):
                    if key.endswith("Seconds") and isinstance(value, float):
                        diagnostics[key] = round(value, 6)
        return [item for item in resolved if item is not None]


_engine: PnLCalibEngine | None = None
_engine_lock = Lock()


def get_engine() -> PnLCalibEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = PnLCalibEngine()
    return _engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Compose already waits for readiness. Preloading here also makes a locally
    # launched worker honest: once it accepts traffic, both HRNet models exist.
    if os.environ.get("PNLCALIB_PRELOAD", "1").lower() not in {"0", "false", "no"}:
        try:
            await run_in_threadpool(get_engine)
        except Exception:
            # Preserve the useful liveness/readiness split. Readiness reports the
            # concrete model error while the process remains inspectable.
            pass
    yield


app = FastAPI(
    title="Replay Studio PnLCalib Worker",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health/live")
def liveness() -> dict:
    return {"status": "ok", "service": "pnlcalib-worker"}


@app.get("/health/ready")
def readiness() -> dict:
    try:
        engine = get_engine()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ready",
        "backend": "pnlcalib-points-lines",
        "device": str(engine.device),
        "batchSize": engine.batch_size,
        "modelVersion": engine.model_version,
        "modelLoadSeconds": round(engine.model_load_seconds, 3),
        "cacheMaxEntries": engine.cache_max_entries,
        "cacheTtlSeconds": engine.cache_ttl_seconds,
        "cacheEntryCount": len(engine._cache),
    }


@app.get("/health")
def health() -> dict:
    """Backward-compatible readiness endpoint."""

    return readiness()


@app.post("/v1/calibrate")
async def calibrate(
    frames: list[UploadFile] = File(...),
    frame_indices: str = Form(...),
) -> dict:
    request_started = perf_counter()
    try:
        indices = json.loads(frame_indices)
        if not isinstance(indices, list) or len(indices) != len(frames):
            raise ValueError
        payloads = [await upload.read() for upload in frames]
        decode_started = perf_counter()
        decoded = await run_in_threadpool(
            lambda: [
                PnLCalibEngine.decode(int(index), payload)
                for index, payload in zip(indices, payloads)
            ]
        )
        decode_seconds = perf_counter() - decode_started
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="frame_indices must match the uploaded frames") from exc
    try:
        acquire_started = perf_counter()
        engine = await run_in_threadpool(get_engine)
        engine_acquire_seconds = perf_counter() - acquire_started
        engine_diagnostics: dict = {}
        calibrated = await run_in_threadpool(
            engine.calibrate,
            decoded,
            engine_diagnostics,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PnLCalib inference failed: {exc}") from exc
    diagnostics = {
        **engine_diagnostics,
        "decodeSeconds": round(decode_seconds, 6),
        "engineAcquireSeconds": round(engine_acquire_seconds, 6),
        "totalSeconds": round(perf_counter() - request_started, 6),
    }
    return {
        "backend": "pnlcalib-points-lines",
        "requestedFrameCount": len(decoded),
        "calibratedFrameCount": len(calibrated),
        "diagnostics": diagnostics,
        "frames": calibrated,
    }
