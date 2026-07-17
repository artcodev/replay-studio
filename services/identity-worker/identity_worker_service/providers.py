from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5, sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from typing import Protocol, Sequence

import numpy as np


BACKEND_NAME = "prtreid-bpbreid-soccernet"
EMBEDDING_DIMENSION = 256
PRTREID_WEIGHTS_MD5 = "9633825232bc89f23a94522c5561650e"
HRNET_WEIGHTS_MD5 = "58ea12b0420aa3adaa2f74114c9f9721"
SOCCERNET_COMMIT = "1c958345067218297d221e45e1a6405f975f83e0"


class _SoccerNetInferenceDataset:
    """Dataset metadata required by PRTReID's inference config builder.

    SoccerNet normally registers its full ``ReidDataset`` from TrackLab. That
    class also imports the complete training/data pipeline, even though
    ``FeatureExtractor`` only asks for ``get_masks_config`` while building the
    model. Keeping the same mask metadata here makes the inference worker
    independent from TrackLab without substituting another dataset or model.
    """

    masks_dirs = {
        "gaussian_joints": (
            10,
            False,
            ".npy",
            [f"p{part}" for part in range(1, 17)],
        ),
        "gaussian_keypoints": (
            17,
            False,
            ".npy",
            [f"p{part}" for part in range(1, 17)],
        ),
        "pose_on_img": (
            35,
            False,
            ".npy",
            [f"p{part}" for part in range(1, 35)],
        ),
    }

    @classmethod
    def get_masks_config(cls, masks_dir: str):
        return cls.masks_dirs.get(masks_dir)


def _register_soccernet_inference_dataset(prtreid_module) -> None:
    """Idempotently expose SoccerNet metadata to PRTReID's registry."""

    from prtreid.data.datasets import get_image_dataset

    try:
        get_image_dataset("SoccerNet")
    except ValueError:
        prtreid_module.data.register_image_dataset(
            "SoccerNet",
            _SoccerNetInferenceDataset,
            "sn",
        )


class ProviderUnavailable(RuntimeError):
    """The configured ReID model is missing, invalid, or could not load."""


@dataclass(frozen=True)
class EmbeddingSample:
    observation_id: str
    image_rgb: np.ndarray


@dataclass(frozen=True)
class ProviderEmbedding:
    observation_id: str
    embedding: np.ndarray
    visibility_scores: np.ndarray | None = None
    role: str | None = None
    role_confidence: float | None = None


class IdentityEmbeddingProvider(Protocol):
    backend: str
    dimension: int

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def info(self) -> dict: ...

    def embed(self, samples: Sequence[EmbeddingSample]) -> list[ProviderEmbedding]: ...


def _file_md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_config(weights_path: Path, model_root: Path, use_gpu: bool) -> dict:
    """Concrete inference config matching the SoccerNet PRTReID baseline.

    The upstream Hydra file contains project-relative interpolations and dataset
    registrations that are only needed for training. Keeping a concrete copy
    here lets the worker use PRTReID's supported FeatureExtractor without
    importing the complete TrackLab pipeline.
    """

    return {
        "use_gpu": use_gpu,
        "project": {
            "name": "ReplayStudioIdentityWorker",
            "experiment_name": "inference",
            "notes": "",
            "tags": [],
            "job_id": "0",
            "logger": {"use_tensorboard": False, "use_wandb": False},
        },
        "data": {
            "root": str(model_root),
            "type": "image",
            "sources": ["SoccerNet"],
            "targets": ["SoccerNet"],
            "height": 256,
            "width": 128,
            "combineall": False,
            "transforms": ["rc", "re"],
            "save_dir": "/tmp/prtreid",
            "workers": 0,
        },
        "sampler": {
            "train_sampler": "PrtreidSampler",
            "train_sampler_t": "PrtreidSampler",
            "num_instances": 4,
        },
        "model": {
            "name": "bpbreid",
            "pretrained": True,
            "save_model_flag": False,
            "load_config": True,
            "load_weights": str(weights_path),
            "bpbreid": {
                "pooling": "gwap",
                "normalization": "identity",
                "mask_filtering_training": False,
                "mask_filtering_testing": False,
                "training_binary_visibility_score": True,
                "testing_binary_visibility_score": True,
                "last_stride": 1,
                "learnable_attention_enabled": False,
                "dim_reduce": "after_pooling",
                "dim_reduce_output": EMBEDDING_DIMENSION,
                "backbone": "hrnet32",
                "test_embeddings": ["globl"],
                "test_use_target_segmentation": "none",
                "shared_parts_id_classifier": False,
                "hrnet_pretrained_path": str(model_root),
                "masks": {"type": "disk", "dir": "", "preprocess": "id"},
            },
        },
        "loss": {
            "name": "part_based",
            "part_based": {
                "name": "part_averaged_triplet_loss",
                "ppl": "cl",
                "weights": {
                    "globl": {"id": 1.0, "tr": 1.0},
                    "foreg": {"id": 0.0, "tr": 0.0},
                    "conct": {"id": 0.0, "tr": 0.0},
                    "parts": {"id": 0.0, "tr": 0.0},
                    "pixls": {"ce": 0.0},
                },
            },
        },
        "train": {"batch_size": 32, "max_epoch": 20},
        "test": {
            "evaluate": True,
            "detailed_ranking": False,
            "start_eval": 40,
            "batch_size": 64,
            "batch_size_pairwise_dist_matrix": 5000,
            "normalize_feature": True,
            "dist_metric": "euclidean",
            "visrank": False,
            "visrank_per_body_part": False,
            "vis_embedding_projection": False,
            "vis_feature_maps": False,
            "visrank_topk": 10,
            "visrank_count": 0,
            "visrank_q_idx_list": [],
            "part_based": {"dist_combine_strat": "mean"},
        },
    }


class PRTReIDProvider:
    """SoccerNet PRTReID/BPBreID feature extractor.

    Imports and model construction are deliberately lazy. The HTTP process can
    stay live and explain a missing dependency or checkpoint while readiness
    remains false. No random model and no generic embedding substitute is ever
    returned.
    """

    backend = BACKEND_NAME
    dimension = EMBEDDING_DIMENSION

    def __init__(self) -> None:
        self.weights_path = Path(
            os.environ.get(
                "PRTREID_WEIGHTS",
                "/models/prtreid-soccernet-baseline.pth.tar",
            )
        )
        self.hrnet_weights_path = Path(
            os.environ.get(
                "PRTREID_HRNET_WEIGHTS",
                "/models/hrnetv2_w32_imagenet_pretrained.pth",
            )
        )
        self.device_name = os.environ.get("REID_DEVICE", "cpu")
        self.batch_size = max(1, int(os.environ.get("REID_BATCH_SIZE", "8")))
        self.soccernet_commit = os.environ.get("SOCCERNET_COMMIT", SOCCERNET_COMMIT)
        self._loaded = False
        self._load_lock = Lock()
        self._inference_lock = Lock()
        self._feature_extractor = None
        self._extract_test_embeddings = None
        self._torch = None
        self._model_version: str | None = None
        self._checkpoint_sha256: str | None = None
        self._hrnet_checkpoint_sha256: str | None = None
        self._model_load_seconds: float | None = None
        self._runtime_directory: TemporaryDirectory[str] | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _verify_asset(self, path: Path, expected_md5: str, label: str) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise ProviderUnavailable(f"{label} is missing: {path}")
        actual = _file_md5(path)
        if actual != expected_md5:
            raise ProviderUnavailable(
                f"{label} checksum mismatch: expected {expected_md5}, received {actual}"
            )

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            started = perf_counter()
            self._verify_asset(
                self.weights_path,
                PRTREID_WEIGHTS_MD5,
                "SoccerNet PRTReID checkpoint",
            )
            self._verify_asset(
                self.hrnet_weights_path,
                HRNET_WEIGHTS_MD5,
                "HRNet-32 ImageNet checkpoint",
            )
            try:
                import prtreid
                import torch
                from prtreid.scripts.main import build_config
                from prtreid.tools.feature_extractor import FeatureExtractor
                from prtreid.utils.tools import extract_test_embeddings
                from yacs.config import CfgNode
            except Exception as exc:  # pragma: no cover - exercised in the image
                raise ProviderUnavailable(f"PRTReID runtime is unavailable: {exc}") from exc

            if self.device_name.startswith("cuda") and not torch.cuda.is_available():
                raise ProviderUnavailable(
                    f"REID_DEVICE={self.device_name} was requested but CUDA is unavailable"
                )
            device = torch.device(self.device_name)
            runtime_directory = TemporaryDirectory(prefix="replay-studio-prtreid-")
            try:
                _register_soccernet_inference_dataset(prtreid)
                config = CfgNode(
                    _reference_config(
                        self.weights_path,
                        self.hrnet_weights_path.parent,
                        device.type == "cuda",
                    )
                )
                # PRTReID's build_config uses os.makedirs without exist_ok and
                # appends project.job_id. Give each load attempt its own parent
                # so a recoverable preload failure never poisons later retries.
                config.data.save_dir = runtime_directory.name
                config = build_config(config=config)
                extractor = FeatureExtractor(
                    config,
                    model_path=str(self.weights_path),
                    # The pinned PRTReID predates torch.device support here and
                    # calls startswith() on this value internally.
                    device=self.device_name,
                    image_size=(256, 128),
                    model=None,
                    verbose=False,
                )
            except Exception as exc:  # pragma: no cover - exercised in the image
                runtime_directory.cleanup()
                raise ProviderUnavailable(f"PRTReID model failed to load: {exc}") from exc

            self._feature_extractor = extractor
            self._extract_test_embeddings = extract_test_embeddings
            self._torch = torch
            self._runtime_directory = runtime_directory
            self._checkpoint_sha256 = _file_sha256(self.weights_path)
            self._hrnet_checkpoint_sha256 = _file_sha256(self.hrnet_weights_path)
            self._model_version = (
                f"{self._checkpoint_sha256[:16]}-sn-{self.soccernet_commit[:12]}"
            )
            self._model_load_seconds = perf_counter() - started
            self._loaded = True

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "dimension": self.dimension,
            "normalized": True,
            "device": self.device_name,
            "batchSize": self.batch_size,
            "modelVersion": self._model_version,
            "checkpointSha256": self._checkpoint_sha256,
            "hrnetCheckpointSha256": self._hrnet_checkpoint_sha256,
            "modelLoadSeconds": self._model_load_seconds,
            "soccerNetCommit": self.soccernet_commit,
        }

    def embed(self, samples: Sequence[EmbeddingSample]) -> list[ProviderEmbedding]:
        self.load()
        if not samples:
            return []
        assert self._feature_extractor is not None
        assert self._extract_test_embeddings is not None
        assert self._torch is not None
        role_names = {0: "ball", 1: "goalkeeper", 2: "other", 3: "player", 4: "referee"}
        output: list[ProviderEmbedding] = []
        with self._inference_lock, self._torch.no_grad():
            for start in range(0, len(samples), self.batch_size):
                batch = list(samples[start : start + self.batch_size])
                images = [np.asarray(item.image_rgb) for item in batch]
                try:
                    raw = self._feature_extractor(images, external_parts_masks=None)
                    embeddings, visibility, _body_masks, _parts, role_scores = (
                        self._extract_test_embeddings(raw, ["globl"])
                    )
                except Exception as exc:
                    raise ProviderUnavailable(f"PRTReID inference failed: {exc}") from exc
                vectors = embeddings.detach().cpu().numpy()
                visibility_values = (
                    visibility.detach().cpu().numpy() if visibility is not None else None
                )
                global_role_scores = (
                    role_scores.get("globl").detach().cpu().numpy()
                    if role_scores is not None and role_scores.get("globl") is not None
                    else None
                )
                for index, sample in enumerate(batch):
                    vector = np.asarray(vectors[index], dtype=np.float32).reshape(-1)
                    if vector.size != self.dimension or not np.isfinite(vector).all():
                        raise ProviderUnavailable("PRTReID returned an invalid embedding")
                    norm = float(np.linalg.norm(vector))
                    if norm < 1e-12:
                        raise ProviderUnavailable("PRTReID returned a zero embedding")
                    vector /= norm
                    role = None
                    role_confidence = None
                    if global_role_scores is not None:
                        scores = np.asarray(global_role_scores[index]).reshape(-1)
                        role_index = int(np.argmax(scores))
                        role = role_names.get(role_index)
                        role_confidence = float(scores[role_index])
                    output.append(
                        ProviderEmbedding(
                            observation_id=sample.observation_id,
                            embedding=vector,
                            visibility_scores=(
                                np.asarray(visibility_values[index], dtype=np.float32)
                                if visibility_values is not None
                                else None
                            ),
                            role=role,
                            role_confidence=role_confidence,
                        )
                    )
        return output
