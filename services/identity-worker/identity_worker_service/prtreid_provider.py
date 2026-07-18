from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from typing import Sequence

import numpy as np

from .provider_contract import (
    EMBEDDING_DIMENSION,
    EmbeddingSample,
    ProviderEmbedding,
    ProviderUnavailable,
)
from .prtreid_evidence import file_md5, file_sha256, role_evidence_from_logits
from .prtreid_reference import reference_config, register_soccernet_inference_dataset


BACKEND_NAME = "prtreid-bpbreid-soccernet"
PRTREID_WEIGHTS_MD5 = "9633825232bc89f23a94522c5561650e"
HRNET_WEIGHTS_MD5 = "58ea12b0420aa3adaa2f74114c9f9721"
SOCCERNET_COMMIT = "1c958345067218297d221e45e1a6405f975f83e0"


class PRTReIDProvider:
    """SoccerNet PRTReID/BPBreID feature extractor."""

    backend = BACKEND_NAME
    dimension = EMBEDDING_DIMENSION

    def __init__(self) -> None:
        self.weights_path = Path(
            os.environ.get("PRTREID_WEIGHTS", "/models/prtreid-soccernet-baseline.pth.tar")
        )
        self.hrnet_weights_path = Path(
            os.environ.get(
                "PRTREID_HRNET_WEIGHTS", "/models/hrnetv2_w32_imagenet_pretrained.pth"
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
        actual = file_md5(path)
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
                self.weights_path, PRTREID_WEIGHTS_MD5, "SoccerNet PRTReID checkpoint"
            )
            self._verify_asset(
                self.hrnet_weights_path, HRNET_WEIGHTS_MD5, "HRNet-32 ImageNet checkpoint"
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
                register_soccernet_inference_dataset(prtreid)
                config = CfgNode(
                    reference_config(
                        self.weights_path,
                        self.hrnet_weights_path.parent,
                        device.type == "cuda",
                    )
                )
                config.data.save_dir = runtime_directory.name
                config = build_config(config=config)
                extractor = FeatureExtractor(
                    config,
                    model_path=str(self.weights_path),
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
            self._checkpoint_sha256 = file_sha256(self.weights_path)
            self._hrnet_checkpoint_sha256 = file_sha256(self.hrnet_weights_path)
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
                        role, role_confidence = role_evidence_from_logits(
                            global_role_scores[index]
                        )
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
