from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import string
from typing import Mapping

from .provider_contract import ProviderUnavailable


BACKEND_NAME = "wasb-sbdt-soccer"
FRAMES_IN = 3
FRAMES_OUT = 3
INPUT_WIDTH = 512
INPUT_HEIGHT = 288
CHECKPOINT_SHA256 = "d0369572807c2baf751880d6cdf3cce9fc6283fa8d153f18af6baf4e64d2646c"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _required_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in string.hexdigits for character in normalized
    ):
        raise ProviderUnavailable(
            f"{label} must be a complete 64-character SHA-256 digest"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class WasbConfiguration:
    weights_path: Path
    source_path: Path
    expected_sha256: str
    device_name: str
    score_threshold: float

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "WasbConfiguration":
        values = os.environ if environment is None else environment
        root = _repository_root()
        try:
            score_threshold = float(values.get("WASB_SCORE_THRESHOLD", "0.5"))
        except (TypeError, ValueError) as exc:
            raise ProviderUnavailable("WASB_SCORE_THRESHOLD must be a number") from exc
        if not 0.0 <= score_threshold <= 1.0:
            raise ProviderUnavailable("WASB_SCORE_THRESHOLD must be between 0 and 1")
        device_name = values.get("WASB_DEVICE", "cpu").strip()
        if not device_name:
            raise ProviderUnavailable("WASB_DEVICE must not be empty")
        return cls(
            weights_path=Path(
                values.get(
                    "WASB_WEIGHTS",
                    str(root / "models" / "wasb-soccer-best.pth.tar"),
                )
            ),
            source_path=Path(
                values.get(
                    "WASB_HRNET_SOURCE",
                    str(
                        root
                        / ".references"
                        / "WASB-SBDT"
                        / "src"
                        / "models"
                        / "hrnet.py"
                    ),
                )
            ),
            expected_sha256=_required_sha256(
                values.get("WASB_WEIGHTS_SHA256", CHECKPOINT_SHA256),
                "WASB_WEIGHTS_SHA256",
            ),
            device_name=device_name,
            score_threshold=score_threshold,
        )
