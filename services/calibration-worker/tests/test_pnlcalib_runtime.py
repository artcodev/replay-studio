from __future__ import annotations

import pytest
import torch

from calibration_worker_service.pnlcalib_runtime import resolve_pnlcalib_device


def test_mps_request_fails_closed_when_pytorch_has_no_metal_build(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)

    with pytest.raises(RuntimeError, match="no MPS support"):
        resolve_pnlcalib_device("mps")


def test_mps_request_fails_closed_when_metal_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="Metal device is unavailable"):
        resolve_pnlcalib_device("mps")


def test_cpu_remains_the_portable_default() -> None:
    assert resolve_pnlcalib_device("cpu") == torch.device("cpu")
