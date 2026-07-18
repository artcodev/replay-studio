from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from model_validation.manifest_loader import load_manifest
from model_validation.orchestration import run_http_validation
from model_validation.worker_transport import WorkerUnavailable


def _real_manifest():
    if os.environ.get("MODEL_VALIDATION_OPT_IN") != "1":
        pytest.skip(
            "real model validation is opt-in; set MODEL_VALIDATION_OPT_IN=1 only after worker assets are provisioned"
        )
    raw_path = os.environ.get("MODEL_VALIDATION_MANIFEST")
    if not raw_path:
        pytest.skip(
            "real labelled assets unavailable: MODEL_VALIDATION_MANIFEST is not set"
        )
    path = Path(raw_path).expanduser()
    if not path.is_file():
        pytest.skip(f"real labelled assets unavailable: manifest does not exist: {path}")
    return load_manifest(path)


def _run(worker: str) -> dict:
    manifest = _real_manifest()
    try:
        return run_http_validation(
            manifest,
            workers=(worker,),
            identity_url=os.environ.get(
                "IDENTITY_WORKER_URL", "http://127.0.0.1:8091"
            ),
            jersey_ocr_url=os.environ.get(
                "JERSEY_OCR_WORKER_URL", "http://127.0.0.1:8093"
            ),
            timeout_seconds=float(os.environ.get("MODEL_VALIDATION_TIMEOUT", "900")),
        )
    except WorkerUnavailable as exc:
        pytest.skip(f"real {worker} assets/worker unavailable: {exc}")


def test_real_identity_worker_on_labelled_manifest():
    report = _run("identity")

    assert report["status"] == "pass", json.dumps(report, indent=2, sort_keys=True)


def test_real_jersey_ocr_worker_on_labelled_manifest():
    report = _run("jersey-ocr")

    assert report["status"] == "pass", json.dumps(report, indent=2, sort_keys=True)
