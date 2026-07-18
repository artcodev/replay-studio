from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_validation.manifest_contract import MANIFEST_SCHEMA_VERSION
from model_validation.manifest_loader import load_manifest
from model_validation.report_writer import write_report
from model_validation.reports import REPORT_VERSION, build_report, build_unavailable_report


MANIFEST = Path(__file__).resolve().parent / "fixtures" / "fake-manifest.json"


def test_combined_report_is_versioned_and_binds_manifest_fingerprint():
    manifest = load_manifest(MANIFEST)

    report = build_report(
        manifest,
        identity={"status": "pass"},
        jersey_ocr={"status": "pass"},
    )

    assert report["reportVersion"] == REPORT_VERSION
    assert report["manifestSchemaVersion"] == MANIFEST_SCHEMA_VERSION
    assert report["dataset"]["fingerprint"].startswith("sha256:")
    assert report["status"] == "pass"
    assert report["thresholds"] == manifest.thresholds


def test_report_requires_at_least_one_worker_result():
    with pytest.raises(ValueError, match="At least one worker result"):
        build_report(load_manifest(MANIFEST))


def test_unavailable_report_and_atomic_writer_preserve_contract(tmp_path):
    manifest = load_manifest(MANIFEST)
    report = build_unavailable_report(
        manifest,
        selected_workers=("identity",),
        reason="worker unavailable",
    )

    target = write_report(tmp_path / "nested" / "report.json", report)

    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert not list(target.parent.glob(".report.json.tmp-*"))
    assert report["status"] == "unavailable"
    assert report["selectedWorkers"] == ["identity"]
