from __future__ import annotations

from pathlib import Path

from model_validation.jersey_evaluator import evaluate_jersey_ocr
from model_validation.manifest_loader import load_manifest


MANIFEST = Path(__file__).resolve().parent / "fixtures" / "fake-manifest.json"


def _provider() -> dict:
    return {
        "backend": "mmocr-dbnet18-sar",
        "providerVersion": "1.0.1",
        "modelVersion": "mmocr-1.0.1/dbnet/SAR",
        "contractVersion": "jersey-ocr.v1",
        "inferenceScope": "crop",
    }


def _predictions() -> dict[str, dict]:
    return {
        "p8-a": {"usable": True, "status": "recognized", "number": "8"},
        "p8-b": {"usable": True, "status": "recognized", "number": "8"},
        "gk1": {"usable": True, "status": "recognized", "number": "1"},
        "ref-hidden": {"usable": True, "status": "no-number", "number": None},
    }


def test_ocr_evaluation_covers_exact_abstention_and_conflict_metrics():
    result = evaluate_jersey_ocr(load_manifest(MANIFEST), _provider(), _predictions())

    assert result["status"] == "pass"
    assert result["metrics"]["readableExactAccuracy"] == 1.0
    assert result["metrics"]["expectedAbstentionAccuracy"] == 1.0
    assert result["metrics"]["conflictGroupRate"] == 0.0
    assert result["metrics"]["abstentionPrecision"] == 1.0


def test_ocr_substitution_and_same_person_number_conflict_fail_closed():
    predictions = _predictions()
    predictions["p8-b"] = {"usable": True, "status": "recognized", "number": "9"}

    result = evaluate_jersey_ocr(load_manifest(MANIFEST), _provider(), predictions)

    assert result["status"] == "fail"
    assert result["metrics"]["substitutionCount"] == 1
    assert result["metrics"]["conflictGroupCount"] == 1
    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert "jersey-readable-exact-accuracy" in failed
    assert "jersey-substitution-rate" in failed
    assert "jersey-conflict-group-rate" in failed


def test_ocr_provenance_rejects_an_injected_fake_backend():
    provider = _provider()
    provider["backend"] = "fake-ocr"

    result = evaluate_jersey_ocr(load_manifest(MANIFEST), provider, _predictions())

    assert result["status"] == "fail"
    provenance = next(
        check for check in result["checks"] if check["id"] == "jersey-provider-provenance"
    )
    assert provenance["passed"] is False
