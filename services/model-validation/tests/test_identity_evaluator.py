from __future__ import annotations

from pathlib import Path

import numpy as np

from model_validation.identity_evaluator import IDENTITY_DIMENSION, evaluate_identity
from model_validation.manifest_loader import load_manifest


MANIFEST = Path(__file__).resolve().parent / "fixtures" / "fake-manifest.json"


def _provider() -> dict:
    return {
        "backend": "prtreid-bpbreid-soccernet",
        "dimension": IDENTITY_DIMENSION,
        "normalized": True,
        "modelVersion": "aaaaaaaaaaaaaaaa-sn-cccccccccccc",
        "checkpointSha256": "a" * 64,
        "hrnetCheckpointSha256": "b" * 64,
        "soccerNetCommit": "c" * 40,
    }


def _vector(*values: float) -> list[float]:
    result = np.zeros(IDENTITY_DIMENSION, dtype=np.float64)
    result[: len(values)] = values
    result /= np.linalg.norm(result)
    return result.tolist()


def _predictions() -> dict[str, dict]:
    return {
        "p8-a": {"usable": True, "embedding": _vector(1.0, 0.0, 0.0), "role": "player"},
        "p8-b": {"usable": True, "embedding": _vector(0.999, 0.02, 0.0), "role": "player"},
        "gk1": {"usable": True, "embedding": _vector(0.0, 1.0, 0.0), "role": "goalkeeper"},
        "ref-hidden": {"usable": True, "embedding": _vector(0.0, 0.0, 1.0), "role": "referee"},
    }


def test_identity_evaluation_covers_distributions_roles_and_provenance():
    result = evaluate_identity(load_manifest(MANIFEST), _provider(), _predictions())

    assert result["status"] == "pass"
    assert result["metrics"]["embeddingContract"]["dimension"] == 256
    assert result["metrics"]["samePersonDistance"]["count"] == 1
    assert result["metrics"]["differentPersonDistance"]["count"] == 2
    assert result["metrics"]["medianDistanceSeparation"] > 0.9
    assert result["metrics"]["roleConfusion"]["accuracy"] == 1.0
    assert all(check["passed"] for check in result["checks"])


def test_identity_contract_fails_wrong_dimension_or_checkpoint_provenance():
    provider = _provider()
    provider["checkpointSha256"] = None
    predictions = _predictions()
    predictions["p8-b"]["embedding"] = [1.0, 0.0]

    result = evaluate_identity(load_manifest(MANIFEST), provider, predictions)

    assert result["status"] == "fail"
    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert "identity-provider-provenance" in failed
    assert "identity-embedding-contract" in failed
    assert result["metrics"]["pairCoverage"] < 1.0


def test_identity_contract_rejects_non_normalized_vector_and_fake_backend():
    provider = _provider()
    provider["backend"] = "fake-provider"
    predictions = _predictions()
    predictions["p8-a"]["embedding"] = [2.0, *([0.0] * (IDENTITY_DIMENSION - 1))]

    result = evaluate_identity(load_manifest(MANIFEST), provider, predictions)

    assert result["status"] == "fail"
    row = next(item for item in result["samples"]["crops"] if item["cropId"] == "p8-a")
    assert "embedding-not-normalized" in row["reasons"]
    assert not next(
        check
        for check in result["checks"]
        if check["id"] == "identity-provider-provenance"
    )["passed"]
