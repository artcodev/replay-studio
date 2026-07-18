from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "model_validation"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_obsolete_validation_harness_is_absent():
    assert not (ROOT / "validation_harness.py").exists()


def test_manifest_aggregate_is_deleted_and_capabilities_have_one_way_dependencies():
    assert not (PACKAGE / "manifest.py").exists()

    contract_imports = _imports(PACKAGE / "manifest_contract.py")
    fingerprint_imports = _imports(PACKAGE / "manifest_fingerprint.py")
    loader_imports = _imports(PACKAGE / "manifest_loader.py")
    capability_modules = {
        "manifest_loader",
        "manifest_crop_labels",
        "manifest_dataset",
        "manifest_fingerprint",
        "manifest_identity_pairs",
        "manifest_parsing",
        "manifest_thresholds",
    }
    assert contract_imports.isdisjoint(capability_modules)
    assert "manifest_loader" not in fingerprint_imports
    assert {
        "manifest_contract",
        "manifest_crop_labels",
        "manifest_dataset",
        "manifest_fingerprint",
        "manifest_identity_pairs",
        "manifest_parsing",
        "manifest_thresholds",
    }.issubset(loader_imports)

    for path in PACKAGE.glob("*.py"):
        assert "model_validation.manifest" not in path.read_text(encoding="utf-8")


def test_package_root_is_not_a_compatibility_facade():
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")

    assert "from ." not in source
    assert "__all__" not in source


def test_pure_evaluators_do_not_depend_on_transport_or_report_persistence():
    forbidden = {
        "httpx",
        "model_validation.orchestration",
        "model_validation.report_writer",
        "model_validation.identity_worker_client",
        "model_validation.jersey_worker_client",
        "model_validation.worker_transport",
        "orchestration",
        "report_writer",
        "identity_worker_client",
        "jersey_worker_client",
        "worker_transport",
    }
    for filename in ("identity_evaluator.py", "jersey_evaluator.py", "evaluation_primitives.py"):
        assert _imports(PACKAGE / filename).isdisjoint(forbidden), filename


def test_only_orchestration_owns_optional_http_dependency():
    httpx_owners = [
        path.name
        for path in PACKAGE.glob("*.py")
        if "httpx" in (PACKAGE / path.name).read_text(encoding="utf-8")
    ]

    assert httpx_owners == ["orchestration.py"]


def test_worker_protocols_have_separate_clients():
    identity_source = (PACKAGE / "identity_worker_client.py").read_text(encoding="utf-8")
    jersey_source = (PACKAGE / "jersey_worker_client.py").read_text(encoding="utf-8")

    assert "/v1/embeddings" in identity_source
    assert "/v1/analyze" not in identity_source
    assert "/v1/analyze" in jersey_source
    assert "/v1/embeddings" not in jersey_source
