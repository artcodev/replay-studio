from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "ball_worker_service"


def test_main_is_only_the_http_composition_root() -> None:
    source = (PACKAGE / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_functions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert top_level_functions == ["create_app"]
    assert "numpy" not in source
    assert "PIL" not in source


def test_ball_processing_does_not_depend_on_http_framework() -> None:
    for name in (
        "settings.py",
        "request_contract.py",
        "provider_contract.py",
        "detection_service.py",
        "wasb_configuration.py",
        "wasb_geometry.py",
        "wasb_model_loading.py",
        "wasb_provider.py",
    ):
        source = (PACKAGE / name).read_text(encoding="utf-8")
        assert "fastapi" not in source
        assert "starlette" not in source


def test_provider_aggregate_was_deleted_and_dependencies_are_one_way() -> None:
    assert not (PACKAGE / "providers.py").exists()
    package_source = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKAGE.glob("*.py")
    )
    assert ".providers" not in package_source

    contract_source = (PACKAGE / "provider_contract.py").read_text(encoding="utf-8")
    assert "wasb_" not in contract_source

    service_source = (PACKAGE / "detection_service.py").read_text(encoding="utf-8")
    assert "from .provider_contract import" in service_source
    assert "wasb_provider" not in service_source
    assert "provider_factory" not in service_source

    main_source = (PACKAGE / "main.py").read_text(encoding="utf-8")
    assert "from .provider_contract import BallDetectionProvider" in main_source
    assert "from .provider_factory import provider_from_environment" in main_source
    assert "wasb_provider" not in main_source


def test_worker_exposes_only_the_canonical_multipart_detection_contract() -> None:
    main_source = (PACKAGE / "main.py").read_text(encoding="utf-8")
    package_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PACKAGE.glob("*.py")
    )
    assert '@application.post("/v1/detections")' in main_source
    assert '@application.post("/detect")' not in main_source
    assert "compatible_detection" not in package_source
    assert "dataBase64" not in package_source
    assert "base64" not in package_source
