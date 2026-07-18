from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "jersey_ocr_worker_service"


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
    assert "cv2" not in source
    assert "PIL" not in source


def test_ocr_processing_does_not_depend_on_http_framework() -> None:
    for name in (
        "request_contract.py",
        "analysis_policy.py",
        "result_cache.py",
        "analysis_service.py",
    ):
        source = (PACKAGE / name).read_text(encoding="utf-8")
        assert "fastapi" not in source
        assert "starlette" not in source


def test_provider_contract_and_backends_have_one_way_dependencies() -> None:
    assert not (PACKAGE / "providers.py").exists()

    contract = (PACKAGE / "provider_contract.py").read_text(encoding="utf-8")
    assert "easyocr_provider" not in contract
    assert "mmocr_provider" not in contract
    assert "provider_factory" not in contract

    for name in ("analysis_policy.py", "analysis_service.py", "result_cache.py"):
        source = (PACKAGE / name).read_text(encoding="utf-8")
        assert "provider_factory" not in source
        assert "easyocr_provider" not in source
        assert "mmocr_provider" not in source

    factory = (PACKAGE / "provider_factory.py").read_text(encoding="utf-8")
    assert "from .easyocr_provider import EasyOCRProvider" in factory
    assert "from .mmocr_provider import MMOCRProvider" in factory
