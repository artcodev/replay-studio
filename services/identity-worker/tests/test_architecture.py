from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "identity_worker_service"


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


def test_identity_processing_does_not_depend_on_http_framework() -> None:
    for name in ("request_contract.py", "evidence.py", "embedding_service.py"):
        source = (PACKAGE / name).read_text(encoding="utf-8")
        assert "fastapi" not in source
        assert "starlette" not in source


def test_provider_contract_is_independent_from_prtreid_runtime() -> None:
    assert not (PACKAGE / "providers.py").exists()

    contract = (PACKAGE / "provider_contract.py").read_text(encoding="utf-8")
    assert "prtreid" not in contract.lower()

    for name in ("evidence.py", "embedding_service.py"):
        source = (PACKAGE / name).read_text(encoding="utf-8")
        assert "prtreid_provider" not in source
        assert "prtreid_reference" not in source

    provider = (PACKAGE / "prtreid_provider.py").read_text(encoding="utf-8")
    assert "from .provider_contract import" in provider
    assert "from .prtreid_reference import" in provider
