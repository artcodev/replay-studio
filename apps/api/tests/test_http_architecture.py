from __future__ import annotations

import ast
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "app"
ROUTE_MODULES = (
    "health_routes.py",
    "match_import_routes.py",
    "video_routes.py",
    "scene_document_routes.py",
    "scene_analysis_routes.py",
    "scene_identity_routes.py",
    "scene_calibration_routes.py",
    "project_routes.py",
    "project_core_routes.py",
    "project_match_routes.py",
    "project_identity_routes.py",
    "project_media_routes.py",
    "project_analysis_routes.py",
    "identity_review_routes.py",
    "identity_decision_routes.py",
)


def _tree(filename: str) -> ast.Module:
    return ast.parse((APP_DIR / filename).read_text(encoding="utf-8"))


def test_main_is_only_the_http_composition_root() -> None:
    tree = _tree("main.py")
    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert functions == {
        "lifespan",
        "scene_revision_conflict_handler",
        "create_app",
    }
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr in {"get", "post", "put", "delete", "patch"}
            for decorator in node.decorator_list
        )
        for node in tree.body
    )


def test_route_modules_never_depend_on_the_composition_root() -> None:
    for filename in ROUTE_MODULES:
        imports = [
            node
            for node in ast.walk(_tree(filename))
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not any(
            isinstance(node, ast.ImportFrom) and node.module in {"main", "app.main"}
            for node in imports
        ), filename
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name == "app.main" for alias in node.names)
            for node in imports
        ), filename


def test_project_router_is_only_a_capability_aggregator() -> None:
    tree = _tree("project_routes.py")
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in tree.body
    )


def test_manual_match_normalizer_is_transport_and_persistence_agnostic() -> None:
    imports = {
        node.module
        for node in ast.walk(_tree("manual_match_import.py"))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "fastapi" not in imports
    assert "database" not in imports
    assert "project_store" not in imports


def test_composed_http_routes_have_one_owner_per_method_and_path() -> None:
    from app.main import app

    owners: dict[tuple[str, str], str] = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        for method in getattr(route, "methods", None) or ():
            if path is None or method in {"HEAD", "OPTIONS"}:
                continue
            key = (method, path)
            assert key not in owners, (
                f"{method} {path} is owned by both {owners[key]} and {route.name}"
            )
            owners[key] = route.name
