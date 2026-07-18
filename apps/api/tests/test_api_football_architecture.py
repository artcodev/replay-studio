from __future__ import annotations

import ast
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1] / "app"
PROVIDERS_ROOT = APP_ROOT / "providers"


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_api_football_mapper_is_a_pure_contract_translation_boundary() -> None:
    mapper = PROVIDERS_ROOT / "api_football_mapping.py"

    assert _import_roots(mapper).isdisjoint(
        {
            "config",
            "database",
            "httpx",
            "persistence",
            "redis",
            "repositories",
            "sqlalchemy",
            "store",
        }
    )
    source = mapper.read_text(encoding="utf-8")
    assert "ApiFootballClient" not in source
    assert "Settings" not in source


def test_registry_depends_on_provider_owner_not_deleted_aggregate_adapter() -> None:
    source = (PROVIDERS_ROOT / "registry.py").read_text(encoding="utf-8")

    assert "from .api_football_provider import ApiFootballProvider" in source
    assert "from .api_football import" not in source
    assert not (PROVIDERS_ROOT / "api_football.py").exists()


def test_provider_delegates_transport_and_mapping_instead_of_owning_them() -> None:
    source = (PROVIDERS_ROOT / "api_football_provider.py").read_text(
        encoding="utf-8"
    )

    assert "httpx" not in source
    assert "redis" not in source.lower()
    assert "ApiFootballClient" in source
    assert "map_fixture" in source
    assert "map_lineups" in source
    assert "map_timeline" in source
