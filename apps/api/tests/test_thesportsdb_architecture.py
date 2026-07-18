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


def test_thesportsdb_mapper_is_a_pure_contract_translation_boundary() -> None:
    mapper = PROVIDERS_ROOT / "thesportsdb_mapping.py"

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
    assert "TheSportsDbClient" not in source
    assert "Settings" not in source


def test_registry_imports_owner_and_has_no_contextual_provider_override() -> None:
    source = (PROVIDERS_ROOT / "registry.py").read_text(encoding="utf-8")

    assert "from .thesportsdb_provider import TheSportsDbProvider" in source
    assert "from .thesportsdb import" not in source
    assert "ContextVar" not in source
    assert "_override" not in source
    assert not (PROVIDERS_ROOT / "thesportsdb.py").exists()


def test_provider_delegates_transport_and_mapping_instead_of_owning_them() -> None:
    source = (PROVIDERS_ROOT / "thesportsdb_provider.py").read_text(
        encoding="utf-8"
    )

    assert "httpx" not in source
    assert "redis" not in source.lower()
    assert "TheSportsDbClient" in source
    assert "map_event" in source
    assert "map_lineup" in source
    assert "map_timeline" in source


def test_provider_neutral_match_normalization_does_not_branch_on_source() -> None:
    source = (APP_ROOT / "manual_match_import.py").read_text(encoding="utf-8")

    assert 'bundle.source == "thesportsdb"' not in source
