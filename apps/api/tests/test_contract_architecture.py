from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.match_contracts import EventBundle, ExternalEvent, ExternalTeam
from app.frame_annotation_contracts import FrameAnalysisRequest
from app.scene_contracts import SceneDocument


APP = Path(__file__).parents[1] / "app"
CAPABILITY_CONTRACTS = (
    "ball_contracts.py",
    "calibration_contracts.py",
    "frame_annotation_contracts.py",
    "match_contracts.py",
    "player_action_contracts.py",
    "project_contract_base.py",
    "project_identity_contract.py",
    "project_http_contracts.py",
    "project_lifecycle_contract.py",
    "project_match_persistence_contract.py",
    "project_segment_contract.py",
    "analysis_run_contract.py",
    "reconstruction_contracts.py",
    "scene_contracts.py",
)


def _imports(filename: str) -> set[str]:
    tree = ast.parse((APP / filename).read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
        elif isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
    return result


def test_generic_schema_barrels_do_not_exist() -> None:
    assert not (APP / "schemas.py").exists()
    assert not (APP / "project_schemas.py").exists()


def test_http_contracts_do_not_depend_on_runtime_layers() -> None:
    forbidden = {
        "database",
        "fastapi",
        "project_store",
        "project_resource_repository",
        "store",
    }
    for filename in CAPABILITY_CONTRACTS:
        assert _imports(filename).isdisjoint(forbidden), filename


def test_match_bundle_requires_explicit_provider_source() -> None:
    with pytest.raises(ValidationError):
        EventBundle(
            event=ExternalEvent(
                id="match-1",
                name="Home vs Away",
                home=ExternalTeam(id="home", name="Home"),
                away=ExternalTeam(id="away", name="Away"),
            ),
            fetched_at="2026-07-18T00:00:00Z",
        )


def test_scene_contracts_do_not_assume_a_global_two_minute_limit() -> None:
    scene = SceneDocument(
        id="scene-long",
        title="Long review",
        duration=180.0,
        payload={},
    )
    frame = FrameAnalysisRequest(scene_time=179.5)

    assert scene.duration == 180.0
    assert frame.scene_time == 179.5


def test_capability_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        FrameAnalysisRequest.model_validate(
            {"scene_time": 1.0, "scene_tmie": 1.0}
        )
