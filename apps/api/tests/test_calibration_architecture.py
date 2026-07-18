from __future__ import annotations

import ast
from pathlib import Path


APP = Path(__file__).parents[1] / "app"


def _imports(module: str) -> set[str]:
    tree = ast.parse((APP / module).read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
        elif isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
    return result


def _functions(module: str) -> set[str]:
    tree = ast.parse((APP / module).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_calibration_capabilities_have_no_aggregate_facade() -> None:
    assert not (APP / "reconstruction_calibration.py").exists()
    assert not (APP / "reconstruction_calibration_commands.py").exists()
    assert not (APP / "reconstruction_calibration_quality.py").exists()

    route_imports = _imports("scene_calibration_routes.py")
    assert {
        "reconstruction_calibration_apply",
        "reconstruction_calibration_manual_preview",
        "reconstruction_calibration_proposal",
        "reconstruction_pitch_side_command",
    }.issubset(route_imports)


def test_calibration_drafts_and_manual_previews_do_not_persist_scenes() -> None:
    for module in (
        "reconstruction_calibration_draft.py",
        "reconstruction_calibration_manual_preview.py",
        "reconstruction_calibration_overrides.py",
    ):
        assert _imports(module).isdisjoint(
            {
                "scene_repository",
                "reconstruction_queue",
                "reconstruction_calibration_preview",
            }
        ), module

    # Proposal orchestration records diagnostics through the dedicated preview
    # boundary; it does not reach into scene persistence itself.
    proposal_imports = _imports("reconstruction_calibration_proposal.py")
    assert "scene_repository" not in proposal_imports
    assert "reconstruction_calibration_preview" in proposal_imports


def test_calibration_apply_is_the_manual_override_rebuild_boundary() -> None:
    apply_imports = _imports("reconstruction_calibration_apply.py")
    assert {"scene_repository", "reconstruction_queue"}.issubset(apply_imports)
    assert "apply_scene_pitch_calibration" in _functions(
        "reconstruction_calibration_apply.py"
    )


def test_pitch_side_command_is_independent_of_preview_and_workers() -> None:
    imports = _imports("reconstruction_pitch_side_command.py")
    assert "scene_repository" in imports
    assert imports.isdisjoint(
        {
            "calibration_worker",
            "reconstruction_calibration_apply",
            "reconstruction_calibration_preview",
            "reconstruction_calibration_proposal",
            "reconstruction_queue",
        }
    )


def test_calibration_evidence_and_quality_are_pure_of_runtime_adapters() -> None:
    runtime_adapters = {
        "calibration_worker",
        "config",
        "database",
        "reconstruction_inputs",
        "scene_repository",
        "reconstruction_calibration_preview",
    }
    for module in (
        "reconstruction_calibration_evidence.py",
        "reconstruction_calibration_policy.py",
        "reconstruction_frame_calibration_quality.py",
        "reconstruction_shot_calibration_quality.py",
        "reconstruction_metric_projection.py",
    ):
        assert _imports(module).isdisjoint(runtime_adapters), module


def test_calibration_detection_and_temporal_resolution_are_independent() -> None:
    assert _imports("reconstruction_calibration_detection.py").isdisjoint(
        {
            "reconstruction_calibration_evidence",
            "reconstruction_frame_calibration_quality",
            "reconstruction_shot_calibration_quality",
            "reconstruction_calibration_resolution",
            "scene_repository",
            "temporal_calibration",
            "temporal_calibration_contract",
            "temporal_calibration_solver",
            "temporal_homography",
        }
    )
    assert _imports("reconstruction_calibration_resolution.py").isdisjoint(
        {
            "calibration_worker",
            "config",
            "reconstruction_calibration_detection",
            "scene_repository",
        }
    )


def test_camera_motion_producer_and_solver_share_only_the_contract() -> None:
    producer_imports = _imports("reconstruction_motion.py")
    assert "camera_motion_contract" in producer_imports
    assert not (APP / "temporal_calibration.py").exists()
    assert "camera_motion_contract" in _imports("temporal_homography.py")
    assert "camera_motion_contract" in _imports(
        "temporal_calibration_hypothesis.py"
    )
    assert "camera_motion_contract" in _imports("temporal_calibration_solver.py")
    assert not any(name.startswith("temporal_") for name in producer_imports)
    assert _imports("temporal_calibration_contract.py").isdisjoint(
        {
            "camera_motion_contract",
            "temporal_calibration_consensus",
            "temporal_calibration_hypothesis",
            "temporal_calibration_solver",
            "temporal_homography",
        }
    )


def test_preview_persistence_is_an_explicit_single_purpose_boundary() -> None:
    assert "scene_repository" in _imports("reconstruction_calibration_preview.py")
    assert _functions("reconstruction_calibration_preview.py") == {
        "persist_frame_calibration_preview"
    }


def test_pitch_calibration_has_capability_owners_instead_of_an_aggregate() -> None:
    assert not (APP / "pitch_calibration.py").exists()
    assert _functions("pitch_anchor_calibration.py") == {
        "calibration_from_anchors"
    }
    assert "calibrate_pitch" in _functions("pitch_line_calibration.py")
    assert _functions("pitch_calibration_visualization.py") == {
        "calibration_overlay"
    }


def test_pitch_contract_geometry_and_quality_do_not_own_runtime_adapters() -> None:
    forbidden = {
        "cv2",
        "calibration_worker",
        "database",
        "reconstruction_inputs",
        "scene_repository",
    }
    for module in (
        "pitch_calibration_contract.py",
        "pitch_calibration_orientation.py",
        "pitch_geometry.py",
        "pitch_calibration_quality.py",
    ):
        assert _imports(module).isdisjoint(forbidden), module


def test_pitch_visualization_does_not_own_orchestrate_fitting() -> None:
    assert _imports("pitch_calibration_visualization.py").isdisjoint(
        {
            "pitch_anchor_calibration",
            "pitch_image_evidence",
            "pitch_line_calibration",
        }
    )
    assert _imports("pitch_calibration_quality.py").isdisjoint(
        {"pitch_anchor_calibration", "pitch_line_calibration"}
    )
