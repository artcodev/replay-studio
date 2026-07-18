from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

from app.quality_gate_report import assess_quality_gates
from app.quality_measurement_domain import ReconstructionQualityMeasurements
from app.quality_measurements import collect_quality_measurements
from app.quality_metric_report import build_quality_metrics
from app.quality_policy import DEFAULT_THRESHOLDS


APP = Path(__file__).parents[1] / "app"
ARTIFACT_CAPABILITIES = {
    "reconstruction_artifact_hydration",
    "reconstruction_artifact_publication",
    "reconstruction_ball_artifacts",
    "reconstruction_calibration_artifacts",
    "reconstruction_identity_artifacts",
}


def _imports(module: str) -> set[str]:
    tree = ast.parse((APP / module).read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
        elif isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
    return result


def _module_functions(module: str) -> set[str]:
    tree = ast.parse((APP / module).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_quality_use_case_does_not_reabsorb_measurement_or_policy_logic() -> None:
    assert _module_functions("quality_metrics.py") == {
        "evaluate_reconstruction_quality"
    }
    assert _imports("quality_measurements.py").isdisjoint(ARTIFACT_CAPABILITIES)
    assert _module_functions("quality_measurements.py") == {
        "collect_quality_measurements"
    }
    assert _imports("quality_measurements.py").isdisjoint(
        {"quality_gate_report", "quality_metric_report", "store", "project_store"}
    )
    assert _imports("quality_metric_report.py").isdisjoint(
        {"quality_measurements", "quality_gate_report", *ARTIFACT_CAPABILITIES}
    )
    assert _imports("quality_gate_report.py").isdisjoint(
        {"quality_measurements", "quality_metric_report", *ARTIFACT_CAPABILITIES}
    )
    for module in (
        "quality_ball_measurements.py",
        "quality_calibration_measurements.py",
        "quality_identity_measurements.py",
        "quality_motion_measurements.py",
        "quality_projection_measurements.py",
    ):
        assert _imports(module).isdisjoint(
            ARTIFACT_CAPABILITIES
            | {
                "quality_gate_report",
                "quality_metric_report",
                "quality_metrics",
                "store",
                "project_store",
            }
        ), module


def test_benchmark_capabilities_have_directional_dependencies() -> None:
    assert not (APP / "quality_benchmark.py").exists()
    app_modules = {path.stem for path in APP.glob("*.py")}
    evaluator_modules = {
        "quality_benchmark_ball",
        "quality_benchmark_calibration",
        "quality_benchmark_people",
    }
    for module in evaluator_modules:
        assert _imports(f"{module}.py").isdisjoint(evaluator_modules - {module}), module

    validation_imports = _imports("quality_benchmark_validation.py")
    assert validation_imports.isdisjoint(
        evaluator_modules
        | ARTIFACT_CAPABILITIES
        | {
            "quality_benchmark_context",
            "quality_benchmark_report",
            "quality_metrics",
            "project_store",
            "store",
            "reconstruction_run_repository",
    }
    )
    assert _imports("quality_benchmark_context.py").isdisjoint(
        evaluator_modules | ARTIFACT_CAPABILITIES
    )
    assert _imports("quality_benchmark_statistics.py").isdisjoint(
        {
            "quality_benchmark_contract",
            "quality_benchmark_validation",
            "quality_benchmark_context",
            *evaluator_modules,
        }
    )
    local_report_imports = {
        imported
        for imported in _imports("quality_benchmark_report.py")
        if imported.startswith("quality_benchmark")
    }
    assert local_report_imports == {
        "quality_benchmark_ball",
        "quality_benchmark_calibration",
        "quality_benchmark_context",
        "quality_benchmark_contract",
        "quality_benchmark_people",
        "quality_benchmark_validation",
    }
    expected_local_dependencies = {
        "quality_benchmark_contract.py": set(),
        "quality_benchmark_statistics.py": set(),
        "quality_benchmark_validation.py": {
            "quality_benchmark_contract",
            "quality_benchmark_statistics",
        },
        "quality_benchmark_context.py": {
            "quality_benchmark_contract",
            "quality_benchmark_statistics",
            "quality_benchmark_validation",
        },
        "quality_benchmark_people.py": {
            "identity_metrics",
            "quality_benchmark_context",
            "quality_benchmark_statistics",
        },
        "quality_benchmark_calibration.py": {
            "quality_benchmark_context",
            "quality_benchmark_statistics",
        },
        "quality_benchmark_ball.py": {
            "quality_benchmark_context",
            "quality_benchmark_statistics",
        },
        "quality_benchmark_report.py": local_report_imports,
    }
    for module, expected in expected_local_dependencies.items():
        assert _imports(module) & app_modules == expected, module
    for module in (
        "quality_benchmark_contract.py",
        "quality_benchmark_statistics.py",
        "quality_benchmark_validation.py",
        "quality_benchmark_context.py",
        "quality_benchmark_people.py",
        "quality_benchmark_calibration.py",
        "quality_benchmark_ball.py",
        "quality_benchmark_report.py",
    ):
        assert _imports(module).isdisjoint({"argparse", "json", "pathlib"}), module
    assert _module_functions("quality_benchmark_report.py") == {
        "_evaluate_contexts",
        "evaluate_benchmark",
    }
    assert _module_functions("quality_benchmark_cli.py") == {"_load_json", "main"}


def test_capability_contracts_replace_the_common_schema_barrel() -> None:
    assert not (APP / "schemas.py").exists()
    for path in APP.glob("*.py"):
        assert "schemas" not in _imports(path.name), path.name
    expected_route_contracts = {
        "scene_document_routes.py": {"scene_contracts"},
        "scene_calibration_routes.py": {"calibration_contracts", "scene_contracts"},
        "scene_identity_routes.py": {"frame_annotation_contracts", "scene_contracts"},
        "scene_analysis_routes.py": {
            "ball_contracts",
            "player_action_contracts",
            "reconstruction_contracts",
            "scene_contracts",
        },
        "match_import_routes.py": {"match_contracts"},
    }
    for module, contracts in expected_route_contracts.items():
        assert contracts <= _imports(module), module


def test_typed_measurements_are_shared_by_metric_and_gate_layers() -> None:
    scene = {
        "duration": 0.4,
        "payload": {
            "pitch": {"length": 105, "width": 68},
            "tracks": [
                {
                    "id": "home-1",
                    "keyframes": [
                        {"t": 0.0, "x": 0.0, "z": 0.0, "confidence": 0.9},
                        {"t": 0.2, "x": 0.5, "z": 0.0, "confidence": 0.9},
                    ],
                }
            ],
            "ball": {"keyframes": []},
            "videoAsset": {
                "reconstruction": {
                    "status": "ready",
                    "frameCount": 2,
                    "calibration": {
                        "frameEvidence": [
                            {
                                "sceneTime": 0.0,
                                "status": "accepted",
                                "projectionSource": "direct",
                                "reprojectionError": 2.0,
                                "inlierRatio": 0.9,
                                "visiblePitchSide": "left",
                                "alignmentMetrics": {"f1": 0.4},
                            },
                            {
                                "sceneTime": 0.2,
                                "status": "accepted",
                                "projectionSource": "direct",
                                "reprojectionError": 2.5,
                                "inlierRatio": 0.8,
                                "visiblePitchSide": "left",
                                "alignmentMetrics": {"f1": 0.35},
                            },
                        ]
                    },
                }
            },
        },
    }
    before = deepcopy(scene)

    measurements = collect_quality_measurements(
        scene,
        None,
        thresholds=DEFAULT_THRESHOLDS,
    )
    metrics = build_quality_metrics(measurements, DEFAULT_THRESHOLDS)
    assessment = assess_quality_gates(measurements, DEFAULT_THRESHOLDS)

    assert isinstance(measurements, ReconstructionQualityMeasurements)
    assert measurements.calibration.coverage == 1.0
    assert measurements.motion.player_speed.ratio == 0.0
    assert metrics["calibrationCoverage"]["value"] == 1.0
    assert {gate["id"] for gate in assessment.gates} >= {
        "calibration-coverage",
        "player-speed",
    }
    assert scene == before


def test_identity_measurement_reads_only_the_canonical_validation_contract() -> None:
    scene = {
        "payload": {
            "groundTruth": {
                "identityAssignments": [
                    {
                        "frameIndex": 0,
                        "groundTruthId": "player-1",
                        "predictedId": "player-1",
                    }
                ]
            },
            "videoAsset": {"reconstruction": {}},
        }
    }

    measurements = collect_quality_measurements(
        scene,
        None,
        thresholds=DEFAULT_THRESHOLDS,
    )

    assert measurements.identity.validation["groundTruthAvailable"] is False
