from __future__ import annotations

import ast
from pathlib import Path

import app.reconstruction as reconstruction
from app.reconstruction_run_repository import ReconstructionRunRepository
from app.scene_repository import SceneRepository


APP_ROOT = Path(__file__).resolve().parents[1] / "app"


def test_reconstruction_module_exposes_only_execution_entrypoint() -> None:
    assert reconstruction.__all__ == ("reconstruct_scene",)
    for superseded_facade_name in (
        "analyze_scene_frame",
        "apply_scene_pitch_calibration",
        "clear_canonical_roster_binding",
        "delete_frame_person_annotation",
        "preview_scene_pitch_calibration",
        "propose_scene_pitch_calibration",
        "queue_reconstruction",
        "set_canonical_roster_binding",
        "set_scene_ball_trajectory",
        "set_scene_pitch_side",
        "upsert_frame_person_annotation",
    ):
        assert not hasattr(reconstruction, superseded_facade_name)


def test_scene_document_domain_has_no_persistence_dependencies() -> None:
    source = (APP_ROOT / "scene_document.py").read_text(encoding="utf-8")
    imported_modules = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    )

    assert not any(name.startswith("sqlalchemy") for name in imported_modules)
    assert not {
        "database",
        "database_transaction",
        "scene_repository",
        "reconstruction_run_repository",
    }.intersection(imported_modules)


def test_scene_and_run_repositories_have_one_persistence_responsibility() -> None:
    scheduler_methods = {
        "enqueue_reconstruction",
        "claim_reconstruction_run",
        "heartbeat_reconstruction_run",
        "put_if_reconstruction_run",
        "list_recoverable_reconstruction_runs",
    }
    document_methods = {"list", "list_by_ids", "get", "put", "put_many"}

    assert scheduler_methods.isdisjoint(vars(SceneRepository))
    assert document_methods.isdisjoint(vars(ReconstructionRunRepository))
    assert not (APP_ROOT / "store.py").exists()


def test_reconstruction_queue_separates_pure_draft_from_persisted_command() -> None:
    draft_tree = ast.parse(
        (APP_ROOT / "reconstruction_queue_draft.py").read_text(encoding="utf-8")
    )
    draft_imports = {
        node.module or ""
        for node in ast.walk(draft_tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert {
        "artifact_store",
        "config",
        "reconstruction_artifact_hydration",
        "reconstruction_artifact_publication",
        "reconstruction_run_repository",
        "scene_repository",
        "uuid",
    }.isdisjoint(draft_imports)

    command_tree = ast.parse(
        (APP_ROOT / "reconstruction_queue.py").read_text(encoding="utf-8")
    )
    queue_command = next(
        node
        for node in command_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "queue_reconstruction"
    )
    assert all(argument.arg != "persist" for argument in queue_command.args.kwonlyargs)
    command_imports = {
        node.module or ""
        for node in ast.walk(command_tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert {
        "reconstruction_artifact_hydration",
        "reconstruction_artifact_publication",
        "reconstruction_queue_draft",
        "reconstruction_run_repository",
    }.issubset(command_imports)


def test_artifact_capabilities_have_no_aggregate_facade() -> None:
    assert not (APP_ROOT / "reconstruction_artifacts.py").exists()
    assert not (APP_ROOT / "reconstruction_artifact_service.py").exists()
    imports_by_module = {
        name: {
            node.module or ""
            for node in ast.walk(
                ast.parse((APP_ROOT / name).read_text(encoding="utf-8"))
            )
            if isinstance(node, ast.ImportFrom)
        }
        for name in (
            "artifact_store.py",
            "reconstruction_artifact_manifest.py",
            "reconstruction_artifact_codec.py",
        )
    }
    assert not any(
        {
            "reconstruction_artifact_hydration",
            "reconstruction_artifact_publication",
            "reconstruction_ball_artifacts",
            "reconstruction_calibration_artifacts",
            "reconstruction_identity_artifacts",
        }
        & imports
        for imports in imports_by_module.values()
    )

    publication_imports = {
        node.module or ""
        for node in ast.walk(
            ast.parse(
                (APP_ROOT / "reconstruction_artifact_publication.py").read_text(
                    encoding="utf-8"
                )
            )
        )
        if isinstance(node, ast.ImportFrom)
    }
    assert {
        "reconstruction_ball_artifacts",
        "reconstruction_calibration_artifacts",
        "reconstruction_identity_artifacts",
    }.issubset(publication_imports)
    assert "reconstruction_artifact_hydration" not in publication_imports


def test_ball_analysis_capabilities_have_one_way_dependencies() -> None:
    """Ball projection capabilities form a one-way graph outside detection."""

    assert not (APP_ROOT / "reconstruction_ball_analysis.py").exists()
    assert not (APP_ROOT / "reconstruction_ball_projection.py").exists()

    def local_imports(module_name: str) -> set[str]:
        source = (APP_ROOT / module_name).read_text(encoding="utf-8")
        return {
            node.module or ""
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
        }

    contract_imports = local_imports("reconstruction_ball_projection_contract.py")
    homography_imports = local_imports("reconstruction_bounded_homography.py")
    context_imports = local_imports(
        "reconstruction_dense_ball_projection_context.py"
    )
    candidate_imports = local_imports(
        "reconstruction_ball_candidate_projection.py"
    )
    status_imports = local_imports("reconstruction_ball_projection_status.py")
    roi_imports = local_imports("reconstruction_ball_roi.py")
    detection_imports = local_imports("reconstruction_ball_detection.py")

    detector_infrastructure = {
        "ball_detection_cache",
        "ball_detection_contract",
        "config",
        "reconstruction_ball_detection",
        "reconstruction_ball_roi",
    }
    assert detector_infrastructure.isdisjoint(contract_imports)
    assert detector_infrastructure.isdisjoint(homography_imports)
    assert detector_infrastructure.isdisjoint(context_imports)
    assert detector_infrastructure.isdisjoint(candidate_imports)
    assert detector_infrastructure.isdisjoint(status_imports)
    assert "reconstruction_ball_projection_contract" in context_imports
    assert "reconstruction_bounded_homography" in context_imports
    assert "reconstruction_ball_projection_contract" in candidate_imports
    assert "reconstruction_dense_ball_projection_context" not in candidate_imports
    assert "reconstruction_ball_detection" not in roi_imports
    assert "reconstruction_ball_roi" in detection_imports
    assert not any("projection" in name for name in detection_imports)


def test_dense_ball_detection_coordinator_uses_direct_capabilities() -> None:
    def imports_for(module_name: str) -> set[str]:
        source = (APP_ROOT / module_name).read_text(encoding="utf-8")
        return {
            node.module or ""
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
        }

    coordinator_source = (APP_ROOT / "reconstruction_ball_detection.py").read_text(
        encoding="utf-8"
    )
    coordinator_tree = ast.parse(coordinator_source)
    top_level_functions = {
        node.name
        for node in coordinator_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert top_level_functions == {"detect_ball_frames"}

    contract_source = (
        APP_ROOT / "reconstruction_ball_detection_contract.py"
    ).read_text(encoding="utf-8")
    assert "ball_detection_cache" not in contract_source
    assert "reconstruction_ball_roi" not in contract_source
    assert "config" not in contract_source

    source_imports = imports_for("reconstruction_ball_detection_source.py")
    assert "reconstruction_ball_detection_attempt" not in source_imports
    assert "ball_detection_cache" not in source_imports

    attempt_imports = imports_for("reconstruction_ball_detection_attempt.py")
    assert "ball_detection_cache" not in attempt_imports
    assert "reconstruction_ball_detection_source" not in attempt_imports


def test_ball_configuration_editing_and_persistence_have_distinct_owners() -> None:
    assert not (APP_ROOT / "reconstruction_ball.py").exists()

    def local_imports(module_name: str) -> set[str]:
        source = (APP_ROOT / module_name).read_text(encoding="utf-8")
        return {
            node.module or ""
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
        }

    configuration_imports = local_imports("ball_detection_configuration.py")
    selection_imports = local_imports(
        "reconstruction_ball_detector_selection.py"
    )
    trajectory_imports = local_imports("reconstruction_ball_trajectory.py")
    command_imports = local_imports(
        "reconstruction_ball_trajectory_command.py"
    )

    assert configuration_imports.isdisjoint(
        {
            "ball_detector_factory",
            "reconstruction_artifact_hydration",
            "reconstruction_artifact_publication",
            "reconstruction_ball_artifacts",
            "scene_repository",
        }
    )
    assert selection_imports.isdisjoint(
        {
            "reconstruction_ball_trajectory",
            "reconstruction_ball_trajectory_command",
            "scene_repository",
        }
    )
    assert trajectory_imports.isdisjoint(
        {
            "ball_detection_configuration",
            "ball_detector_factory",
            "config",
            "reconstruction_artifact_hydration",
            "reconstruction_artifact_publication",
            "reconstruction_ball_artifacts",
            "scene_repository",
        }
    )
    assert "reconstruction_ball_trajectory" in command_imports
    assert "reconstruction_artifact_hydration" in command_imports
    assert "reconstruction_ball_artifacts" in command_imports
    assert "reconstruction_artifact_publication" not in command_imports
    assert "scene_repository" in command_imports

    command_tree = ast.parse(
        (APP_ROOT / "reconstruction_ball_trajectory_command.py").read_text(
            encoding="utf-8"
        )
    )
    command_functions = {
        node.name
        for node in command_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert command_functions == {"set_scene_ball_trajectory"}


def _top_level_function_lengths(module_name: str) -> dict[str, int]:
    tree = ast.parse((APP_ROOT / module_name).read_text(encoding="utf-8"))
    return {
        node.name: node.end_lineno - node.lineno + 1
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_method_lengths(module_name: str, class_name: str) -> dict[str, int]:
    tree = ast.parse((APP_ROOT / module_name).read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name: node.end_lineno - node.lineno + 1
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_reconstruction_run_transactions_have_pure_planners_and_one_owner() -> None:
    def local_imports(module_name: str) -> set[str]:
        source = (APP_ROOT / module_name).read_text(encoding="utf-8")
        return {
            node.module or ""
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
        }

    repository_imports = local_imports("reconstruction_run_repository.py")
    assert {
        "reconstruction_run_contract",
        "reconstruction_run_queries",
        "reconstruction_run_scene_transition",
    }.issubset(repository_imports)

    infrastructure = {
        "analysis_run_telemetry",
        "database",
        "database_transaction",
        "project_resource_repository",
        "reconstruction_run_queries",
        "reconstruction_run_repository",
    }
    assert local_imports("reconstruction_run_contract.py").isdisjoint(
        infrastructure
    )
    assert local_imports(
        "reconstruction_run_scene_transition.py"
    ).isdisjoint(infrastructure)

    query_source = (APP_ROOT / "reconstruction_run_queries.py").read_text(
        encoding="utf-8"
    )
    assert "SessionLocal" not in query_source
    assert "begin_write_transaction" not in query_source
    query_consumers = {
        path.name
        for path in APP_ROOT.glob("*.py")
        if path.name != "reconstruction_run_queries.py"
        and "reconstruction_run_queries" in local_imports(path.name)
    }
    assert query_consumers == {"reconstruction_run_repository.py"}

    lengths = _class_method_lengths(
        "reconstruction_run_repository.py",
        "ReconstructionRunRepository",
    )
    assert lengths["enqueue_reconstruction"] <= 110
    assert lengths["put_if_reconstruction_run"] <= 110
    assert lengths["claim_reconstruction_run"] <= 150
    assert lengths["publish_reconstruction_progress"] <= 80


def test_identity_capabilities_have_no_god_facade() -> None:
    for removed_facade in (
        "identity_review.py",
        "reconstruction_identity_association.py",
        "reconstruction_identity_graph.py",
        "reconstruction_identity_evidence.py",
        "reconstruction_identity_roster.py",
        "reconstruction_identity_undo.py",
    ):
        assert not (APP_ROOT / removed_facade).exists()

    semantics_source = (
        APP_ROOT / "reconstruction_identity_semantics.py"
    ).read_text(encoding="utf-8")
    semantics_imports = {
        node.module or ""
        for node in ast.walk(ast.parse(semantics_source))
        if isinstance(node, ast.ImportFrom)
    }
    assert not {
        "reconstruction_identity_validation",
        "reconstruction_identity_reference_cleanup",
        "reconstruction_identity_roster_commands",
        "reconstruction_identity_roster_undo_planning",
        "scene_repository",
    }.intersection(semantics_imports)


def test_identity_review_query_media_and_http_presentation_are_separate() -> None:
    def local_imports(module_name: str) -> set[str]:
        source = (APP_ROOT / module_name).read_text(encoding="utf-8")
        return {
            node.module or ""
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
        }

    contract_imports = local_imports("identity_review_contract.py")
    projection_imports = local_imports("identity_review_projection.py")
    observation_imports = local_imports(
        "identity_review_observation_projection.py"
    )
    person_imports = local_imports("identity_review_person_projection.py")
    crop_imports = local_imports("identity_review_crop_service.py")
    presenter_imports = local_imports("identity_review_http_presenter.py")

    assert contract_imports.isdisjoint(
        {
            "artifact_store",
            "config",
            "identity_review_crop_service",
            "identity_review_projection",
            "scene_repository",
        }
    )
    assert projection_imports.isdisjoint(
        {
            "config",
            "identity_review_crop_service",
            "identity_review_http_presenter",
            "video_media_paths",
        }
    )
    pure_projection_forbidden = {
        "artifact_store",
        "config",
        "identity_review_crop_service",
        "identity_review_http_presenter",
        "reconstruction_artifact_hydration",
        "reconstruction_artifact_publication",
        "reconstruction_identity_artifacts",
        "scene_repository",
        "video_media_paths",
    }
    assert observation_imports.isdisjoint(pure_projection_forbidden)
    assert person_imports.isdisjoint(pure_projection_forbidden)
    assert crop_imports.isdisjoint(
        {
            "identity_decisions",
            "identity_review_http_presenter",
            "identity_review_projection",
            "project_contract_base",
            "project_identity_contract",
            "project_lifecycle_contract",
            "project_match_persistence_contract",
            "project_segment_contract",
        }
    )
    assert presenter_imports.isdisjoint(
        {
            "artifact_store",
            "config",
            "identity_review_crop_service",
            "identity_review_projection",
            "reconstruction_artifact_hydration",
            "reconstruction_artifact_publication",
            "reconstruction_identity_artifacts",
        }
    )

    routes = (APP_ROOT / "identity_review_routes.py").read_text(encoding="utf-8")
    assert "response_model=IdentityReviewResponse" in routes


def test_roster_binding_collaborators_are_pure_of_commands_and_persistence() -> None:
    collaborators = (
        "reconstruction_identity_match_roster.py",
        "reconstruction_identity_roster_binding_planning.py",
        "reconstruction_identity_roster_baseline.py",
        "reconstruction_identity_roster_clear_planning.py",
        "reconstruction_identity_roster_lineage.py",
        "reconstruction_identity_roster_observations.py",
        "reconstruction_identity_roster_ownership.py",
        "reconstruction_identity_reference_cleanup.py",
        "reconstruction_identity_roster_undo_planning.py",
    )
    forbidden = {
        "reconstruction_artifact_hydration",
        "reconstruction_artifact_publication",
        "reconstruction_identity_artifacts",
        "reconstruction_identity_annotation_commit",
        "reconstruction_identity_roster_draft",
        "reconstruction_identity_roster_commands",
        "reconstruction_queue",
        "scene_repository",
    }
    for module_name in collaborators:
        source = (APP_ROOT / module_name).read_text(encoding="utf-8")
        imports = {
            node.module or ""
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
        }
        assert forbidden.isdisjoint(imports), module_name


def test_roster_binding_has_explicit_draft_and_always_persist_command() -> None:
    command_tree = ast.parse(
        (APP_ROOT / "reconstruction_identity_roster_commands.py").read_text(
            encoding="utf-8"
        )
    )
    commands = {
        node.name: node
        for node in command_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert set(commands) == {
        "clear_canonical_roster_binding",
        "set_canonical_roster_binding",
    }
    assert all(
        argument.arg != "persist"
        for command in commands.values()
        for argument in (*command.args.args, *command.args.kwonlyargs)
    )
    command_imports = {
        node.module or ""
        for node in ast.walk(command_tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert {
        "reconstruction_identity_annotation_commit",
        "reconstruction_identity_roster_draft",
    }.issubset(command_imports)

    route_imports = {
        node.module or ""
        for node in ast.walk(
            ast.parse(
                (APP_ROOT / "scene_identity_routes.py").read_text(encoding="utf-8")
            )
        )
        if isinstance(node, ast.ImportFrom)
    }
    assert "reconstruction_identity_roster_draft" in route_imports
    assert "reconstruction_identity_roster_commands" not in route_imports


def test_identity_annotation_capabilities_have_no_aggregate_or_dual_write_path() -> None:
    assert not (APP_ROOT / "reconstruction_identity_annotations.py").exists()
    assert not (APP_ROOT / "reconstruction_identity_annotation_planning.py").exists()
    pure_modules = (
        "reconstruction_identity_annotation_upsert_planning.py",
        "reconstruction_identity_annotation_undo_planning.py",
    )
    forbidden = {
        "cv2",
        "reconstruction_artifact_hydration",
        "reconstruction_artifact_publication",
        "reconstruction_identity_artifacts",
        "reconstruction_frame_annotation_target",
        "reconstruction_identity_annotation_commit",
        "scene_repository",
    }
    for module_name in pure_modules:
        tree = ast.parse((APP_ROOT / module_name).read_text(encoding="utf-8"))
        imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert forbidden.isdisjoint(imports), module_name

    for module_name, function_name in (
        (
            "reconstruction_identity_annotation_upsert_command.py",
            "upsert_frame_person_annotation",
        ),
        (
            "reconstruction_identity_annotation_delete_command.py",
            "delete_frame_person_annotation",
        ),
    ):
        tree = ast.parse((APP_ROOT / module_name).read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        assert all(argument.arg != "persist" for argument in function.args.kwonlyargs)
        imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert "reconstruction_identity_annotation_commit" in imports
        assert "reconstruction_identity_annotation_draft" in imports

    command_source = (
        APP_ROOT / "reconstruction_identity_roster_commands.py"
    ).read_text(encoding="utf-8")
    command_imports = {
        node.module or ""
        for node in ast.walk(ast.parse(command_source))
        if isinstance(node, ast.ImportFrom)
    }
    assert not {
        "fastapi",
        "project_match_repository",
        "reconstruction_queue",
        "scene_identity_routes",
    }.intersection(command_imports)

    assert _top_level_function_lengths(
        "reconstruction_identity_roster_commands.py"
    )["set_canonical_roster_binding"] <= 150


def test_frame_and_identity_entrypoints_stay_orchestrators() -> None:
    assert _top_level_function_lengths("reconstruction_detection_phase.py")[
        "detect_and_calibrate_phase"
    ] <= 110
    detection_phase_source = (
        APP_ROOT / "reconstruction_detection_phase.py"
    ).read_text(encoding="utf-8")
    detection_phase_tree = ast.parse(detection_phase_source)
    detection_phase_functions = {
        node.name
        for node in detection_phase_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert detection_phase_functions == {"detect_and_calibrate_phase"}
    for consumer in (
        "reconstruction_publish_payloads.py",
        "reconstruction_publish_phase.py",
    ):
        imports = {
            node.module or ""
            for node in ast.walk(
                ast.parse((APP_ROOT / consumer).read_text(encoding="utf-8"))
            )
            if isinstance(node, ast.ImportFrom)
        }
        assert "reconstruction_detection_contract" in imports
        assert "reconstruction_detection_phase" not in imports
    dense_ball_imports = {
        node.module or ""
        for node in ast.walk(
            ast.parse(
                (APP_ROOT / "reconstruction_dense_ball_phase.py").read_text(
                    encoding="utf-8"
                )
            )
        )
        if isinstance(node, ast.ImportFrom)
    }
    assert "config" not in dense_ball_imports
    assert "reconstruction_detection_phase" not in dense_ball_imports
    assert _top_level_function_lengths("reconstruction_frame_analysis.py")[
        "analyze_scene_frame"
    ] <= 150
    assert not (APP_ROOT / "reconstruction_identity_documents.py").exists()
    assert set(
        _top_level_function_lengths(
            "reconstruction_canonical_people_projection.py"
        )
    ) == {"canonical_people_documents"}
    assert max(
        _top_level_function_lengths(
            "reconstruction_identity_document_projection.py"
        ).values()
    ) <= 120
    assert "reconstruction_roster_identity_resolution" not in {
        node.module or ""
        for node in ast.walk(
            ast.parse(
                (
                    APP_ROOT / "reconstruction_identity_document_projection.py"
                ).read_text(encoding="utf-8")
            )
        )
        if isinstance(node, ast.ImportFrom)
    }
