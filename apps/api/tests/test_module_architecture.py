from __future__ import annotations

import ast
from pathlib import Path


APP = Path(__file__).parents[1] / "app"


def _local_dependencies(module: Path, module_names: set[str]) -> set[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:
                candidate = node.module.split(".", 1)[0]
            elif node.level == 0 and node.module and node.module.startswith("app."):
                candidate = node.module.removeprefix("app.").split(".", 1)[0]
            else:
                continue
            if candidate in module_names:
                dependencies.add(candidate)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("app."):
                    continue
                candidate = alias.name.removeprefix("app.").split(".", 1)[0]
                if candidate in module_names:
                    dependencies.add(candidate)
    return dependencies


def _imports(module: str) -> set[str]:
    tree = ast.parse((APP / module).read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
        elif isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
    return result


def _imported_names(module: str, source_module: str) -> set[str]:
    tree = ast.parse((APP / module).read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == source_module
        for alias in node.names
    }


def test_multi_pass_has_no_aggregate_service_locator() -> None:
    assert not (APP / "multi_pass.py").exists()
    assert "multi_pass_job" in _imports("pipeline_job.py")
    assert "multi_pass_composition" in _imports("project_media_routes.py")


def test_application_module_import_graph_is_acyclic() -> None:
    modules = {path.stem: path for path in APP.glob("*.py") if path.stem != "__init__"}
    graph = {
        name: _local_dependencies(path, set(modules))
        for name, path in modules.items()
    }
    visited: set[str] = set()
    active: list[str] = []

    def visit(name: str) -> None:
        if name in active:
            cycle = active[active.index(name):] + [name]
            raise AssertionError("local import cycle: " + " -> ".join(cycle))
        if name in visited:
            return
        active.append(name)
        for dependency in sorted(graph[name]):
            visit(dependency)
        active.pop()
        visited.add(name)

    for module_name in sorted(graph):
        visit(module_name)


def test_multi_pass_algorithms_do_not_depend_on_persistence() -> None:
    persistence_modules = {
        "store",
        "project_store",
        "project_match_repository",
        "pipeline_store",
        "model_comparison_pipeline_service",
        "multi_pass_pipeline_service",
    }
    for module in (
        "multi_pass_alignment.py",
        "multi_pass_fusion.py",
        "multi_pass_metrics.py",
        "multi_pass_progress.py",
    ):
        assert _imports(module).isdisjoint(persistence_modules), module


def test_multi_pass_job_does_not_own_alignment_or_fusion_algorithms() -> None:
    tree = ast.parse((APP / "multi_pass_job.py").read_text(encoding="utf-8"))
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert functions == {
        "_failed",
        "_prepare",
        "_collect_terminal_results",
        "_poll_dependencies",
        "advance_multi_pass_pipeline_job",
    }


def test_quality_policy_is_independent_from_scene_and_persistence() -> None:
    imports = _imports("quality_policy.py")
    assert imports.isdisjoint(
        {
            "store",
            "project_store",
            "reconstruction_artifact_hydration",
            "reconstruction_artifact_publication",
            "reconstruction_identity_artifacts",
            "quality_metrics",
        }
    )
    cli_tree = ast.parse((APP / "quality_cli.py").read_text(encoding="utf-8"))
    cli_functions = {
        node.name for node in cli_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert cli_functions == {"main"}


def test_runtime_services_use_the_project_resource_repository_boundary() -> None:
    resource_rows = {"ProjectSceneRow", "ProjectVideoAssetRow", "SegmentRow"}
    for module in (
        "analysis_run_repository.py",
        "project_identity_repository.py",
        "model_comparison_pipeline_service.py",
        "multi_pass_pipeline_service.py",
        "video_pipeline.py",
    ):
        assert _imported_names(module, "project_models").isdisjoint(resource_rows), module


def test_scene_pipeline_capabilities_have_no_aggregate_service() -> None:
    assert not (APP / "scene_pipeline.py").exists()
    assert {
        "model_comparison_pipeline_service",
        "multi_pass_pipeline_service",
    }.issubset(_imports("pipeline_job.py"))
    expected_methods = {"__init__", "_session", "enqueue", "publish"}
    for module, class_name in (
        ("model_comparison_pipeline_service.py", "ModelComparisonPipelineService"),
        ("multi_pass_pipeline_service.py", "MultiPassPipelineService"),
    ):
        tree = ast.parse((APP / module).read_text(encoding="utf-8"))
        service = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        methods = {
            node.name
            for node in service.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert methods == expected_methods, module


def test_analysis_repository_does_not_own_scheduler_cancellation() -> None:
    repository_imports = _imports("analysis_run_repository.py")
    assert "analysis_cancellation" not in repository_imports
    assert _imported_names("analysis_run_repository.py", "database").isdisjoint(
        {"ReconstructionJobRow", "ReconstructionLeaseRow", "SceneRow"}
    )

    repository_tree = ast.parse(
        (APP / "analysis_run_repository.py").read_text(encoding="utf-8")
    )
    repository_classes = {
        node.name for node in repository_tree.body if isinstance(node, ast.ClassDef)
    }
    assert "AnalysisCancellationService" not in repository_classes
    assert "AnalysisCancellationService" in {
        node.name
        for node in ast.parse(
            (APP / "analysis_cancellation.py").read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.ClassDef)
    }


def test_transaction_local_analysis_telemetry_opens_no_sessions() -> None:
    assert "database" not in _imports("analysis_run_telemetry.py")
    assert not _imported_names("analysis_run_telemetry.py", "database")


def test_video_processing_capabilities_have_no_aggregate_facade() -> None:
    assert not (APP / "video_processing.py").exists()
    assert _imports("video_segment_planning.py").isdisjoint(
        {
            "database",
            "project_resource_repository",
            "scene_repository",
            "video_store",
        }
    )
    assert _imports("video_ingest_preparation.py").isdisjoint(
        {
            "database",
            "project_resource_repository",
            "scene_repository",
        }
    )
    assert _imports("video_segment_materialization.py").isdisjoint(
        {"video_ffmpeg", "video_ingest_preparation", "video_pipeline"}
    )


def test_ball_detectors_have_provider_specific_owners() -> None:
    assert not (APP / "ball_detection.py").exists()
    assert _imports("ball_detection_contract.py").isdisjoint(
        {
            "ball_detector_factory",
            "ultralytics_ball_detector",
            "wasb_ball_detector",
        }
    )
    assert "wasb_ball_detector" not in _imports(
        "ultralytics_ball_detector.py"
    )
    assert "ultralytics_ball_detector" not in _imports(
        "wasb_ball_detector.py"
    )
    assert "wasb_ball_transport" not in _imports("wasb_ball_protocol.py")
    assert "wasb_ball_detector" not in _imports("wasb_ball_transport.py")


def test_ball_detection_cache_separates_contract_codec_and_storage() -> None:
    assert _imports("ball_detection_cache_contract.py").isdisjoint(
        {"ball_detection_cache", "ball_detection_cache_codec", "fcntl", "os"}
    )
    assert _imports("ball_detection_cache_codec.py").isdisjoint(
        {"ball_detection_cache", "fcntl", "os"}
    )
    storage_imports = _imports("ball_detection_cache.py")
    assert {
        "ball_detection_cache_contract",
        "ball_detection_cache_codec",
    }.issubset(storage_imports)


def test_person_detection_has_direct_capability_owners() -> None:
    assert not (APP / "reconstruction_person_detection.py").exists()

    infrastructure = {
        "database",
        "person_base_detection_cache",
        "person_detection_cache",
        "pipeline_store",
        "reconstruction_inputs",
        "scene_repository",
        "ultralytics_person_inference",
    }
    assert _imports("person_appearance.py").isdisjoint(infrastructure)
    assert _imports("reconstruction_person_annotations.py").isdisjoint(
        infrastructure
    )

    assert _imports("person_detector_provenance.py").isdisjoint(
        {
            "person_base_detection_cache",
            "person_detection_cache",
            "ultralytics_person_inference",
        }
    )
    assert _imports("ultralytics_person_inference.py").isdisjoint(
        {
            "person_base_detection_cache",
            "person_detection_cache",
            "person_detector_provenance",
        }
    )
    cache_imports = _imports("person_base_detection_cache.py")
    assert "person_detection_cache" in cache_imports
    assert "person_detection_provider_contract" in cache_imports
    assert "person_detection_candidate_selection" in cache_imports
    assert "ultralytics_person_inference" not in cache_imports


def test_ball_tracking_has_one_way_algorithm_dependencies() -> None:
    contract_imports = _imports("ball_tracking_contract.py")
    candidate_imports = _imports("ball_tracking_candidates.py")
    solver_imports = _imports("ball_tracking_solver.py")
    projection_imports = _imports("ball_trajectory_projection.py")
    materialization_imports = _imports("ball_trajectory_materialization.py")

    capability_modules = {
        "ball_tracking_candidates",
        "ball_tracking_solver",
        "ball_trajectory_projection",
        "ball_trajectory_materialization",
        "ball_tracking",
    }
    infrastructure_modules = {
        "database",
        "scene_repository",
        "reconstruction_run_repository",
        "ball_detection_cache",
        "ball_detector_factory",
        "ultralytics_ball_detector",
        "wasb_ball_detector",
        "wasb_ball_transport",
    }

    assert contract_imports.isdisjoint(capability_modules | infrastructure_modules)
    assert candidate_imports.isdisjoint(
        {"ball_tracking_solver", "ball_trajectory_projection", "ball_trajectory_materialization"}
        | infrastructure_modules
    )
    assert solver_imports.isdisjoint(
        {"ball_trajectory_projection", "ball_trajectory_materialization", "ball_tracking"}
        | infrastructure_modules
    )
    assert projection_imports.isdisjoint(
        {"ball_tracking_solver", "ball_trajectory_materialization", "ball_tracking"}
        | infrastructure_modules
    )
    assert "ball_tracking" not in materialization_imports

    orchestrator = ast.parse((APP / "ball_tracking.py").read_text(encoding="utf-8"))
    functions = {
        node.name
        for node in orchestrator.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert functions == {"resolve_ball_trajectory"}


def test_wasb_service_has_one_multipart_transport_contract() -> None:
    detector_source = (APP / "wasb_ball_detector.py").read_text(encoding="utf-8")
    transport_source = (APP / "wasb_ball_transport.py").read_text(encoding="utf-8")
    contract_source = (APP / "ball_detection_contract.py").read_text(encoding="utf-8")
    assert "files=files" in transport_source
    assert '"manifest"' in transport_source
    assert "dataBase64" not in detector_source + transport_source
    assert "wasb_subprocess" not in detector_source + transport_source
    assert '"wasb-subprocess"' not in contract_source


def test_completed_legacy_cutover_tools_are_not_runtime_modules() -> None:
    assert not (APP / "artifact_cutover.py").exists()
    assert not (APP / "video_generation_cutover.py").exists()
    assert not (APP / "repair_match_chains.py").exists()


def test_jersey_evidence_contract_fusion_and_roster_ranking_are_separate() -> None:
    assert _imports("jersey_ocr_contract.py").isdisjoint(
        {"jersey_ocr_fusion", "jersey_roster_candidates"}
    )
    assert "jersey_roster_candidates" not in _imports("jersey_ocr_fusion.py")
    assert "jersey_ocr_fusion" not in _imports("jersey_roster_candidates.py")
    assert _imported_names(
        "reconstruction_canonical_identity_resolution.py", "jersey_ocr_fusion"
    ).isdisjoint({"JerseyEvidenceSummary", "JerseyFusionConfig", "JerseyOcrObservation"})


def test_reconstruction_tracking_capabilities_are_separate() -> None:
    assert not (APP / "reconstruction_identity_tracking.py").exists()

    person_tracking_imports = _imports("reconstruction_person_tracking.py")
    canonical_resolution_imports = _imports(
        "reconstruction_canonical_identity_resolution.py"
    )
    team_classification_imports = _imports(
        "reconstruction_team_classification.py"
    )

    assert person_tracking_imports.isdisjoint(
        {
            "cv2",
            "identity_resolver",
            "jersey_ocr_contract",
            "reconstruction_canonical_identity_resolution",
            "reconstruction_team_classification",
        }
    )
    assert canonical_resolution_imports.isdisjoint(
        {
            "cv2",
            "numpy",
            "scipy.optimize",
            "reconstruction_person_tracking",
            "reconstruction_team_classification",
        }
    )
    assert team_classification_imports.isdisjoint(
        {
            "identity_resolver",
            "jersey_ocr_contract",
            "reconstruction_canonical_identity_resolution",
            "reconstruction_person_tracking",
            "scipy.optimize",
        }
    )


def test_sampled_frame_pipeline_capabilities_are_separate() -> None:
    assert not (APP / "reconstruction_sampled_frame_phase.py").exists()

    contract_imports = _imports("reconstruction_sampled_frame_contract.py")
    preparation_imports = _imports(
        "reconstruction_sampled_detection_preparation.py"
    )
    calibration_imports = _imports("reconstruction_sampled_calibration.py")
    detection_imports = _imports("reconstruction_sampled_frame_detection.py")
    reid_imports = _imports("reconstruction_reid_phase.py")

    assert contract_imports.isdisjoint(
        {
            "ball_detection_contract",
            "config",
            "identity_worker",
            "reconstruction_ball_detector_selection",
            "reconstruction_inputs",
            "reconstruction_progress",
            "reconstruction_reid_phase",
            "reconstruction_sampled_calibration",
            "reconstruction_sampled_detection_preparation",
            "reconstruction_sampled_frame_detection",
        }
    )
    assert preparation_imports.isdisjoint(
        {
            "identity_worker",
            "reconstruction_reid_phase",
            "reconstruction_sampled_calibration",
            "reconstruction_sampled_frame_detection",
        }
    )
    assert calibration_imports.isdisjoint(
        {
            "app.person_base_detection_cache",
            "identity_worker",
            "reconstruction_ball_detector_selection",
            "reconstruction_reid_phase",
            "reconstruction_sampled_detection_preparation",
            "reconstruction_sampled_frame_detection",
        }
    )
    assert detection_imports.isdisjoint(
        {
            "identity_worker",
            "reconstruction_ball_detector_selection",
            "reconstruction_calibration_detection",
            "reconstruction_reid_phase",
        }
    )
    assert reid_imports.isdisjoint(
        {
            "app.person_base_detection_cache",
            "reconstruction_ball_detector_selection",
            "reconstruction_calibration_detection",
            "reconstruction_sampled_calibration",
            "reconstruction_sampled_detection_preparation",
            "reconstruction_sampled_frame_detection",
        }
    )

    streaming_source = (
        APP / "reconstruction_sampled_frame_detection.py"
    ).read_text(encoding="utf-8")
    accumulator_source = (APP / "reconstruction_sampled_calibration.py").read_text(
        encoding="utf-8"
    )
    assert streaming_source.count("cached_base_frame_detections(") == 1
    assert "cached_base_frame_detections(" not in accumulator_source
    assert "imread(" not in accumulator_source


def test_scene_track_publication_capabilities_are_separate() -> None:
    assert not (APP / "reconstruction_scene_tracks.py").exists()
    assert not (APP / "reconstruction_tracks.py").exists()

    trajectory_imports = _imports("reconstruction_track_trajectory.py")
    observation_imports = _imports("reconstruction_track_observations.py")
    publisher_imports = _imports("reconstruction_scene_track_publisher.py")

    assert trajectory_imports.isdisjoint(
        {
            "reconstruction_inputs",
            "reconstruction_scene_track_publisher",
            "reconstruction_team_classification",
            "reconstruction_track_observations",
        }
    )
    assert observation_imports.isdisjoint(
        {
            "pitch_calibration_contract",
            "reconstruction_inputs",
            "reconstruction_scene_track_publisher",
            "reconstruction_team_classification",
        }
    )
    assert {
        "reconstruction_track_trajectory",
        "reconstruction_track_observations",
    }.issubset(publisher_imports)
    assert publisher_imports.isdisjoint(
        {
            "math",
            "numpy",
            "reconstruction_tracks",
        }
    )
    assert "reconstruction_scene_track_publisher" in _imports(
        "reconstruction_identity_phase.py"
    )
    assert "reconstruction_scene_track_publisher" in _imports(
        "model_comparison.py"
    )
    assert not any(
        "def ball_keyframes(" in path.read_text(encoding="utf-8")
        for path in APP.glob("*.py")
    )


def test_track_primitives_have_direct_capability_owners() -> None:
    bbox_imports = _imports("bounding_box_geometry.py")
    pitch_projection_imports = _imports("reconstruction_pitch_projection.py")
    latent_presence_imports = _imports("reconstruction_latent_presence.py")
    canonical_id_imports = _imports("reconstruction_canonical_person_id.py")

    capability_modules = {
        "bounding_box_geometry",
        "reconstruction_canonical_person_id",
        "reconstruction_latent_presence",
        "reconstruction_pitch_projection",
        "reconstruction_track_observations",
        "reconstruction_track_trajectory",
    }
    assert bbox_imports.isdisjoint(capability_modules)
    assert latent_presence_imports.isdisjoint(capability_modules)
    assert canonical_id_imports.isdisjoint(capability_modules)
    assert pitch_projection_imports.isdisjoint(
        capability_modules - {"reconstruction_pitch_projection"}
    )

    trajectory_imports = _imports("reconstruction_track_trajectory.py")
    assert {
        "reconstruction_latent_presence",
        "reconstruction_pitch_projection",
    }.issubset(trajectory_imports)
    assert "reconstruction_track_observations" not in trajectory_imports
    assert "reconstruction_track_trajectory" in _imports(
        "reconstruction_track_observations.py"
    )

    for path in APP.glob("*.py"):
        assert "reconstruction_tracks" not in _imports(path.name), path.name


def test_identity_resolution_has_explicit_algorithm_boundaries() -> None:
    from app import identity_resolver as identity_resolution_workflow

    contract_dependencies = _imports("identity_resolution_contract.py")
    assert contract_dependencies.isdisjoint(
        {
            "identity_assignment",
            "identity_pairwise_scoring",
            "identity_resolution_components",
            "identity_resolver",
            "reconstruction",
            "database",
        }
    )
    assert "identity_assignment" not in _imports("identity_pairwise_scoring.py")
    assert "identity_pairwise_scoring" not in _imports("identity_assignment.py")
    assert _imports("identity_resolution_components.py").isdisjoint(
        {"identity_assignment", "identity_pairwise_scoring", "identity_resolver"}
    )

    resolver_tree = ast.parse(
        (APP / "identity_resolver.py").read_text(encoding="utf-8")
    )
    resolver_classes = {
        node.name for node in resolver_tree.body if isinstance(node, ast.ClassDef)
    }
    resolver_functions = {
        node.name
        for node in resolver_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert resolver_classes == set()
    assert resolver_functions == {"_edge_sort_key", "resolve_identities"}
    assert identity_resolution_workflow.__all__ == ["resolve_identities"]
    assert not hasattr(identity_resolution_workflow, "IdentityTracklet")
    assert not hasattr(identity_resolution_workflow, "IdentityResolverConfig")
    assert _imported_names(
        "reconstruction_canonical_identity_resolution.py", "identity_resolver"
    ) == {"resolve_identities"}


def test_roster_identity_has_explicit_review_pipeline_boundaries() -> None:
    assert not (APP / "roster_identity_resolver.py").exists()
    assert _imports("roster_identity_temporal.py").isdisjoint(
        {
            "roster_identity_contract",
            "roster_identity_scoring",
            "roster_identity_assignment",
            "closed_set_roster_resolution",
            "numpy",
            "scipy.optimize",
        }
    )
    assert _imports("roster_identity_contract.py").isdisjoint(
        {
            "roster_identity_scoring",
            "roster_identity_assignment",
            "roster_identity_conflicts",
            "closed_set_roster_resolution",
            "numpy",
            "scipy.optimize",
        }
    )
    assert "roster_identity_assignment" not in _imports(
        "roster_identity_scoring.py"
    )
    assert "roster_identity_scoring" not in _imports(
        "roster_identity_assignment.py"
    )
    assert _imported_names(
        "reconstruction_roster_identity_resolution.py",
        "closed_set_roster_resolution",
    ) == {"resolve_closed_set_roster"}
    assert _imported_names(
        "reconstruction_roster_identity_resolution.py",
        "roster_identity_contract",
    ) == {
        "AttributeEvidence",
        "CanonicalPersonEvidence",
        "PersistedRosterPlayer",
    }


def test_reconstruction_jersey_pipeline_has_direct_capability_owners() -> None:
    assert not (APP / "reconstruction_jersey_evidence.py").exists()
    assert not (APP / "jersey_ocr_worker.py").exists()
    assert not (APP / "jersey_ocr_worker_protocol.py").exists()
    assert _imports("reconstruction_jersey_sampling.py").isdisjoint(
        {
            "jersey_ocr_worker_client",
            "jersey_ocr_worker_transport",
            "jersey_ocr_fusion",
            "reconstruction_jersey_inference",
        }
    )
    assert _imports("reconstruction_jersey_resolution.py").isdisjoint(
        {
            "jersey_ocr_worker_client",
            "jersey_ocr_worker_transport",
            "cv2",
            "reconstruction_jersey_inference",
        }
    )
    assert _imports("reconstruction_jersey_policy.py") == {
        "dataclasses",
        "jersey_ocr_contract",
    }
    assert "reconstruction_jersey_resolution" not in _imports(
        "reconstruction_jersey_inference.py"
    )
    assert _imports("jersey_ocr_worker_contract.py").isdisjoint(
        {"config", "httpx", "jersey_ocr_worker_client", "jersey_ocr_worker_transport"}
    )
    validation_modules = (
        "jersey_ocr_worker_wire_validation.py",
        "jersey_ocr_worker_model_contract.py",
        "jersey_ocr_worker_item_validation.py",
        "jersey_ocr_worker_batch_validation.py",
    )
    forbidden_validation_dependencies = {
        "config",
        "httpx",
        "jersey_ocr_worker_client",
        "jersey_ocr_worker_transport",
    }
    for module in validation_modules:
        assert _imports(module).isdisjoint(forbidden_validation_dependencies)
    assert _imported_names(
        "jersey_ocr_worker_client.py",
        "jersey_ocr_worker_model_contract",
    ) == {"project_model_contract", "validate_readiness_payload"}
    assert _imported_names(
        "jersey_ocr_worker_client.py",
        "jersey_ocr_worker_batch_validation",
    ) == {"validate_analysis_payload"}
    assert "jersey_ocr_worker_item_validation" in _imports(
        "jersey_ocr_worker_batch_validation.py"
    )
    assert "jersey_ocr_worker_batch_validation" not in _imports(
        "jersey_ocr_worker_item_validation.py"
    )
    assert "httpx" in _imports("jersey_ocr_worker_transport.py")
    assert "httpx" not in _imports("jersey_ocr_worker_client.py")


def test_identity_worker_client_has_strict_capability_owners() -> None:
    assert not (APP / "identity_worker.py").exists()
    assert _imports("identity_worker_contract.py").isdisjoint(
        {"config", "httpx", "identity_worker_client", "identity_worker_transport"}
    )
    validation_modules = (
        "identity_worker_wire_validation.py",
        "identity_worker_model_contract.py",
        "identity_worker_item_validation.py",
        "identity_worker_batch_validation.py",
    )
    forbidden_validation_dependencies = {
        "config",
        "httpx",
        "identity_worker_client",
        "identity_worker_transport",
    }
    for module in validation_modules:
        assert _imports(module).isdisjoint(forbidden_validation_dependencies)
    assert _imported_names(
        "identity_worker_client.py",
        "identity_worker_model_contract",
    ) == {
        "project_model_contract",
        "project_runtime_contract",
        "validate_readiness_payload",
    }
    assert _imported_names(
        "identity_worker_client.py",
        "identity_worker_batch_validation",
    ) == {"validate_embedding_payload"}
    assert "identity_worker_item_validation" in _imports(
        "identity_worker_batch_validation.py"
    )
    assert "identity_worker_batch_validation" not in _imports(
        "identity_worker_item_validation.py"
    )
    assert "httpx" in _imports("identity_worker_transport.py")
    assert "httpx" not in _imports("identity_worker_client.py")
    assert _imports("reconstruction_reid_phase.py").isdisjoint(
        {
            "identity_worker_transport",
            "identity_worker_wire_validation",
            "identity_worker_item_validation",
            "identity_worker_batch_validation",
        }
    )


def test_pipeline_store_has_no_resource_terminal_authority() -> None:
    store_tree = ast.parse((APP / "pipeline_store.py").read_text(encoding="utf-8"))
    store_class = next(
        node
        for node in store_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PipelineStore"
    )
    store_methods = {
        node.name
        for node in store_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"cancel", "fail"}.isdisjoint(store_methods)
    assert _imports("pipeline_store.py").isdisjoint(
        {
            "pipeline_terminal_service",
            "scene_document",
            "scene_index_projection",
        }
    )
    assert _imported_names("pipeline_store.py", "database").isdisjoint(
        {"SceneRow", "VideoAssetRow"}
    )
    assert _imported_names(
        "pipeline_job.py", "pipeline_terminal_service"
    ) == {"pipeline_terminals"}
    assert _imported_names(
        "project_analysis_routes.py", "pipeline_terminal_service"
    ) == {"pipeline_terminals"}


def test_player_action_planning_and_persistence_are_distinct() -> None:
    assert not (APP / "player_actions.py").exists()
    assert _imports("player_action_planning.py").isdisjoint(
        {"database", "scene_repository", "player_action_commands"}
    )
    assert {
        "player_action_planning",
        "scene_repository",
    }.issubset(_imports("player_action_commands.py"))
    for module in ("player_action_commands.py", "player_action_planning.py"):
        tree = ast.parse((APP / module).read_text(encoding="utf-8"))
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            assert "persist" not in {
                argument.arg
                for argument in (
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                )
            }


def test_project_identity_reconciliation_and_projection_are_direct_boundaries() -> None:
    assert _imports("project_identity_reconciliation.py").isdisjoint(
        {
            "database",
            "project_models",
            "project_identity_repository",
            "project_identity_projection",
            "sqlalchemy",
        }
    )
    assert _imports("project_identity_projection.py").isdisjoint(
        {"project_identity_repository", "sqlalchemy"}
    )
    assert {
        "project_identity_projection",
        "project_identity_reconciliation",
    }.issubset(_imports("project_identity_repository.py"))


def test_reconstruction_domain_has_direct_contract_and_accumulator_owners() -> None:
    assert not (APP / "reconstruction_domain.py").exists()

    state_tree = ast.parse(
        (APP / "reconstruction_track_state.py").read_text(encoding="utf-8")
    )
    state_class = next(
        node
        for node in state_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrackState"
    )
    state_methods = {
        node.name
        for node in state_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert state_methods == {
        "feature",
        "local_tracklet_id",
        "positive_annotation_ids",
        "reid_feature",
    }
    assert _imports("reconstruction_track_state.py") == {
        "__future__",
        "dataclasses",
        "numpy",
    }
    assert _imports("reconstruction_person_detection_contract.py") == {
        "__future__",
        "dataclasses",
        "numpy",
    }
    assert _imports("reconstruction_errors.py") == {"__future__"}

    assert _imported_names(
        "reconstruction_person_tracking.py", "track_observation_accumulator"
    ) == {"append_track_observation"}
    assert _imports("track_observation_accumulator.py").isdisjoint(
        {
            "database",
            "reconstruction_person_tracking",
            "reconstruction_identity_merging",
        }
    )
    assert _imports("track_reid_accumulator.py").isdisjoint(
        {
            "database",
            "track_observation_accumulator",
            "reconstruction_identity_merging",
        }
    )
    for module in APP.glob("*.py"):
        assert "reconstruction_domain" not in _imports(module.name), module.name
