from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "calibration_worker_service"


def test_main_is_only_the_http_composition_root() -> None:
    source = (PACKAGE / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_functions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert top_level_functions == ["create_app"]
    assert "PnLCalibEngine" not in source
    assert "numpy" not in source
    assert "torch" not in source


def test_calibration_processing_does_not_depend_on_http_framework() -> None:
    for name in (
        "calibration_cache.py",
        "calibration_contract.py",
        "calibration_projector.py",
        "calibration_service.py",
        "engine_factory.py",
        "frame_decoder.py",
        "pnlcalib_engine.py",
        "pnlcalib_inference.py",
        "pnlcalib_runtime.py",
        "runtime.py",
    ):
        source = (PACKAGE / name).read_text(encoding="utf-8")
        assert "fastapi" not in source
        assert "starlette" not in source


def test_engine_god_file_was_removed_and_service_uses_narrow_contract() -> None:
    assert not (PACKAGE / "engine.py").exists()
    service = (PACKAGE / "calibration_service.py").read_text(encoding="utf-8")
    assert "PnLCalibEngine" not in service
    assert "load_pnlcalib_models" not in service
    assert "decode_frame" in service
    assert "CalibrationEngineProvider" in service

    runtime = (PACKAGE / "runtime.py").read_text(encoding="utf-8")
    assert "pnlcalib_engine" not in runtime
    assert "engine_factory" not in runtime


def test_contract_does_not_import_runtime_or_transport_owners() -> None:
    source = (PACKAGE / "calibration_contract.py").read_text(encoding="utf-8")
    assert "fastapi" not in source
    assert "pnlcalib_runtime" not in source
    assert "pnlcalib_engine" not in source
    assert "calibration_cache" not in source
