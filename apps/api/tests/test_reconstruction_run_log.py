from __future__ import annotations

import json

from app.reconstruction_progress import ReconstructionProgress
from app.reconstruction_run_log import (
    NullRunLog,
    ReconstructionRunLog,
    open_reconstruction_run_log,
)


def _events(path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_run_log_appends_one_json_object_per_event(tmp_path):
    log = ReconstructionRunLog(
        tmp_path / "analysis-runs", scene_id="scene-1", run_id="run-a"
    )
    log.event("phase-finished", phase="identity", rawTrackCount=7)
    log.close("ready", extra="value")

    assert log.path is not None
    assert log.path.name == "run-scene-1-run-a.jsonl"
    events = _events(log.path)
    assert [item["event"] for item in events] == [
        "run-log-opened",
        "phase-finished",
        "run-finished",
    ]
    assert events[0]["sceneId"] == "scene-1"
    assert events[1]["phase"] == "identity"
    assert events[1]["rawTrackCount"] == 7
    assert events[2]["status"] == "ready"
    assert events[2]["writeErrorCount"] == 0
    assert all("t" in item and "elapsedSeconds" in item for item in events)


def test_run_log_swallows_serialization_faults_without_failing(tmp_path):
    log = ReconstructionRunLog(
        tmp_path / "analysis-runs", scene_id="scene-1", run_id="run-a"
    )

    class Weird:
        def __str__(self):
            return "weird"

    # Non-JSON values are stringified; the analysis never fails on a log line.
    log.event("odd", value=Weird())
    log.close("ready")

    events = _events(log.path)
    assert events[1]["value"] == "weird"
    assert events[2]["writeErrorCount"] == 0


def test_run_log_factory_returns_null_log_when_disabled(tmp_path):
    disabled = open_reconstruction_run_log(
        scene_id="scene-1",
        run_id="run-a",
        directory=tmp_path,
        enabled=False,
    )
    assert isinstance(disabled, NullRunLog)
    disabled.event("ignored")
    disabled.close("ready")
    assert list(tmp_path.iterdir()) == []


def test_progress_ticks_are_journaled_without_throttling(tmp_path):
    log = ReconstructionRunLog(
        tmp_path / "analysis-runs", scene_id="scene-1", run_id="run-a"
    )
    scene = {"payload": {"videoAsset": {}}}
    progress = ReconstructionProgress(scene, run_log=log)

    progress.update("detection", 3, "Detecting", "frame 1", 60, 80, completed=1, total=2)
    progress.update("detection", 3, "Detecting", "frame 2", 60, 80, completed=2, total=2)
    progress.complete(track_count=5, ball_samples=9)
    log.close("ready")

    events = _events(log.path)
    progress_events = [item for item in events if item["event"] == "progress"]
    assert [item["detail"] for item in progress_events[:2]] == ["frame 1", "frame 2"]
    assert progress_events[-1]["phase"] == "complete"
    assert progress_events[-1]["overallPercent"] == 100
