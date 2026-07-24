from types import SimpleNamespace

from app.reconstruction_publish_payloads import identity_runtime_quality


def _frame(status: str):
    return SimpleNamespace(identity_worker_diagnostics={"status": status})


def _identity(status: str):
    return SimpleNamespace(jersey_ocr_diagnostics={"status": status})


def test_unavailable_reid_marks_identity_runtime_degraded():
    result = identity_runtime_quality(
        _frame("unavailable"),
        _identity("ready"),
        jersey_ocr_profile="automatic",
    )

    assert result["status"] == "degraded"
    assert result["reasons"] == ["reid-unavailable"]
    assert result["automaticCrossGapIdentityAvailable"] is False


def test_intentionally_disabled_jersey_ocr_does_not_degrade_runtime():
    result = identity_runtime_quality(
        _frame("ready"),
        _identity("skipped-by-profile"),
        jersey_ocr_profile="off",
    )

    assert result["status"] == "ready"
    assert result["reasons"] == []
