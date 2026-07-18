from __future__ import annotations

"""Fail-closed errors shared by reconstruction capabilities."""


class ReconstructionError(RuntimeError):
    pass


class ReconstructionCancelled(ReconstructionError):
    """The owning AnalysisRun acknowledged a cooperative cancellation."""


class IdentityCorrectionError(ReconstructionError):
    """An identity correction failed with machine-readable diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        correction_id: str,
        action: str,
        status: str,
        reason: str,
        source_track_id: str | None = None,
        target_id: str | None = None,
        candidates: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic = {
            "correctionId": correction_id,
            "action": action,
            "status": status,
            "reason": reason,
            "message": message,
            "sourceTrackId": source_track_id,
            "targetId": target_id,
            "candidates": candidates or [],
        }


class StaleReconstructionRun(ReconstructionError):
    """The worker no longer owns the scene revision it started from."""

