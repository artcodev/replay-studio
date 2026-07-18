from __future__ import annotations

from collections.abc import Callable
from threading import Lock

from .calibration_contract import CalibrationEngine


class CalibrationEngineRuntime:
    """Own the process-local model lifecycle and its single loaded instance."""

    def __init__(
        self,
        factory: Callable[[], CalibrationEngine],
    ) -> None:
        self._factory = factory
        self._engine: CalibrationEngine | None = None
        self._lock = Lock()

    def get_engine(self) -> CalibrationEngine:
        with self._lock:
            if self._engine is None:
                self._engine = self._factory()
            return self._engine
