"""Append-only JSONL journal for one reconstruction run.

Every pipeline step, phase summary and degradation event is appended as one
JSON object per line so a finished run can be analyzed independently of the
editor (jq, pandas, a notebook). The journal is strictly an observer: any IO
or serialization fault is swallowed after being counted, because a log line
must never fail an analysis that is otherwise healthy.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, IO


RUN_LOG_SCHEMA_VERSION = 1


def _json_default(value: Any) -> str:
    return str(value)


class NullRunLog:
    """No-op journal used when run logging is disabled or unavailable."""

    path: Path | None = None

    def event(self, name: str, /, **payload: Any) -> None:
        return None

    def close(self, status: str, **payload: Any) -> None:
        return None


class ReconstructionRunLog(NullRunLog):
    """One JSONL file per run: ``<directory>/run-<sceneId>-<runId>.jsonl``."""

    def __init__(
        self,
        directory: str | Path,
        *,
        scene_id: str,
        run_id: str,
    ) -> None:
        self.scene_id = str(scene_id)
        self.run_id = str(run_id)
        self._started = monotonic()
        self._write_errors = 0
        self._handle: IO[str] | None = None
        self.path: Path | None = None
        try:
            root = Path(directory).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            self.path = root / f"run-{self.scene_id}-{self.run_id}.jsonl"
            self._handle = self.path.open("a", encoding="utf-8")
        except OSError:
            self._handle = None
        self.event(
            "run-log-opened",
            schemaVersion=RUN_LOG_SCHEMA_VERSION,
            sceneId=self.scene_id,
            runId=self.run_id,
        )

    def event(self, name: str, /, **payload: Any) -> None:
        if self._handle is None:
            return
        record = {
            "t": datetime.now(UTC).isoformat(),
            "elapsedSeconds": round(monotonic() - self._started, 3),
            "event": str(name),
            **payload,
        }
        try:
            self._handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=_json_default,
                )
            )
            self._handle.write("\n")
            self._handle.flush()
        except (OSError, TypeError, ValueError):
            self._write_errors += 1

    def close(self, status: str, **payload: Any) -> None:
        self.event(
            "run-finished",
            status=str(status),
            writeErrorCount=self._write_errors,
            **payload,
        )
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None


def open_reconstruction_run_log(
    *,
    scene_id: str,
    run_id: str,
    directory: str | Path | None,
    enabled: bool,
) -> NullRunLog:
    """Open a run journal, or a no-op journal when disabled/unavailable."""

    if not enabled or not directory:
        return NullRunLog()
    return ReconstructionRunLog(directory, scene_id=scene_id, run_id=run_id)


__all__ = (
    "RUN_LOG_SCHEMA_VERSION",
    "NullRunLog",
    "ReconstructionRunLog",
    "open_reconstruction_run_log",
)
