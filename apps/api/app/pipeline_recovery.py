from __future__ import annotations

"""Process-isolated supervisor for durable generic pipeline jobs."""

import logging
import os
import signal
import subprocess
import sys
from threading import Event, Thread
from uuid import uuid4

from .config import get_settings
from .pipeline_store import PipelineStore, pipeline_store


logger = logging.getLogger(__name__)


def _spawn_pipeline_process(job_id: str, lease_token: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "app.pipeline_job", job_id, lease_token],
        start_new_session=(os.name == "posix"),
    )


def _terminate_process_tree(process: subprocess.Popen, grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=max(0.0, grace_seconds))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover
            process.kill()
    except ProcessLookupError:
        return


class DedicatedPipelineMonitor:
    def __init__(
        self,
        *,
        store: PipelineStore = pipeline_store,
        poll_seconds: float | None = None,
        max_workers: int | None = None,
        termination_grace_seconds: float = 1.0,
    ) -> None:
        settings = get_settings()
        self.store = store
        self.poll_seconds = max(
            0.05,
            float(
                settings.pipeline_recovery_poll_seconds
                if poll_seconds is None
                else poll_seconds
            ),
        )
        self.max_workers = max(
            1,
            int(
                settings.pipeline_recovery_max_workers
                if max_workers is None
                else max_workers
            ),
        )
        self.termination_grace_seconds = max(0.0, termination_grace_seconds)
        self.owner_id = f"pipeline-runner-{uuid4().hex}"
        self._jobs: dict[str, tuple[str, subprocess.Popen]] = {}
        self._stop = Event()
        self._thread = Thread(
            target=self._run,
            name="dedicated-pipeline-monitor",
            daemon=True,
        )

    def _reap_and_fence(self) -> None:
        for job_id, (token, process) in list(self._jobs.items()):
            return_code = process.poll()
            if return_code is not None:
                self._jobs.pop(job_id, None)
                if return_code != 0:
                    logger.warning(
                        "Pipeline child job=%s exited with %s", job_id, return_code
                    )
                continue
            try:
                current = self.store.is_claim_current(job_id, token)
            except Exception:
                logger.exception("Could not verify pipeline claim %s", job_id)
                continue
            if current:
                continue
            _terminate_process_tree(process, self.termination_grace_seconds)
            self._jobs.pop(job_id, None)

    def _start_jobs(self) -> None:
        available = self.max_workers - len(self._jobs)
        if available <= 0:
            return
        for job_id in self.store.list_recoverable(limit=available * 2):
            if self._stop.is_set() or available <= 0:
                return
            if job_id in self._jobs:
                continue
            claim = self.store.claim(job_id, self.owner_id)
            if claim is None or claim.lease_token is None:
                continue
            try:
                process = _spawn_pipeline_process(job_id, claim.lease_token)
            except Exception:
                self.store.abandon_claim(job_id, claim.lease_token)
                raise
            self._jobs[job_id] = (claim.lease_token, process)
            available -= 1

    def scan_once(self) -> None:
        # Fenced children are stopped before their slot can be reused.
        self._reap_and_fence()
        self._start_jobs()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception:
                logger.exception("Dedicated pipeline recovery scan failed")
            self._stop.wait(self.poll_seconds)

    def start(self) -> "DedicatedPipelineMonitor":
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=max(0.0, timeout))
        for job_id, (_token, process) in list(self._jobs.items()):
            _terminate_process_tree(process, self.termination_grace_seconds)
            self._jobs.pop(job_id, None)


def start_dedicated_pipeline_monitor() -> DedicatedPipelineMonitor:
    return DedicatedPipelineMonitor().start()
