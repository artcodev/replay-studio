from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from threading import Event, Thread
from typing import Final

from .config import get_settings
from .reconstruction_run_repository import reconstruction_runs


logger = logging.getLogger(__name__)


_CHILD_OWNED_RECONSTRUCTION_STATUSES: Final = {
    "queued",
    "processing",
    # Publication stores the terminal scene immediately before it updates the
    # compact AnalysisRun. Let the same child finish that short epilogue instead
    # of killing it between the two durable writes.
    "ready",
    "failed",
}


def _spawn_reconstruction_process(
    scene_id: str,
    run_id: str,
    input_fingerprint: str,
) -> subprocess.Popen:
    """Start one killable reconstruction process.

    A process boundary is intentional here. Native model inference and worker
    HTTP calls can remain blocked long after a run has been fenced in the
    database; Python cannot safely stop the equivalent worker thread.
    ``start_new_session`` gives the supervisor ownership of the complete child
    process group, including any model/runtime descendants.
    """

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "app.reconstruction_job",
            scene_id,
            run_id,
            input_fingerprint,
        ],
        start_new_session=(os.name == "posix"),
    )


def _terminate_process_tree(
    process: subprocess.Popen,
    grace_seconds: float,
) -> None:
    """Terminate a reconstruction process group, escalating when necessary."""

    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - the production runner is a Linux container
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=max(0.0, float(grace_seconds)))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - the production runner is a Linux container
            process.kill()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=max(0.1, float(grace_seconds)))
    except subprocess.TimeoutExpired:
        logger.error(
            "Reconstruction process %s did not exit after SIGKILL",
            process.pid,
        )


class DedicatedReconstructionRecoveryMonitor:
    """Supervise killable reconstruction subprocesses in the dedicated runner.

    Reconstruction never executes in the API process. This process supervisor
    makes cancellation release physical compute: a fenced blocking child is
    terminated before its replacement is started.
    """

    def __init__(
        self,
        poll_seconds: float | None = None,
        max_workers: int | None = None,
        termination_grace_seconds: float = 1.0,
    ) -> None:
        configured_poll = (
            get_settings().reconstruction_recovery_poll_seconds
            if poll_seconds is None
            else poll_seconds
        )
        self.poll_seconds = max(0.05, float(configured_poll))
        configured_workers = (
            get_settings().reconstruction_recovery_max_workers
            if max_workers is None
            else max_workers
        )
        self.max_workers = max(1, int(configured_workers))
        self.termination_grace_seconds = max(
            0.0,
            float(termination_grace_seconds),
        )
        self._stop = Event()
        self._jobs: dict[
            tuple[str, str],
            tuple[str, subprocess.Popen],
        ] = {}
        self._thread = Thread(
            target=self._run,
            name="dedicated-reconstruction-recovery-monitor",
            daemon=True,
        )

    @staticmethod
    def _run_is_current(
        scene_id: str,
        run_id: str,
        input_fingerprint: str,
    ) -> bool:
        """Return false once cancellation/retry has fenced this exact child."""

        return reconstruction_runs.reconstruction_run_is_current(
            scene_id,
            run_id,
            input_fingerprint,
            statuses=_CHILD_OWNED_RECONSTRUCTION_STATUSES,
        )

    def _stop_job(
        self,
        key: tuple[str, str],
        process: subprocess.Popen,
        *,
        reason: str,
    ) -> None:
        logger.info(
            "Stopping reconstruction child scene=%s run=%s pid=%s: %s",
            key[0],
            key[1],
            process.pid,
            reason,
        )
        _terminate_process_tree(process, self.termination_grace_seconds)

    def _reap_and_cancel_fenced_jobs(self) -> None:
        for key, (input_fingerprint, process) in list(self._jobs.items()):
            return_code = process.poll()
            if return_code is not None:
                self._jobs.pop(key, None)
                if return_code != 0:
                    logger.warning(
                        "Reconstruction child scene=%s run=%s exited with %s",
                        key[0],
                        key[1],
                        return_code,
                    )
                    # The supervisor just observed the hard death: free the
                    # dead owner's lease now instead of stranding the job in
                    # processing for the remaining lease TTL.
                    try:
                        reconstruction_runs.release_crashed_reconstruction_run(
                            key[0],
                            key[1],
                            input_fingerprint,
                            error=(
                                "Reconstruction child exited with code "
                                f"{return_code}"
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "Could not release the crashed lease scene=%s run=%s",
                            key[0],
                            key[1],
                        )
                continue
            try:
                current = self._run_is_current(
                    key[0],
                    key[1],
                    input_fingerprint,
                )
            except Exception:
                # A transient database outage is not evidence that ownership
                # was revoked. Keep the child and retry on the next scan.
                logger.exception(
                    "Could not verify reconstruction child scene=%s run=%s",
                    key[0],
                    key[1],
                )
                continue
            if current:
                continue
            self._stop_job(key, process, reason="run was cancelled or superseded")
            self._jobs.pop(key, None)

    def _start_recoverable_jobs(self) -> None:
        available = self.max_workers - len(self._jobs)
        if available <= 0 or self._stop.is_set():
            return
        for scene_id, run_id, input_fingerprint in (
            reconstruction_runs.list_recoverable_reconstruction_runs()
        ):
            if self._stop.is_set():
                return
            key = (scene_id, run_id)
            if key in self._jobs:
                continue
            process = _spawn_reconstruction_process(
                scene_id,
                run_id,
                input_fingerprint,
            )
            self._jobs[key] = (input_fingerprint, process)
            logger.info(
                "Started reconstruction child scene=%s run=%s pid=%s",
                scene_id,
                run_id,
                process.pid,
            )
            available -= 1
            if available <= 0:
                return

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                # Reaping/fencing comes before discovery. With max_workers=1,
                # this ordering guarantees the old YOLO process is gone before
                # a retry can consume the newly free slot.
                self._reap_and_cancel_fenced_jobs()
                self._start_recoverable_jobs()
            except Exception:
                logger.exception("Dedicated reconstruction recovery scan failed")
            self._stop.wait(self.poll_seconds)

    def start(self) -> "DedicatedReconstructionRecoveryMonitor":
        self._thread.start()
        return self

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout=max(0.0, float(timeout)))
        for key, (_input_fingerprint, process) in list(self._jobs.items()):
            self._stop_job(key, process, reason="runner is shutting down")
            self._jobs.pop(key, None)

    def is_alive(self) -> bool:
        return self._thread.is_alive()


def start_dedicated_reconstruction_recovery(
) -> DedicatedReconstructionRecoveryMonitor:
    """Start the process-isolated monitor used by reconstruction-runner."""

    try:
        from .analysis_runtime import recover_missed_identity_sync

        repaired = recover_missed_identity_sync()
        if repaired:
            logger.info(
                "Repaired %d missed identity-sync epilogue(s) on startup",
                repaired,
            )
    except Exception:
        # The sweep is a best-effort repair; recovery itself must start.
        logger.exception("Missed identity-sync sweep failed on startup")
    return DedicatedReconstructionRecoveryMonitor().start()
