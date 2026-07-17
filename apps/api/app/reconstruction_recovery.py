from __future__ import annotations

import logging
from threading import Event, Thread

from .config import get_settings
from .reconstruction import reconstruct_scene_by_id
from .store import scene_store


logger = logging.getLogger(__name__)


def recover_queued_reconstruction_jobs() -> int:
    """Claim and execute queued or stale/missing-lease processing runs.

    Candidate discovery is intentionally non-locking. ``reconstruct_scene_by_id``
    performs the cross-process atomic claim, so any number of API recovery
    monitors can scan the same SQLite database without duplicate ownership.
    """

    recovered = 0
    scene_store.fail_unrecoverable_reconstruction_runs()
    for scene_id, run_id, input_fingerprint in (
        scene_store.list_recoverable_reconstruction_runs()
    ):
        try:
            if reconstruct_scene_by_id(scene_id, run_id, input_fingerprint):
                recovered += 1
        except Exception:
            # The claimed wrapper normally persists a terminal failure and
            # clears its lease. A process-level crash instead leaves a lease
            # that this same monitor will reclaim after TTL.
            logger.exception("Recovered reconstruction %s crashed", scene_id)
    return recovered


class ReconstructionRecoveryMonitor:
    """One bounded daemon that continuously recovers durable work."""

    def __init__(
        self,
        poll_seconds: float | None = None,
        max_workers: int | None = None,
    ) -> None:
        configured = (
            get_settings().reconstruction_recovery_poll_seconds
            if poll_seconds is None
            else poll_seconds
        )
        self.poll_seconds = max(0.05, float(configured))
        configured_workers = (
            get_settings().reconstruction_recovery_max_workers
            if max_workers is None
            else max_workers
        )
        self.max_workers = max(1, int(configured_workers))
        self._stop = Event()
        self._jobs: dict[tuple[str, str], Thread] = {}
        self._thread = Thread(
            target=self._run,
            name="reconstruction-recovery-monitor",
            daemon=True,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._jobs = {
                    key: worker
                    for key, worker in self._jobs.items()
                    if worker.is_alive()
                }
                available = self.max_workers - len(self._jobs)
                if available > 0:
                    scene_store.fail_unrecoverable_reconstruction_runs()
                    for scene_id, run_id, input_fingerprint in (
                        scene_store.list_recoverable_reconstruction_runs()
                    ):
                        key = (scene_id, run_id)
                        if key in self._jobs:
                            continue
                        worker = Thread(
                            target=self._recover_one,
                            args=(scene_id, run_id, input_fingerprint),
                            name=f"reconstruction-recovery-{scene_id}",
                            daemon=True,
                        )
                        self._jobs[key] = worker
                        worker.start()
                        available -= 1
                        if available <= 0:
                            break
            except Exception:
                logger.exception("Reconstruction recovery scan failed")
            self._stop.wait(self.poll_seconds)

    @staticmethod
    def _recover_one(scene_id: str, run_id: str, input_fingerprint: str) -> None:
        try:
            reconstruct_scene_by_id(scene_id, run_id, input_fingerprint)
        except Exception:
            logger.exception("Recovered reconstruction %s crashed", scene_id)

    def start(self) -> "ReconstructionRecoveryMonitor":
        self._thread.start()
        return self

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout=max(0.0, float(timeout)))

    def is_alive(self) -> bool:
        return self._thread.is_alive()


def start_queued_reconstruction_recovery() -> ReconstructionRecoveryMonitor:
    """Start continuous recovery without delaying API readiness."""

    return ReconstructionRecoveryMonitor().start()
