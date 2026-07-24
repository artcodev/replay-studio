from __future__ import annotations

"""Atomic reconstruction job, lease, progress, and terminal publication."""

from datetime import UTC, datetime
from typing import Any, Callable

from .analysis_run_telemetry import (
    cancel_active_analysis,
    upsert_reconstruction_telemetry,
)
from .config import get_settings
from .database_transaction import begin_write_transaction
from .database import (
    ReconstructionJobRow,
    ReconstructionLeaseRow,
    SessionLocal,
)
from .project_resource_repository import project_resources
from .reconstruction_run_contract import (
    ReconstructionRunFence,
    job_matches_fence,
    lease_matches_fence,
    queued_run_from_scene,
    reconstruction_state,
    scene_matches_fence,
    terminal_run_from_scene,
)
from .reconstruction_run_queries import (
    current_run_status_statement,
    locked_job_statement,
    locked_lease_statement,
    locked_scene_statement,
    recoverable_runs_statement,
)
from .reconstruction_run_scene_transition import (
    transition_matching_scene_to_failed,
    transition_scene_to_processing,
)
from .scene_document import (
    SceneRevisionConflict,
    next_scene_payload,
    reconstruction_input_fingerprint,
    scene_revision,
)
from .scene_index_projection import sync_scene_index


class ReconstructionRunRepository:
    def __init__(
        self,
        session_factory=None,
        *,
        clock: Callable[[], datetime | float] | None = None,
        reconstruction_lease_ttl_seconds: float | None = None,
    ) -> None:
        # Tests and separate worker processes may supply independent engines.
        # The production singleton deliberately resolves the module-level
        # factory lazily so existing isolated-store fixtures can replace it.
        self._session_factory = session_factory
        self._clock = clock
        self._configured_lease_ttl_seconds = reconstruction_lease_ttl_seconds

    def _session(self):
        factory = self._session_factory or SessionLocal
        return factory()

    def _now_timestamp(self) -> float:
        value = self._clock() if self._clock is not None else datetime.now(UTC)
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return float(value.timestamp())
        return float(value)

    def _current_time(self, value: float | datetime | None = None) -> float:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return float(value.timestamp())
        if value is None:
            return self._now_timestamp()
        return float(value)

    def _lease_ttl_seconds(self, override: float | None = None) -> float:
        value = (
            override
            if override is not None
            else self._configured_lease_ttl_seconds
            if self._configured_lease_ttl_seconds is not None
            else get_settings().reconstruction_lease_ttl_seconds
        )
        return max(1.0, float(value))

    @staticmethod
    def _write_scene(row, scene: dict[str, Any], revision: int) -> dict[str, Any]:
        persisted = next_scene_payload(scene, revision)
        row.title = scene["title"]
        row.payload = persisted
        sync_scene_index(row, persisted)
        return persisted

    @staticmethod
    def _upsert_telemetry(
        session,
        fence: ReconstructionRunFence,
        *,
        status: str,
        now: float,
        model: str | None = None,
        progress: dict[str, Any] | None = None,
        error: str | None = None,
        strict_context: bool = False,
    ) -> None:
        context = project_resources.reconstruction_context_in_transaction(
            session,
            fence.scene_id,
            strict=strict_context,
        )
        if context is None:
            return
        upsert_reconstruction_telemetry(
            session,
            project_id=context.project_id,
            scene_id=fence.scene_id,
            segment_id=context.segment_id,
            run_id=fence.run_id,
            input_fingerprint=fence.input_fingerprint,
            status=status,
            now=now,
            model=model,
            progress=progress,
            error=error,
        )

    def enqueue_reconstruction(
        self,
        scene: dict,
        *,
        expected_input_fingerprint: str,
    ) -> dict:
        """Atomically persist one explicitly prepared reconstruction run.

        Generic scene writes never create or rewrite scheduler state.  Queue
        creation therefore has a dedicated command which commits the dense
        queued read model, compact current job, and repairable AnalysisRun
        telemetry together behind the current scene revision/input fence.
        """

        queued_run = queued_run_from_scene(scene)
        fence = queued_run.fence
        expected_fingerprint = str(expected_input_fingerprint or "")
        if not expected_fingerprint:
            raise ValueError(
                "A reconstruction queue command requires the previous input fingerprint"
            )

        session = self._session()
        try:
            begin_write_transaction(session)
            job = session.scalar(locked_job_statement(fence.scene_id))
            lease = session.scalar(locked_lease_statement(fence.scene_id))
            row = session.scalar(locked_scene_statement(fence.scene_id))
            if row is None:
                session.rollback()
                raise SceneRevisionConflict(
                    f"Scene {fence.scene_id} must exist before reconstruction is queued"
                )

            current = row.payload
            current_revision = scene_revision(current)
            if scene_revision(scene) != current_revision:
                session.rollback()
                raise SceneRevisionConflict(
                    f"Scene {fence.scene_id} changed from revision "
                    f"{scene_revision(scene)} to {current_revision}"
                )
            if reconstruction_input_fingerprint(current) != expected_fingerprint:
                session.rollback()
                raise SceneRevisionConflict(
                    f"Scene {fence.scene_id} reconstruction inputs changed "
                    "while queuing"
                )

            now = self._now_timestamp()
            if lease is not None and float(lease.expires_at) > now:
                session.rollback()
                raise SceneRevisionConflict(
                    f"Scene {fence.scene_id} has an active reconstruction lease"
                )
            if lease is not None:
                session.delete(lease)

            if job is not None and job.run_id != fence.run_id:
                cancel_active_analysis(session, job.run_id, now)

            next_revision = current_revision + 1
            self._write_scene(row, scene, next_revision)
            if job is None:
                job = ReconstructionJobRow(
                    scene_id=fence.scene_id,
                    run_id=fence.run_id,
                    input_fingerprint=fence.input_fingerprint,
                    input_revision=queued_run.input_revision,
                    status="queued",
                    requested_at=now,
                    updated_at=now,
                )
                session.add(job)
            else:
                job.run_id = fence.run_id
                job.input_fingerprint = fence.input_fingerprint
                job.input_revision = queued_run.input_revision
                job.status = "queued"
                job.requested_at = now
                # An explicit new queue command is a fresh life: exhausted
                # attempts from a dead-lettered predecessor must not poison
                # it into an instant give-up on first claim.
                job.attempts = 0
                job.error = None
                job.updated_at = now
            self._upsert_telemetry(
                session,
                fence,
                status="queued",
                now=now,
                model=queued_run.model,
                progress=queued_run.progress,
                strict_context=True,
            )
            session.commit()
        except SceneRevisionConflict:
            raise
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        scene["revision"] = next_revision
        return scene

    def put_if_reconstruction_run(
        self,
        scene: dict,
        expected_run_id: str,
        expected_input_fingerprint: str,
        expected_lease_owner_id: str | None = None,
    ) -> bool:
        """Atomically publish terminal output for the exact leased job.

        Queue creation has its own command.  This compare-and-swap accepts
        terminal worker output only; AnalysisRun is written as repairable
        telemetry and never participates in the execution fence.
        """

        fence = ReconstructionRunFence(
            scene_id=str(scene.get("id") or ""),
            run_id=str(expected_run_id or ""),
            input_fingerprint=str(expected_input_fingerprint or ""),
        )
        terminal_run = terminal_run_from_scene(scene, fence)
        if terminal_run is None:
            return False
        owner_id = str(expected_lease_owner_id or "")
        if not owner_id:
            return False

        session = self._session()
        try:
            begin_write_transaction(session)
            job = session.scalar(locked_job_statement(fence.scene_id))
            if not job_matches_fence(job, fence, statuses={"processing"}):
                session.rollback()
                return False
            lease = session.scalar(locked_lease_statement(fence.scene_id))
            row = session.scalar(locked_scene_statement(fence.scene_id))
            if row is None:
                session.rollback()
                return False
            current = row.payload
            now = self._now_timestamp()
            if (
                not scene_matches_fence(current, fence, statuses={"processing"})
                or not lease_matches_fence(
                    lease,
                    fence,
                    owner_id=owner_id,
                    current_time=now,
                )
            ):
                session.rollback()
                return False
            # runId/inputFingerprint guard reconstruction inputs; revision is
            # the full-document guard that also detects unrelated concurrent
            # edits (title, output, diagnostics, queued state, etc.).
            current_revision = scene_revision(current)
            if scene_revision(scene) != current_revision:
                session.rollback()
                return False
            next_revision = current_revision + 1
            self._write_scene(row, scene, next_revision)
            session.delete(lease)
            job.status = terminal_run.scene_status
            job.updated_at = now
            self._upsert_telemetry(
                session,
                fence,
                status=terminal_run.telemetry_status,
                now=now,
                model=terminal_run.model,
                progress=terminal_run.progress,
                error=terminal_run.error,
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        scene["revision"] = next_revision
        return True

    def list_recoverable_reconstruction_runs(
        self,
        *,
        include_processing: bool = True,
        now: float | datetime | None = None,
    ) -> list[tuple[str, str, str]]:
        """List queued plus stale/missing-lease processing runs.

        Candidate discovery reads only compact control and lease columns. It
        never fetches or deserializes ``SceneRow.payload``. Every caller must
        still use the atomic claim, so concurrent runners may safely observe
        the same expired candidate.
        """

        current_time = self._current_time(now)
        statuses = ["queued", "processing"] if include_processing else ["queued"]
        with self._session() as session:
            rows = session.execute(
                recoverable_runs_statement(
                    statuses,
                    current_time=current_time,
                )
            ).all()
            return [
                (str(scene_id), str(run_id), str(input_fingerprint))
                for scene_id, run_id, input_fingerprint in rows
            ]

    def claim_reconstruction_run(
        self,
        scene_id: str,
        expected_run_id: str,
        expected_input_fingerprint: str,
        lease_owner_id: str,
        *,
        now: float | datetime | None = None,
        lease_ttl_seconds: float | None = None,
    ) -> bool:
        """Atomically claim queued work or reclaim one expired processing run."""

        fence = ReconstructionRunFence(
            scene_id=str(scene_id),
            run_id=str(expected_run_id),
            input_fingerprint=str(expected_input_fingerprint),
        )
        current_time = self._current_time(now)
        ttl = self._lease_ttl_seconds(lease_ttl_seconds)
        owner_id = str(lease_owner_id or "")
        if not owner_id:
            raise ValueError("A reconstruction claim requires a lease owner id")

        session = self._session()
        try:
            begin_write_transaction(session)
            job = session.scalar(locked_job_statement(fence.scene_id))
            if not job_matches_fence(
                job,
                fence,
                statuses={"queued", "processing"},
            ):
                session.rollback()
                return False

            existing_lease = session.scalar(
                locked_lease_statement(fence.scene_id)
            )
            if (
                existing_lease is not None
                and float(existing_lease.expires_at) > current_time
            ):
                session.rollback()
                return False

            # The dense scene is loaded once, and only after the compact claim
            # has passed. It remains the reconstruction input/read model until
            # those artifacts are split out, but it is no longer the queue.
            row = session.scalar(locked_scene_statement(fence.scene_id))
            if row is None:
                job.status = "invalid"
                job.updated_at = current_time
                if existing_lease is not None:
                    session.delete(existing_lease)
                session.commit()
                return False
            scene = row.payload
            claimed_scene = transition_scene_to_processing(
                scene,
                fence,
                current_time=current_time,
            )
            if claimed_scene is None:
                error = "Reconstruction scene state does not match its queued job"
                failed_scene = transition_matching_scene_to_failed(
                    scene,
                    fence,
                    current_time=current_time,
                    error=error,
                )
                if failed_scene is not None:
                    row.payload = failed_scene.payload
                    sync_scene_index(row, failed_scene.payload)
                    reconstruction = failed_scene.reconstruction
                else:
                    reconstruction = reconstruction_state(scene)
                job.status = "invalid"
                job.updated_at = current_time
                self._upsert_telemetry(
                    session,
                    fence,
                    status="failed",
                    now=current_time,
                    model=(
                        str(reconstruction.get("model"))
                        if reconstruction.get("model") is not None
                        else None
                    ),
                    progress=(
                        reconstruction.get("progress")
                        if isinstance(reconstruction.get("progress"), dict)
                        else None
                    ),
                    error=error,
                )
                if existing_lease is not None:
                    session.delete(existing_lease)
                session.commit()
                return False

            if existing_lease is None:
                lease = ReconstructionLeaseRow(
                    scene_id=fence.scene_id,
                    run_id=fence.run_id,
                    input_fingerprint=fence.input_fingerprint,
                    owner_id=owner_id,
                    acquired_at=current_time,
                    heartbeat_at=current_time,
                    expires_at=current_time + ttl,
                )
                session.add(lease)
            else:
                lease = existing_lease
                lease.run_id = fence.run_id
                lease.input_fingerprint = fence.input_fingerprint
                lease.owner_id = owner_id
                lease.acquired_at = current_time
                lease.heartbeat_at = current_time
                lease.expires_at = current_time + ttl

            attempts = int(getattr(job, "attempts", 0) or 0) + 1
            if attempts > max(1, int(get_settings().reconstruction_max_attempts)):
                self._dead_letter_exhausted_job(
                    session, fence, job=job, row=row, scene=scene,
                    existing_lease=existing_lease,
                    current_time=current_time, attempts=attempts,
                )
                session.commit()
                return False

            row.payload = claimed_scene.payload
            sync_scene_index(row, claimed_scene.payload)
            job.status = "processing"
            job.attempts = attempts
            job.updated_at = current_time
            self._upsert_telemetry(
                session,
                fence,
                status="running",
                now=current_time,
                model=(
                    str(claimed_scene.reconstruction.get("model"))
                    if claimed_scene.reconstruction.get("model") is not None
                    else None
                ),
                progress=(
                    claimed_scene.reconstruction.get("progress")
                    if isinstance(
                        claimed_scene.reconstruction.get("progress"), dict
                    )
                    else None
                ),
            )
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def publish_reconstruction_progress(
        self,
        scene_id: str,
        expected_run_id: str,
        expected_input_fingerprint: str,
        expected_lease_owner_id: str,
        progress: dict,
        *,
        now: float | datetime | None = None,
    ) -> str:
        """Publish compact telemetry for the exact live fenced worker.

        The return value is ``published``, ``cancelled`` or ``stale``.  This
        hot path never reads or writes ``SceneRow``; dense editor state is
        published exactly once at the terminal compare-and-swap.
        """

        fence = ReconstructionRunFence(
            scene_id=str(scene_id),
            run_id=str(expected_run_id),
            input_fingerprint=str(expected_input_fingerprint),
        )
        current_time = self._current_time(now)
        owner_id = str(expected_lease_owner_id or "")
        if not owner_id:
            raise ValueError("Reconstruction progress requires a lease owner id")

        session = self._session()
        try:
            begin_write_transaction(session)
            job = session.scalar(locked_job_statement(fence.scene_id))
            lease = session.scalar(locked_lease_statement(fence.scene_id))
            if (
                job_matches_fence(job, fence, statuses={"cancelled"})
            ):
                session.rollback()
                return "cancelled"
            if (
                not job_matches_fence(job, fence, statuses={"processing"})
                or not lease_matches_fence(
                    lease,
                    fence,
                    owner_id=owner_id,
                    current_time=current_time,
                )
            ):
                session.rollback()
                return "stale"
            job.updated_at = current_time
            self._upsert_telemetry(
                session,
                fence,
                status="running",
                now=current_time,
                progress=progress,
            )
            session.commit()
            return "published"
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _dead_letter_exhausted_job(
        self,
        session,
        fence: ReconstructionRunFence,
        *,
        job,
        row,
        scene,
        existing_lease,
        current_time: float,
        attempts: int,
    ) -> None:
        """Terminally invalidate a job whose child keeps dying.

        A deterministically crashing run must not loop forever: the job
        dead-letters with its last recorded error and the scene fails
        explicitly so the user sees why. Runs inside the claim transaction.
        """

        error = (
            f"Reconstruction gave up after {attempts - 1} attempts"
            + (f": {job.error}" if job.error else "")
        )
        failed_scene = transition_matching_scene_to_failed(
            scene,
            fence,
            current_time=current_time,
            error=error,
        )
        if failed_scene is not None:
            row.payload = failed_scene.payload
            sync_scene_index(row, failed_scene.payload)
        job.status = "invalid"
        job.error = error
        job.updated_at = current_time
        self._upsert_telemetry(
            session,
            fence,
            status="failed",
            now=current_time,
            model=None,
            progress=None,
            error=error,
        )
        if existing_lease is not None:
            session.delete(existing_lease)

    def release_crashed_reconstruction_run(
        self,
        scene_id: str,
        expected_run_id: str,
        expected_input_fingerprint: str,
        *,
        error: str,
        now: float | datetime | None = None,
    ) -> bool:
        """Free the lease of a child the supervisor saw die hard.

        The monitor already knows the process exited non-zero this poll;
        waiting out the full lease TTL only strands the job in ``processing``.
        The job stays recoverable (the claim path owns attempt accounting and
        the dead-letter cap); only the dead owner's lease is removed and the
        failure text is recorded on the authoritative record.
        """

        fence = ReconstructionRunFence(
            scene_id=str(scene_id),
            run_id=str(expected_run_id),
            input_fingerprint=str(expected_input_fingerprint),
        )
        current_time = self._current_time(now)
        session = self._session()
        try:
            begin_write_transaction(session)
            job = session.scalar(locked_job_statement(fence.scene_id))
            if not job_matches_fence(job, fence, statuses={"processing"}):
                session.rollback()
                return False
            lease = session.scalar(locked_lease_statement(fence.scene_id))
            if lease is None or str(lease.run_id) != fence.run_id:
                session.rollback()
                return False
            session.delete(lease)
            job.error = str(error)
            job.updated_at = current_time
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def reconstruction_run_is_current(
        self,
        scene_id: str,
        expected_run_id: str,
        expected_input_fingerprint: str,
        *,
        statuses: set[str] | frozenset[str] | None = None,
    ) -> bool:
        """Check child ownership using compact state only."""

        accepted = statuses or {"queued", "processing", "ready", "failed"}
        fence = ReconstructionRunFence(
            scene_id=str(scene_id),
            run_id=str(expected_run_id),
            input_fingerprint=str(expected_input_fingerprint),
        )
        with self._session() as session:
            status = session.scalar(current_run_status_statement(fence))
            return bool(status in accepted)

    def heartbeat_reconstruction_run(
        self,
        scene_id: str,
        expected_run_id: str,
        expected_input_fingerprint: str,
        expected_lease_owner_id: str,
        *,
        now: float | datetime | None = None,
        lease_ttl_seconds: float | None = None,
    ) -> bool:
        """Renew only the exact active lease without touching scene revision."""

        fence = ReconstructionRunFence(
            scene_id=str(scene_id),
            run_id=str(expected_run_id),
            input_fingerprint=str(expected_input_fingerprint),
        )
        current_time = self._current_time(now)
        ttl = self._lease_ttl_seconds(lease_ttl_seconds)

        session = self._session()
        try:
            begin_write_transaction(session)
            job = session.scalar(locked_job_statement(fence.scene_id))
            lease = session.scalar(locked_lease_statement(fence.scene_id))
            if (
                not job_matches_fence(job, fence, statuses={"processing"})
                or not lease_matches_fence(
                    lease,
                    fence,
                    owner_id=str(expected_lease_owner_id),
                    current_time=current_time,
                )
            ):
                session.rollback()
                return False
            lease.heartbeat_at = current_time
            lease.expires_at = current_time + ttl
            job.updated_at = current_time
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


reconstruction_runs = ReconstructionRunRepository()
