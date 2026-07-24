from __future__ import annotations

"""Lease-fenced execution of one claimed reconstruction job."""

from datetime import UTC, datetime
from threading import Event, Thread
from typing import Mapping
from uuid import uuid4

from .config import get_settings
from .project_match import reconstruction_match_snapshot_reference, snapshot_matches_reference
from .project_match_repository import project_matches
from .project_resource_repository import project_resources
from .project_match_persistence_contract import MatchSnapshotDocument
from .reconstruction_errors import ReconstructionError
from .reconstruction_progress import queued_progress
from .reconstruction_run_repository import reconstruction_runs
from .scene_document import reconstruction_input_fingerprint
from .scene_repository import scenes

class _ReconstructionLeaseHeartbeat:
    """Keep one claimed database lease alive without mutating scene revision."""

    def __init__(
        self,
        scene_id: str,
        run_id: str,
        input_fingerprint: str,
        owner_id: str,
    ) -> None:
        settings = get_settings()
        ttl = max(1.0, float(settings.reconstruction_lease_ttl_seconds))
        configured = max(
            0.05,
            float(settings.reconstruction_lease_heartbeat_seconds),
        )
        self.interval = min(configured, max(0.05, ttl / 3.0))
        self.scene_id = scene_id
        self.run_id = run_id
        self.input_fingerprint = input_fingerprint
        self.owner_id = owner_id
        self._stop = Event()
        self._thread = Thread(
            target=self._run,
            name=f"reconstruction-heartbeat-{scene_id}",
            daemon=True,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                renewed = reconstruction_runs.heartbeat_reconstruction_run(
                    self.scene_id,
                    self.run_id,
                    self.input_fingerprint,
                    self.owner_id,
                )
            except Exception:
                # A transient database busy/connection error must not turn a
                # healthy long-running analysis into an abandoned lease. Retry
                # on the next interval; expiry and every publish still fence
                # the worker if renewal never succeeds.
                continue
            if not renewed:
                return

    def __enter__(self) -> "_ReconstructionLeaseHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join(timeout=max(0.1, self.interval * 2.0))


def _mark_owned_reconstruction_crashed(
    scene_id: str,
    run_id: str,
    input_fingerprint: str,
    owner_id: str,
    error: Exception,
) -> None:
    """Best-effort terminal cleanup for failures outside pipeline handling."""

    scene = scenes.get(scene_id)
    if scene is None:
        return
    reconstruction = (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
    )
    if (
        reconstruction.get("status") != "processing"
        or str(reconstruction.get("runId") or "") != run_id
        or reconstruction_input_fingerprint(scene) != input_fingerprint
    ):
        return
    now = datetime.now(UTC).isoformat()
    message = f"Reconstruction worker crashed: {error}"
    reconstruction.update(
        {
            "status": "failed",
            "processingStatus": "failed",
            "qualityVerdict": "reject",
            "error": message,
            "completedAt": now,
            "progress": {
                **(
                    reconstruction.get("progress")
                    or queued_progress(
                        0,
                        mode=str(reconstruction.get("mode") or "full"),
                    )
                ),
                "phase": "failed",
                "label": "Analysis failed",
                "detail": message,
                "etaSeconds": 0.0,
                "updatedAt": now,
            },
        }
    )
    reconstruction_runs.put_if_reconstruction_run(
        scene,
        run_id,
        input_fingerprint,
        owner_id,
    )


def captured_match_snapshot(
    scene: Mapping[str, object],
) -> MatchSnapshotDocument | None:
    """Resolve the exact immutable match input captured when the run queued."""

    reference = reconstruction_match_snapshot_reference(scene)
    if reference is None:
        return None
    scene_id = str(scene.get("id") or "")
    project_id = project_resources.scene_owner(scene_id)
    if project_id is None:
        raise ReconstructionError(
            "The reconstruction match snapshot has no owning project"
        )
    snapshot = project_matches.get_snapshot(
        project_id,
        str(reference["id"]),
    )
    if snapshot is None or not snapshot_matches_reference(snapshot, reference):
        raise ReconstructionError(
            "The reconstruction match snapshot is unavailable or failed its hash fence"
        )
    return snapshot


def reconstruct_scene_by_id(
    scene_id: str,
    expected_run_id: str,
    expected_input_fingerprint: str,
) -> bool:
    # The compact scheduler fence is a cheap guard against delayed children
    # from already-terminal or superseded runs.  Do not duplicate the dense
    # Scene fence here: claim_reconstruction_run owns that decision under one
    # database transaction and terminally invalidates a current job whose
    # Scene has drifted.  Returning early on Scene drift leaves a poison job
    # recoverable forever and can starve every newer job when max_workers=1.
    if not reconstruction_runs.reconstruction_run_is_current(
        scene_id,
        expected_run_id,
        expected_input_fingerprint,
        statuses={"queued", "processing"},
    ):
        return False
    run_id = expected_run_id
    input_fingerprint = expected_input_fingerprint
    owner_id = f"worker-{uuid4().hex}"
    if not reconstruction_runs.claim_reconstruction_run(
        scene_id,
        run_id,
        input_fingerprint,
        owner_id,
    ):
        return False
    claimed_scene = scenes.get(scene_id)
    if claimed_scene is None:
        return False
    try:
        with _ReconstructionLeaseHeartbeat(
            scene_id,
            run_id,
            input_fingerprint,
            owner_id,
        ):
            match_snapshot = captured_match_snapshot(claimed_scene)
            mode = str(
                claimed_scene.get("payload", {})
                .get("videoAsset", {})
                .get("reconstruction", {})
                .get("mode")
                or "full"
            )
            if mode == "calibrate":
                from .reconstruction_calibration_process import calibrate_scene

                process_entry = calibrate_scene
            elif mode == "full":
                from .reconstruction import reconstruct_scene

                process_entry = reconstruct_scene
            else:
                raise ReconstructionError(
                    f"Unknown claimed reconstruction process mode {mode!r}"
                )

            process_entry(
                claimed_scene,
                expected_run_id=run_id,
                expected_input_fingerprint=input_fingerprint,
                expected_lease_owner_id=owner_id,
                match_snapshot=(
                    match_snapshot.payload if match_snapshot is not None else None
                ),
            )
    except ReconstructionError as exc:
        _mark_owned_reconstruction_crashed(
            scene_id,
            run_id,
            input_fingerprint,
            owner_id,
            exc,
        )
        return True
    except Exception as exc:
        _mark_owned_reconstruction_crashed(
            scene_id,
            run_id,
            input_fingerprint,
            owner_id,
            exc,
        )
        return True
    return True
