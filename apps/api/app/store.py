from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from typing import Callable
from uuid import uuid4

from sqlalchemy import select, text

from .config import get_settings
from .database import ReconstructionLeaseRow, SceneRow, SessionLocal
from .project_match import (
    is_multi_pass_scene,
    project_parent_scene_id,
    semantic_match_binding,
)
from .sample import make_demo_scene


RECONSTRUCTION_INPUT_CHANGED_ERROR = (
    "Reconstruction input changed after this run was queued; "
    "start a fresh reconstruction from the current scene."
)


class SceneRevisionConflict(RuntimeError):
    """The caller tried to replace a scene snapshot that is no longer current."""


def scene_revision(scene: dict) -> int:
    """Read the backward-compatible full-document CAS revision.

    Documents written before the revision guard was introduced do not carry
    the field and therefore start at revision zero.  The first successful
    write upgrades them in place without a database migration.
    """

    try:
        return max(0, int(scene.get("revision") or 0))
    except (TypeError, ValueError):
        return 0


def reconstruction_input_fingerprint(scene: dict) -> str:
    """Return a stable digest of inputs a reconstruction job is allowed to use.

    Runtime fields such as progress, diagnostics, generated tracks, and the
    detected pitch side are deliberately excluded: the worker itself updates
    them. Manual anchors, identity corrections, attack direction, model, and
    source range are included so a late worker cannot publish over a newer
    user decision even when a stale client happened to retain the same run id.
    """

    payload = scene.get("payload", {})
    video = payload.get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    orientation = reconstruction.get("pitchOrientation") or {}
    visible_side_source = str(orientation.get("visiblePitchSideSource") or "")
    manual_visible_side = (
        orientation.get("visiblePitchSide")
        if visible_side_source.startswith("manual")
        else None
    )
    inputs = {
        "source": {
            "assetId": video.get("id"),
            "selectedSegmentId": video.get("selectedSegmentId"),
            "sourceStart": video.get("sourceStart"),
            "sourceEnd": video.get("sourceEnd"),
            "analysisFps": video.get("analysisFps"),
        },
        "model": reconstruction.get("model"),
        "ballDetection": {
            "backend": reconstruction.get("ballBackend"),
            "input": reconstruction.get("ballDetectionInput"),
        },
        "frameAnnotations": reconstruction.get("frameAnnotations") or [],
        "pitchCalibrationOverrides": reconstruction.get("pitchCalibrationOverrides") or [],
        "pitchCalibrationOverride": reconstruction.get("pitchCalibrationOverride"),
        "manualOrientation": {
            "attackingGoal": orientation.get("attackingGoal"),
            "attackingGoalSource": orientation.get("attackingGoalSource"),
            "visiblePitchSide": manual_visible_side,
            "visiblePitchSideSource": visible_side_source if manual_visible_side else None,
        },
        # Match/roster data participates in identity resolution. A run started
        # before the user binds another event must not publish roster
        # candidates from the stale match or overwrite the newer binding.
        # Project ownership metadata is storage/UI state, not an identity
        # reconstruction input. Root and child copies of the same roster must
        # therefore produce the same semantic match component.
        "matchBinding": semantic_match_binding(payload.get("matchBinding")),
        # Human review decisions constrain future hypotheses just like manual
        # frame annotations. A stale worker must not resurrect a rejected row.
        "identityReviewDecisions": payload.get("identityReviewDecisions"),
    }
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def scene_kind(scene: dict) -> str:
    video = scene.get("payload", {}).get("videoAsset") or {}
    if not video:
        return "demo"
    title = str(scene.get("title") or "").lower()
    filename = str(video.get("filename") or "").lower()
    if "smoke test" in title or "smoke" in filename:
        return "demo"
    if video.get("multiPass"):
        return "multi-pass"
    if video.get("parentSceneId") or video.get("selectedSegmentId"):
        return "segment"
    return "video"


class SceneStore:
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
    def _iso_timestamp(value: float) -> str:
        return datetime.fromtimestamp(float(value), UTC).isoformat()

    @classmethod
    def _lease_metadata(cls, lease: ReconstructionLeaseRow) -> dict:
        return {
            "ownerId": lease.owner_id,
            "runId": lease.run_id,
            "acquiredAt": cls._iso_timestamp(lease.acquired_at),
            "heartbeatAt": cls._iso_timestamp(lease.heartbeat_at),
            "expiresAt": cls._iso_timestamp(lease.expires_at),
        }

    @classmethod
    def _with_live_lease(
        cls,
        scene: dict,
        lease: ReconstructionLeaseRow | None,
    ) -> dict:
        result = deepcopy(scene)
        reconstruction = (
            result.get("payload", {})
            .get("videoAsset", {})
            .get("reconstruction")
        )
        if not isinstance(reconstruction, dict):
            return result
        if lease is None:
            reconstruction.pop("lease", None)
        else:
            reconstruction["lease"] = cls._lease_metadata(lease)
        return result

    @staticmethod
    def _legacy_run_id(scene: dict, input_fingerprint: str) -> str:
        material = f"{scene.get('id') or ''}:{input_fingerprint}".encode("utf-8")
        return f"legacy-{hashlib.sha256(material).hexdigest()[:24]}"

    @classmethod
    def _recoverable_tokens(cls, scene: dict) -> tuple[str, str] | None:
        reconstruction = (
            scene.get("payload", {})
            .get("videoAsset", {})
            .get("reconstruction", {})
        )
        computed = reconstruction_input_fingerprint(scene)
        stored_fingerprint = str(reconstruction.get("inputFingerprint") or "")
        if stored_fingerprint and stored_fingerprint != computed:
            return None
        input_fingerprint = stored_fingerprint or computed
        run_id = str(reconstruction.get("runId") or "")
        if not run_id:
            run_id = cls._legacy_run_id(scene, input_fingerprint)
        return run_id, input_fingerprint

    @staticmethod
    def _begin_atomic_write(session) -> None:
        """Start a write transaction that is exclusive across SQLite processes."""

        bind = session.get_bind()
        if bind.dialect.name == "sqlite":
            # SQLite ignores SELECT ... FOR UPDATE. BEGIN IMMEDIATE obtains the
            # database write reservation before we inspect the current JSON,
            # so compare-and-swap remains atomic across API/worker processes.
            session.execute(text("BEGIN IMMEDIATE"))
        else:
            session.begin()

    @staticmethod
    def _next_payload(scene: dict, revision: int) -> dict:
        payload = deepcopy(scene)
        payload["revision"] = revision
        reconstruction = (
            payload.get("payload", {})
            .get("videoAsset", {})
            .get("reconstruction")
        )
        if isinstance(reconstruction, dict):
            # Lease state is authoritative in ReconstructionLeaseRow and is
            # overlaid by get(); never persist a transient heartbeat snapshot
            # back into the revisioned document.
            reconstruction.pop("lease", None)
        return payload

    def seed(self) -> None:
        """Atomically seed an empty database across concurrent API processes."""

        session = self._session()
        try:
            self._begin_atomic_write(session)
            if session.scalar(select(SceneRow.id).limit(1)) is None:
                scene = make_demo_scene()
                session.add(SceneRow(id=scene["id"], title=scene["title"], payload=scene))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list(self) -> list[dict]:
        with self._session() as session:
            rows = session.scalars(select(SceneRow).order_by(SceneRow.updated_at.desc())).all()
            return [
                {
                    "id": row.id,
                    "title": row.title,
                    "duration": float(row.payload.get("duration", 0)),
                    "kind": scene_kind(row.payload),
                    "parent_scene_id": (
                        row.payload.get("payload", {}).get("videoAsset", {}).get("parentSceneId")
                    ),
                    "updated_at": row.updated_at.isoformat() if isinstance(row.updated_at, datetime) else None,
                }
                for row in rows
            ]

    def get(self, scene_id: str) -> dict | None:
        with self._session() as session:
            row = session.get(SceneRow, scene_id)
            if row is None:
                return None
            lease = session.get(ReconstructionLeaseRow, scene_id)
            return self._with_live_lease(row.payload, lease)

    def project_scenes(self, scene_id: str) -> tuple[dict | None, list[dict]]:
        """Resolve a scene to its canonical video project and all descendants.

        Parent pointers are followed rather than trusting a client-provided
        project id. Broken pointers and cycles fail closed by treating the
        requested document as a standalone project.
        """

        with self._session() as session:
            rows = session.scalars(select(SceneRow)).all()
            scenes = {
                str(row.id): self._with_live_lease(
                    row.payload,
                    session.get(ReconstructionLeaseRow, str(row.id)),
                )
                for row in rows
            }
        requested = scenes.get(str(scene_id))
        if requested is None:
            return None, []

        def root_id(candidate_id: str) -> str:
            current_id = candidate_id
            visited: set[str] = set()
            while current_id not in visited:
                visited.add(current_id)
                current = scenes.get(current_id)
                if current is None:
                    return candidate_id
                parent_id = project_parent_scene_id(current)
                if not parent_id:
                    return current_id
                if parent_id not in scenes:
                    return candidate_id
                current_id = parent_id
            return candidate_id

        resolved_root_id = root_id(str(scene_id))
        root = scenes[resolved_root_id]
        members = [
            scene
            for candidate_id, scene in scenes.items()
            if root_id(candidate_id) == resolved_root_id
        ]
        members.sort(key=lambda item: (item["id"] != resolved_root_id, item["id"]))
        return root, members

    def put_many(self, scenes: list[dict]) -> list[dict]:
        """Atomically replace several exact scene revisions.

        Match data is a project invariant. A single-scene CAS would expose a
        partially synchronized roster when a sibling changed concurrently, so
        project mutations validate and commit the full member set together.
        """

        if not scenes:
            return []
        scene_ids = [str(scene.get("id") or "") for scene in scenes]
        if any(not scene_id for scene_id in scene_ids) or len(set(scene_ids)) != len(
            scene_ids
        ):
            raise ValueError("Atomic scene writes require unique non-empty ids")

        session = self._session()
        next_revisions: dict[str, int] = {}
        try:
            self._begin_atomic_write(session)
            rows = session.scalars(
                select(SceneRow)
                .where(SceneRow.id.in_(scene_ids))
                .with_for_update()
            ).all()
            rows_by_id = {str(row.id): row for row in rows}
            missing = [scene_id for scene_id in scene_ids if scene_id not in rows_by_id]
            if missing:
                session.rollback()
                raise SceneRevisionConflict(
                    f"Scenes disappeared during atomic write: {', '.join(missing)}"
                )

            now = self._now_timestamp()
            for scene, scene_id in zip(scenes, scene_ids):
                row = rows_by_id[scene_id]
                lease = session.get(ReconstructionLeaseRow, scene_id)
                if lease is not None and float(lease.expires_at) > now:
                    session.rollback()
                    raise SceneRevisionConflict(
                        f"Scene {scene_id} has an active reconstruction lease"
                    )
                current_revision = scene_revision(row.payload)
                if scene_revision(scene) != current_revision:
                    session.rollback()
                    raise SceneRevisionConflict(
                        f"Scene {scene_id} changed from revision "
                        f"{scene_revision(scene)} to {current_revision}"
                    )

            for scene, scene_id in zip(scenes, scene_ids):
                row = rows_by_id[scene_id]
                next_revision = scene_revision(row.payload) + 1
                row.title = scene["title"]
                row.payload = self._next_payload(scene, next_revision)
                next_revisions[scene_id] = next_revision
            session.commit()
        except SceneRevisionConflict:
            raise
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        for scene in scenes:
            scene["revision"] = next_revisions[str(scene["id"])]
        return scenes

    def put(self, scene: dict) -> dict:
        """Insert a scene or atomically replace the exact revision supplied.

        Existing documents are never updated unconditionally.  This is the
        common full-document CAS used by HTTP edits as well as internal
        get-modify-put paths.  On success the caller's dictionary receives the
        incremented revision so a long-running worker can safely publish its
        next progress snapshot.
        """

        session = self._session()
        try:
            self._begin_atomic_write(session)
            row = session.scalar(
                select(SceneRow)
                .where(SceneRow.id == scene["id"])
                .with_for_update()
            )
            if row is None:
                next_revision = 1
                persisted = self._next_payload(scene, next_revision)
                session.add(
                    SceneRow(
                        id=scene["id"],
                        title=scene["title"],
                        payload=persisted,
                    )
                )
            else:
                lease = session.get(ReconstructionLeaseRow, scene["id"])
                if (
                    lease is not None
                    and float(lease.expires_at) > self._now_timestamp()
                ):
                    session.rollback()
                    raise SceneRevisionConflict(
                        f"Scene {scene['id']} has an active reconstruction lease"
                    )
                current_revision = scene_revision(row.payload)
                if scene_revision(scene) != current_revision:
                    session.rollback()
                    raise SceneRevisionConflict(
                        f"Scene {scene['id']} changed from revision "
                        f"{scene_revision(scene)} to {current_revision}"
                    )
                next_revision = current_revision + 1
                persisted = self._next_payload(scene, next_revision)
                row.title = scene["title"]
                row.payload = persisted
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
        """Publish worker state only while run, inputs, revision and lease match.

        Legacy queued CAS callers remain supported before a worker claims the
        job. Once a scene is ``processing`` (or a lease row exists), a matching
        non-expired owner token is mandatory. This fences an old worker after
        stale-lease takeover even though the run id and input digest are
        intentionally unchanged during recovery.
        """

        session = self._session()
        try:
            self._begin_atomic_write(session)
            row = session.scalar(
                select(SceneRow)
                .where(SceneRow.id == scene["id"])
                .with_for_update()
            )
            if row is None:
                session.rollback()
                return False
            current = row.payload
            current_reconstruction = (
                current.get("payload", {})
                .get("videoAsset", {})
                .get("reconstruction", {})
            )
            if str(current_reconstruction.get("runId") or "") != expected_run_id:
                session.rollback()
                return False
            if reconstruction_input_fingerprint(current) != expected_input_fingerprint:
                session.rollback()
                return False
            lease = session.get(ReconstructionLeaseRow, scene["id"])
            now = self._now_timestamp()
            processing = current_reconstruction.get("status") == "processing"
            if lease is not None or processing:
                if (
                    lease is None
                    or not expected_lease_owner_id
                    or lease.owner_id != expected_lease_owner_id
                    or lease.run_id != expected_run_id
                    or lease.input_fingerprint != expected_input_fingerprint
                    or float(lease.expires_at) <= now
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
            persisted = self._next_payload(scene, next_revision)
            persisted_reconstruction = (
                persisted.get("payload", {})
                .get("videoAsset", {})
                .get("reconstruction", {})
            )
            terminal = persisted_reconstruction.get("status") in {"ready", "failed"}
            if terminal:
                persisted_reconstruction.pop("lease", None)
                if lease is not None:
                    session.delete(lease)
            row.title = scene["title"]
            row.payload = persisted
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        scene["revision"] = next_revision
        return True

    def list_queued_reconstruction_runs(self) -> list[tuple[str, str, str]]:
        """Return queued runs, including safely upgradable legacy documents."""

        return self.list_recoverable_reconstruction_runs(include_processing=False)

    def fail_unrecoverable_reconstruction_runs(
        self,
        *,
        now: float | datetime | None = None,
    ) -> int:
        """Release orphaned jobs whose recorded input digest is corrupted.

        Such a run cannot be resumed without violating the input CAS. Leaving
        it queued/processing would also block every explicit retry forever, so
        the monitor atomically marks it failed after its lease is absent or
        expired. The user can then queue a fresh run from the current inputs.
        """

        if isinstance(now, datetime):
            if now.tzinfo is None:
                now = now.replace(tzinfo=UTC)
            current_time = float(now.timestamp())
        elif now is None:
            current_time = self._now_timestamp()
        else:
            current_time = float(now)
        failed = 0
        session = self._session()
        try:
            self._begin_atomic_write(session)
            rows = session.scalars(select(SceneRow).with_for_update()).all()
            for row in rows:
                scene = row.payload
                video = scene.get("payload", {}).get("videoAsset", {})
                reconstruction = video.get("reconstruction", {})
                if reconstruction.get("status") not in {"queued", "processing"}:
                    continue
                lease = session.get(ReconstructionLeaseRow, str(row.id))
                if lease is not None and float(lease.expires_at) > current_time:
                    continue
                if self._recoverable_tokens(scene) is not None:
                    continue
                if is_multi_pass_scene(scene):
                    # Multi-angle composites have their own analyzer and must
                    # never be consumed or failed by the single-pass monitor.
                    continue
                message = RECONSTRUCTION_INPUT_CHANGED_ERROR
                timestamp = self._iso_timestamp(current_time)
                failed_scene = deepcopy(scene)
                failed_video = failed_scene["payload"]["videoAsset"]
                failed_reconstruction = failed_video["reconstruction"]
                failed_reconstruction.update(
                    {
                        "status": "failed",
                        "processingStatus": "failed",
                        "qualityVerdict": "reject",
                        "error": message,
                        "completedAt": timestamp,
                        "progress": {
                            **(failed_reconstruction.get("progress") or {}),
                            "phase": "failed",
                            "label": "Analysis failed",
                            "detail": message,
                            "etaSeconds": 0.0,
                            "updatedAt": timestamp,
                        },
                    }
                )
                failed_video["processingState"] = "frames-ready"
                row.payload = self._next_payload(
                    failed_scene,
                    scene_revision(scene) + 1,
                )
                if lease is not None:
                    session.delete(lease)
                failed += 1
            session.commit()
            return failed
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_recoverable_reconstruction_runs(
        self,
        *,
        include_processing: bool = True,
        now: float | datetime | None = None,
    ) -> list[tuple[str, str, str]]:
        """List queued plus stale/missing-lease processing runs.

        This is only a candidate scan. Every caller must still use the atomic
        claim before analysis, so concurrent API processes may safely observe
        the same candidate. An active lease is never returned.
        """

        if isinstance(now, datetime):
            if now.tzinfo is None:
                now = now.replace(tzinfo=UTC)
            current_time = float(now.timestamp())
        elif now is None:
            current_time = self._now_timestamp()
        else:
            current_time = float(now)

        with self._session() as session:
            rows = session.scalars(
                select(SceneRow).order_by(SceneRow.updated_at.asc(), SceneRow.id.asc())
            ).all()
            runs: list[tuple[str, str, str]] = []
            for row in rows:
                scene = row.payload
                if is_multi_pass_scene(scene):
                    continue
                reconstruction = (
                    scene.get("payload", {})
                    .get("videoAsset", {})
                    .get("reconstruction", {})
                )
                status = reconstruction.get("status")
                if status != "queued" and not (
                    include_processing and status == "processing"
                ):
                    continue
                tokens = self._recoverable_tokens(scene)
                if tokens is None:
                    continue
                lease = session.get(ReconstructionLeaseRow, str(row.id))
                if lease is not None and float(lease.expires_at) > current_time:
                    continue
                run_id, input_fingerprint = tokens
                runs.append((str(row.id), run_id, input_fingerprint))
            return runs

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

        if isinstance(now, datetime):
            if now.tzinfo is None:
                now = now.replace(tzinfo=UTC)
            current_time = float(now.timestamp())
        elif now is None:
            current_time = self._now_timestamp()
        else:
            current_time = float(now)
        ttl = self._lease_ttl_seconds(lease_ttl_seconds)
        owner_id = str(lease_owner_id or "")
        if not owner_id:
            raise ValueError("A reconstruction claim requires a lease owner id")

        session = self._session()
        try:
            self._begin_atomic_write(session)
            row = session.scalar(
                select(SceneRow)
                .where(SceneRow.id == scene_id)
                .with_for_update()
            )
            if row is None:
                session.rollback()
                return False
            scene = row.payload
            if is_multi_pass_scene(scene):
                session.rollback()
                return False
            reconstruction = (
                scene.get("payload", {})
                .get("videoAsset", {})
                .get("reconstruction", {})
            )
            if reconstruction.get("status") not in {"queued", "processing"}:
                session.rollback()
                return False
            tokens = self._recoverable_tokens(scene)
            if tokens != (expected_run_id, expected_input_fingerprint):
                session.rollback()
                return False

            existing_lease = session.get(ReconstructionLeaseRow, scene_id)
            if (
                existing_lease is not None
                and float(existing_lease.expires_at) > current_time
            ):
                session.rollback()
                return False

            if existing_lease is None:
                lease = ReconstructionLeaseRow(
                    scene_id=scene_id,
                    run_id=expected_run_id,
                    input_fingerprint=expected_input_fingerprint,
                    owner_id=owner_id,
                    acquired_at=current_time,
                    heartbeat_at=current_time,
                    expires_at=current_time + ttl,
                )
                session.add(lease)
            else:
                lease = existing_lease
                lease.run_id = expected_run_id
                lease.input_fingerprint = expected_input_fingerprint
                lease.owner_id = owner_id
                lease.acquired_at = current_time
                lease.heartbeat_at = current_time
                lease.expires_at = current_time + ttl

            claimed = deepcopy(scene)
            video = claimed["payload"]["videoAsset"]
            claimed_reconstruction = video["reconstruction"]
            claimed_reconstruction["status"] = "processing"
            claimed_reconstruction["processingStatus"] = "processing"
            if not claimed_reconstruction.get("runId"):
                claimed_reconstruction["runId"] = expected_run_id
            if not claimed_reconstruction.get("runRevision"):
                claimed_reconstruction["runRevision"] = 1
            if not claimed_reconstruction.get("inputFingerprint"):
                claimed_reconstruction["inputFingerprint"] = (
                    expected_input_fingerprint
                )
            video["processingState"] = "reconstructing"
            claimed["revision"] = scene_revision(scene) + 1
            row.payload = self._next_payload(claimed, claimed["revision"])
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def claim_queued_reconstruction_run(
        self,
        scene_id: str,
        expected_run_id: str,
        expected_input_fingerprint: str,
        lease_owner_id: str | None = None,
        *,
        now: float | datetime | None = None,
        lease_ttl_seconds: float | None = None,
    ) -> bool:
        """Backward-compatible name for the lease-aware atomic claim."""

        return self.claim_reconstruction_run(
            scene_id,
            expected_run_id,
            expected_input_fingerprint,
            lease_owner_id or f"claim-{uuid4().hex}",
            now=now,
            lease_ttl_seconds=lease_ttl_seconds,
        )

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

        if isinstance(now, datetime):
            if now.tzinfo is None:
                now = now.replace(tzinfo=UTC)
            current_time = float(now.timestamp())
        elif now is None:
            current_time = self._now_timestamp()
        else:
            current_time = float(now)
        ttl = self._lease_ttl_seconds(lease_ttl_seconds)

        session = self._session()
        try:
            self._begin_atomic_write(session)
            scene_row = session.scalar(
                select(SceneRow)
                .where(SceneRow.id == scene_id)
                .with_for_update()
            )
            lease = session.scalar(
                select(ReconstructionLeaseRow)
                .where(ReconstructionLeaseRow.scene_id == scene_id)
                .with_for_update()
            )
            if scene_row is None or lease is None:
                session.rollback()
                return False
            reconstruction = (
                scene_row.payload.get("payload", {})
                .get("videoAsset", {})
                .get("reconstruction", {})
            )
            if (
                reconstruction.get("status") != "processing"
                or str(reconstruction.get("runId") or "") != expected_run_id
                or reconstruction_input_fingerprint(scene_row.payload)
                != expected_input_fingerprint
                or lease.run_id != expected_run_id
                or lease.input_fingerprint != expected_input_fingerprint
                or lease.owner_id != expected_lease_owner_id
                or float(lease.expires_at) <= current_time
            ):
                session.rollback()
                return False
            lease.heartbeat_at = current_time
            lease.expires_at = current_time + ttl
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def find_segment_scene(self, parent_scene_id: str, segment_id: str) -> dict | None:
        with self._session() as session:
            rows = session.scalars(select(SceneRow)).all()
            for row in rows:
                video = row.payload.get("payload", {}).get("videoAsset") or {}
                if video.get("parentSceneId") == parent_scene_id and video.get("selectedSegmentId") == segment_id:
                    return row.payload
        return None


scene_store = SceneStore()
