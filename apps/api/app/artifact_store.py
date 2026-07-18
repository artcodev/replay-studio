"""Backend-neutral immutable JSON artifact storage."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Protocol

from .config import get_settings


class ReconstructionArtifactError(RuntimeError):
    """An artifact could not be published or verified."""


class ArtifactStore(Protocol):
    def put_json(
        self,
        *,
        kind: str,
        schema_version: int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def get_json(
        self,
        reference: Mapping[str, Any],
        *,
        expected_kind: str,
        expected_schema_version: int,
    ) -> dict[str, Any]: ...


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReconstructionArtifactError(
            f"Artifact payload is not canonical JSON: {exc}"
        ) from exc


def digest_from_reference(reference: Mapping[str, Any]) -> str:
    digest = str(reference.get("sha256") or "").lower()
    artifact_id = str(reference.get("id") or "")
    uri = str(reference.get("uri") or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ReconstructionArtifactError(
            "Artifact reference has an invalid SHA-256 digest"
        )
    if artifact_id != f"sha256:{digest}":
        raise ReconstructionArtifactError(
            "Artifact id does not match its SHA-256 digest"
        )
    if uri != f"artifact://sha256/{digest}":
        raise ReconstructionArtifactError(
            "Artifact URI does not match its SHA-256 digest"
        )
    return digest


@dataclass(frozen=True)
class FilesystemArtifactStore:
    """Atomic local storage using the same content keys as an object store."""

    root: Path

    def _path(self, digest: str) -> Path:
        return self.root / "sha256" / digest[:2] / f"{digest}.json"

    def put_json(
        self,
        *,
        kind: str,
        schema_version: int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not kind.strip() or schema_version < 1:
            raise ReconstructionArtifactError(
                "Artifact kind and schema version are required"
            )
        envelope = {
            "kind": kind,
            "schemaVersion": schema_version,
            "payload": dict(payload),
        }
        content = canonical_json_bytes(envelope)
        digest = sha256(content).hexdigest()
        target = self._path(digest)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            try:
                existing = target.read_bytes()
            except OSError as exc:
                raise ReconstructionArtifactError(
                    f"Existing artifact cannot be read: {exc}"
                ) from exc
            if existing != content:
                raise ReconstructionArtifactError(
                    "Content-addressed artifact path contains different bytes"
                )
        else:
            temporary_path: Path | None = None
            try:
                with NamedTemporaryFile(
                    mode="wb",
                    dir=target.parent,
                    prefix=f".{digest}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    temporary.write(content)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_path, target)
                temporary_path = None
            except OSError as exc:
                raise ReconstructionArtifactError(
                    f"Artifact could not be published atomically: {exc}"
                ) from exc
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        return {
            "id": f"sha256:{digest}",
            "kind": kind,
            "schemaVersion": schema_version,
            "uri": f"artifact://sha256/{digest}",
            "sha256": digest,
            "byteSize": len(content),
            "contentType": "application/json",
        }

    def get_json(
        self,
        reference: Mapping[str, Any],
        *,
        expected_kind: str,
        expected_schema_version: int,
    ) -> dict[str, Any]:
        digest = digest_from_reference(reference)
        if reference.get("kind") != expected_kind:
            raise ReconstructionArtifactError(
                "Artifact kind does not match the consumer contract"
            )
        if reference.get("schemaVersion") != expected_schema_version:
            raise ReconstructionArtifactError(
                "Artifact schema version does not match the consumer contract"
            )
        target = self._path(digest)
        try:
            content = target.read_bytes()
        except FileNotFoundError as exc:
            raise ReconstructionArtifactError("Referenced artifact is missing") from exc
        except OSError as exc:
            raise ReconstructionArtifactError(
                f"Referenced artifact cannot be read: {exc}"
            ) from exc
        if sha256(content).hexdigest() != digest:
            raise ReconstructionArtifactError(
                "Referenced artifact failed checksum validation"
            )
        if reference.get("byteSize") != len(content):
            raise ReconstructionArtifactError(
                "Referenced artifact size does not match its manifest"
            )
        try:
            envelope = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReconstructionArtifactError(
                "Referenced artifact is not valid JSON"
            ) from exc
        if not isinstance(envelope, dict):
            raise ReconstructionArtifactError(
                "Artifact envelope must be a JSON object"
            )
        if envelope.get("kind") != expected_kind:
            raise ReconstructionArtifactError("Artifact envelope has the wrong kind")
        if envelope.get("schemaVersion") != expected_schema_version:
            raise ReconstructionArtifactError(
                "Artifact envelope has the wrong schema version"
            )
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise ReconstructionArtifactError("Artifact payload must be a JSON object")
        return payload


def reconstruction_artifact_store() -> FilesystemArtifactStore:
    return FilesystemArtifactStore(
        Path(get_settings().media_root).resolve() / "artifacts"
    )
