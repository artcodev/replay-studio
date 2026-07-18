from __future__ import annotations

"""Identity correction lineage, target resolution, and immutable split anchors."""

from .reconstruction_errors import IdentityCorrectionError, ReconstructionError
from .reconstruction_identity_read_model import canonical_analysis_subjects
from .reconstruction_identity_semantics import (
    annotation_action,
    annotation_source_identity,
    observation_identifier,
    split_range,
)

def ordered_split_corrections(annotations: list[dict]) -> list[dict]:
    """Topologically order nested splits by canonical lineage.

    Range ordering alone is incorrect when a child range starts at the same
    time as its parent: the child does not exist until the parent has produced
    its ``splitCanonicalPersonId``.  Independent splits keep deterministic
    range/id ordering.
    """

    splits = [
        annotation
        for annotation in annotations
        if annotation_action(annotation) == "split" and annotation.get("id")
    ]
    by_id = {str(annotation["id"]): annotation for annotation in splits}
    producers: dict[str, str] = {}
    for annotation in splits:
        correction_id = str(annotation["id"])
        produced_id = str(annotation.get("splitCanonicalPersonId") or "").strip()
        if not produced_id:
            continue
        previous = producers.get(produced_id)
        if previous is not None and previous != correction_id:
            raise IdentityCorrectionError(
                f"Split corrections {previous} and {correction_id} produce the same canonical identity",
                correction_id=correction_id,
                action="split",
                status="conflict",
                reason="duplicate-split-identity-producer",
                source_track_id=annotation_source_identity(annotation),
                target_id=produced_id,
            )
        producers[produced_id] = correction_id

    dependencies: dict[str, set[str]] = {correction_id: set() for correction_id in by_id}
    children: dict[str, set[str]] = {correction_id: set() for correction_id in by_id}
    for correction_id, annotation in by_id.items():
        parent_id = producers.get(str(annotation_source_identity(annotation) or ""))
        if parent_id is None:
            continue
        dependencies[correction_id].add(parent_id)
        children[parent_id].add(correction_id)

    def sort_key(correction_id: str) -> tuple[float, float, str]:
        time_range = split_range(by_id[correction_id])
        start, end = time_range if time_range is not None else (float("inf"), float("inf"))
        return start, end, correction_id

    ready = sorted(
        (correction_id for correction_id, parents in dependencies.items() if not parents),
        key=sort_key,
    )
    ordered: list[dict] = []
    while ready:
        correction_id = ready.pop(0)
        ordered.append(by_id[correction_id])
        for child_id in sorted(children[correction_id], key=sort_key):
            dependencies[child_id].discard(correction_id)
            if not dependencies[child_id] and child_id not in {
                str(item["id"]) for item in ordered
            } and child_id not in ready:
                ready.append(child_id)
        ready.sort(key=sort_key)
    if len(ordered) != len(splits):
        cyclic_ids = sorted(
            correction_id for correction_id, parents in dependencies.items() if parents
        )
        correction_id = cyclic_ids[0]
        annotation = by_id[correction_id]
        raise IdentityCorrectionError(
            "Split correction lineage contains a cycle",
            correction_id=correction_id,
            action="split",
            status="conflict",
            reason="split-lineage-cycle",
            source_track_id=annotation_source_identity(annotation),
            target_id=str(annotation.get("splitCanonicalPersonId") or "") or None,
            candidates=[{"correctionId": value} for value in cyclic_ids],
        )
    return ordered


def split_target_snapshot(
    scene: dict,
    canonical_person_id: str,
    target_observation_id: str,
) -> tuple[dict, dict]:
    """Resolve a user-selected published observation exactly once.

    The snapshot, rather than a detector list position, becomes the immutable
    correction input. A later rebuild may remap its bbox conservatively, but an
    ambiguous observation ID is rejected here rather than silently choosing a
    neighbour.
    """

    subjects = [
        subject
        for subject in canonical_analysis_subjects(scene)
        if str(subject.get("canonicalPersonId") or "") == canonical_person_id
    ]
    if len(subjects) != 1:
        raise ReconstructionError("The canonical person no longer exists or is ambiguous")
    subject = subjects[0]
    matches = [
        observation
        for observation in subject.get("observations") or []
        if observation_identifier(observation) == target_observation_id
    ]
    if len(matches) != 1:
        raise ReconstructionError(
            "Split requires one immutable tracked observation; rebuild or select another frame"
        )
    observation = matches[0]
    if observation.get("frameIndex") is None or not observation.get("bbox"):
        raise ReconstructionError("The selected split observation has no detector-backed bbox")
    scene_time = observation.get("sceneTime")
    if scene_time is None:
        raise ReconstructionError("The selected split observation has no scene timestamp")
    bbox = observation["bbox"]
    snapshot = {
        "observationId": target_observation_id,
        "frameIndex": int(observation["frameIndex"]),
        "sceneTime": round(float(scene_time), 3),
        "bbox": {
            "x": round(float(bbox["x"]), 2),
            "y": round(float(bbox["y"]), 2),
            "width": round(float(bbox["width"]), 2),
            "height": round(float(bbox["height"]), 2),
        },
        "canonicalPersonId": canonical_person_id,
    }
    return subject, snapshot


def terminal_identity_target(target_id: str, annotations: dict[str, dict]) -> str:
    current = target_id
    visited: set[str] = set()
    while current in annotations and annotation_action(annotations[current]) == "merge":
        if current in visited:
            raise ReconstructionError("Identity merge graph contains a cycle")
        visited.add(current)
        next_target = annotations[current].get("mergeTargetId")
        if not next_target:
            raise ReconstructionError("A merge correction is missing its target")
        current = str(next_target)
    return current


def canonical_correction_identity_key(
    scene: dict,
    annotation_by_id: dict[str, dict],
    identifier: str | None,
) -> str | None:
    current = str(identifier or "").strip()
    if not current:
        return None
    if current in annotation_by_id:
        annotation = annotation_by_id[current]
        if annotation_action(annotation) == "merge":
            current = terminal_identity_target(
                str(annotation.get("mergeTargetId") or ""), annotation_by_id
            )
        else:
            current = str(annotation_source_identity(annotation) or current)
    subject = next(
        (
            item
            for item in canonical_analysis_subjects(scene)
            if current
            in {
                str(item.get("id") or ""),
                str(item.get("canonicalPersonId") or ""),
            }
        ),
        None,
    )
    return str(
        (subject or {}).get("canonicalPersonId")
        or (subject or {}).get("id")
        or current
    )


def correction_endpoint_ids(
    scene: dict,
    correction: dict,
    annotations: list[dict],
) -> tuple[set[str], set[str]]:
    """Return valid source and target lineage ids for split/merge undo data."""

    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }

    def expanded(*values: object) -> set[str]:
        result = {
            str(value).strip()
            for value in values
            if str(value or "").strip()
        }
        for value in list(result):
            canonical = canonical_correction_identity_key(
                scene,
                annotation_by_id,
                value,
            )
            if canonical:
                result.add(str(canonical))
        return result

    source_id = str(annotation_source_identity(correction) or "").strip()
    source_ids = expanded(source_id)
    if annotation_action(correction) == "split":
        return source_ids, expanded(correction.get("splitCanonicalPersonId"))
    if annotation_action(correction) != "merge":
        return source_ids, set()
    target_id = str(correction.get("mergeTargetId") or "").strip()
    terminal_id = terminal_identity_target(target_id, annotation_by_id)
    terminal_annotation = annotation_by_id.get(terminal_id)
    return source_ids, expanded(
        target_id,
        terminal_id,
        annotation_source_identity(terminal_annotation),
    )
