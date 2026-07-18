from __future__ import annotations

"""Pure cleanup of annotation references in identity read-model drafts."""


def remove_annotation_references(scene: dict, annotation_ids: set[str]) -> None:
    if not annotation_ids:
        return
    payload = scene.get("payload", {})
    for subject in [
        *(payload.get("canonicalPeople") or []),
        *(payload.get("tracks") or []),
    ]:
        retained = [
            str(value)
            for value in subject.get("annotationIds") or []
            if str(value) not in annotation_ids
        ]
        if retained:
            subject["annotationIds"] = sorted(set(retained))
        else:
            subject.pop("annotationIds", None)
        for observation in subject.get("observations") or []:
            if str(observation.get("annotationId") or "") in annotation_ids:
                observation["annotationId"] = None
            observation_annotation_ids = [
                str(value)
                for value in observation.get("annotationIds") or []
                if str(value) not in annotation_ids
            ]
            if observation_annotation_ids:
                observation["annotationIds"] = sorted(
                    set(observation_annotation_ids)
                )
            else:
                observation.pop("annotationIds", None)
