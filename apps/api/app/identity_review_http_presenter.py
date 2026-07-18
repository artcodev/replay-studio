"""HTTP link presentation for the transport-neutral identity-review read model."""

from __future__ import annotations

from copy import deepcopy
from urllib.parse import quote


def identity_observation_crop_url(
    project_id: str,
    scene_id: str,
    observation_id: str,
) -> str:
    return (
        f"/api/projects/{quote(project_id, safe='')}/scenes/"
        f"{quote(scene_id, safe='')}/identity-observations/"
        f"{quote(observation_id, safe='')}/crop"
    )


def present_identity_review(
    review: dict,
    *,
    project_id: str,
    scene_id: str,
) -> dict:
    result = deepcopy(review)
    for item in result.get("items") or []:
        for observation in item.get("representativeObservations") or []:
            observation["cropUrl"] = identity_observation_crop_url(
                project_id,
                scene_id,
                str(observation["observationId"]),
            )
    return result


__all__ = ("identity_observation_crop_url", "present_identity_review")
