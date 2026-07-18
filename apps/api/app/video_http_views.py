def project_video_view(project_id: str, asset: dict) -> dict:
    result = dict(asset)
    ready = result.get("status") == "ready"
    result["media_url"] = (
        f"/api/projects/{project_id}/videos/{asset['id']}/media" if ready else None
    )
    result["poster_url"] = (
        f"/api/projects/{project_id}/videos/{asset['id']}/poster" if ready else None
    )
    return result
