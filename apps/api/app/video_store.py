from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select

from .database import SessionLocal, VideoAssetRow


def _serialize(row: VideoAssetRow) -> dict[str, Any]:
    ready = row.status == "ready"
    return {
        "id": row.id,
        "filename": row.filename,
        "original_name": row.original_name,
        "content_type": row.content_type,
        "status": row.status,
        "stage": row.stage,
        "progress": row.progress,
        "duration": row.duration,
        "width": row.width,
        "height": row.height,
        "fps": row.fps,
        "frame_count": row.frame_count,
        "scene_id": row.scene_id,
        "media_url": f"/api/videos/{row.id}/media" if ready else None,
        "poster_url": f"/api/videos/{row.id}/poster" if ready else None,
        "error": row.error,
        "created_at": row.created_at.isoformat() if isinstance(row.created_at, datetime) else None,
    }


class VideoStore:
    def create(self, **values: Any) -> dict:
        with SessionLocal.begin() as session:
            row = VideoAssetRow(**values)
            session.add(row)
        return _serialize(row)

    def get(self, asset_id: str) -> dict | None:
        with SessionLocal() as session:
            row = session.get(VideoAssetRow, asset_id)
            return _serialize(row) if row else None

    def list(self) -> list[dict]:
        with SessionLocal() as session:
            rows = session.scalars(select(VideoAssetRow).order_by(VideoAssetRow.created_at.desc())).all()
            return [_serialize(row) for row in rows]

    def update(self, asset_id: str, **values: Any) -> dict | None:
        with SessionLocal.begin() as session:
            row = session.get(VideoAssetRow, asset_id)
            if row is None:
                return None
            for key, value in values.items():
                setattr(row, key, value)
        return _serialize(row)


video_store = VideoStore()
