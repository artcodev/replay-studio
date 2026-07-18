"""Remove the superseded embedded Scene match snapshot.

Revision ID: 20260718_0007
Revises: 20260718_0006
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260718_0007"
down_revision = "20260718_0006"
branch_labels = None
depends_on = None


def _portable_cleanup(connection) -> None:
    """Remove the field in bounded batches on unsupported SQL dialects."""

    scenes = sa.table(
        "scenes",
        sa.column("id", sa.String(120)),
        sa.column("payload", sa.JSON()),
    )
    result = connection.execution_options(stream_results=True).execute(
        sa.select(scenes.c.id, scenes.c.payload)
    )
    for partition in result.mappings().partitions(100):
        updates: list[dict] = []
        for row in partition:
            document = row["payload"]
            payload = document.get("payload") if isinstance(document, dict) else None
            if not isinstance(payload, dict) or "matchBinding" not in payload:
                continue
            cleaned = dict(document)
            cleaned_payload = dict(payload)
            cleaned_payload.pop("matchBinding")
            cleaned["payload"] = cleaned_payload
            updates.append({"scene_id": row["id"], "scene_payload": cleaned})
        if updates:
            connection.execute(
                scenes.update()
                .where(scenes.c.id == sa.bindparam("scene_id"))
                .values(payload=sa.bindparam("scene_payload")),
                updates,
            )


def upgrade() -> None:
    """Delete the former Scene-owned match copy exactly once at cutover.

    Match data is owned by ``MatchSnapshotRow``. Runtime Scene reads and
    writes intentionally contain no compatibility sanitizer after this
    migration, so stale copies cannot survive as a dormant second source of
    truth.
    """

    connection = op.get_bind()
    if "scenes" not in sa.inspect(connection).get_table_names():
        return

    if connection.dialect.name == "postgresql":
        # The model uses generic JSON rather than JSONB. Cast for the removal
        # operator and cast back so this remains a data-only cutover.
        connection.execute(
            sa.text(
                """
                UPDATE scenes
                   SET payload = ((payload::jsonb #- '{payload,matchBinding}')::json)
                 WHERE (payload::jsonb -> 'payload') ? 'matchBinding'
                """
            )
        )
        return

    if connection.dialect.name == "sqlite":
        connection.execute(
            sa.text(
                """
                UPDATE scenes
                   SET payload = json_remove(payload, '$.payload.matchBinding')
                 WHERE json_valid(payload)
                   AND json_type(payload, '$.payload.matchBinding') IS NOT NULL
                """
            )
        )
        return

    _portable_cleanup(connection)


def downgrade() -> None:
    # Deleted embedded snapshots cannot be reconstructed faithfully and the
    # previous storage contract is intentionally retired.
    pass
