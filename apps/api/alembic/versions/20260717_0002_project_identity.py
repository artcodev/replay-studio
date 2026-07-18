"""Add project-level identity and scene membership persistence.

Revision ID: 20260717_0002
Revises: 20260717_0001
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260717_0002"
down_revision = "20260717_0001"
branch_labels = None
depends_on = None


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _index(name: str, table: str, columns: list[str], *, unique: bool = False) -> None:
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table)}
    if name not in indexes:
        op.create_index(name, table, columns, unique=unique)


def upgrade() -> None:
    tables = _tables()
    timestamp = sa.text("CURRENT_TIMESTAMP")
    if "project_people" not in tables:
        op.create_table(
            "project_people",
            sa.Column("id", sa.String(160), primary_key=True),
            sa.Column(
                "project_id",
                sa.String(120),
                sa.ForeignKey("projects.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("roster_person_id", sa.String(160), nullable=True),
            sa.Column("display_name", sa.String(240), nullable=False),
            sa.Column("team_id", sa.String(160), nullable=True),
            sa.Column("role", sa.String(80), nullable=True),
            sa.Column("jersey_number", sa.String(40), nullable=True),
            sa.Column("status", sa.String(40), nullable=False),
            sa.Column("identity_confidence", sa.Float(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=timestamp,
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=timestamp,
                nullable=False,
            ),
            sa.UniqueConstraint(
                "project_id",
                "roster_person_id",
                name="uq_project_person_roster_person",
            ),
        )
        _index("ix_project_people_project_id", "project_people", ["project_id"])
        _index(
            "ix_project_person_project_team",
            "project_people",
            ["project_id", "team_id"],
        )

    tables = _tables()
    if "project_person_memberships" not in tables:
        op.create_table(
            "project_person_memberships",
            sa.Column("id", sa.String(160), primary_key=True),
            sa.Column(
                "project_id",
                sa.String(120),
                sa.ForeignKey("projects.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "project_person_id",
                sa.String(160),
                sa.ForeignKey("project_people.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "scene_id",
                sa.String(120),
                sa.ForeignKey("scenes.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("scene_person_id", sa.String(160), nullable=False),
            sa.Column("assignment_source", sa.String(40), nullable=False),
            sa.Column("identity_status", sa.String(40), nullable=True),
            sa.Column("identity_confidence", sa.Float(), nullable=True),
            sa.Column("observation_count", sa.Integer(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=timestamp,
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=timestamp,
                nullable=False,
            ),
            sa.UniqueConstraint(
                "project_id",
                "scene_id",
                "scene_person_id",
                name="uq_project_person_membership_scene_identity",
            ),
        )
        _index(
            "ix_project_person_memberships_project_id",
            "project_person_memberships",
            ["project_id"],
        )
        _index(
            "ix_project_person_memberships_project_person_id",
            "project_person_memberships",
            ["project_person_id"],
        )
        _index(
            "ix_project_person_memberships_scene_id",
            "project_person_memberships",
            ["scene_id"],
        )
        _index(
            "ix_project_person_membership_person_scene",
            "project_person_memberships",
            ["project_person_id", "scene_id"],
        )


def downgrade() -> None:
    for table in ("project_person_memberships", "project_people"):
        if table in _tables():
            op.drop_table(table)

