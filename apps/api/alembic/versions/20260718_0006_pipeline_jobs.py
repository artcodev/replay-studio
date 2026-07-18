"""Add the generic durable pipeline scheduler.

Revision ID: 20260718_0006
Revises: 20260718_0005
"""

from alembic import op
import sqlalchemy as sa


revision = "20260718_0006"
down_revision = "20260718_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    video_columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("video_assets")
    }
    if "generation_key" not in video_columns:
        op.add_column(
            "video_assets",
            sa.Column("generation_key", sa.String(120), nullable=True),
        )

    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "pipeline_jobs" not in tables:
        op.create_table(
            "pipeline_jobs",
            sa.Column("id", sa.String(160), primary_key=True),
            sa.Column(
                "project_id",
                sa.String(120),
                sa.ForeignKey("projects.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("kind", sa.String(60), nullable=False),
            sa.Column("subject_id", sa.String(160), nullable=False),
            sa.Column("status", sa.String(40), nullable=False),
            sa.Column("state", sa.JSON(), nullable=False),
            sa.Column("parameters", sa.JSON(), nullable=False),
            sa.Column("available_at", sa.Float(), nullable=False),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("requested_at", sa.Float(), nullable=False),
            sa.Column("updated_at", sa.Float(), nullable=False),
            sa.UniqueConstraint(
                "kind", "subject_id", name="uq_pipeline_job_kind_subject"
            ),
        )
        op.create_index(
            "ix_pipeline_jobs_status_available",
            "pipeline_jobs",
            ["status", "available_at"],
        )
        op.create_index(
            "ix_pipeline_jobs_project_id", "pipeline_jobs", ["project_id"]
        )
        op.create_index("ix_pipeline_jobs_kind", "pipeline_jobs", ["kind"])

    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "pipeline_job_leases" not in tables:
        op.create_table(
            "pipeline_job_leases",
            sa.Column(
                "job_id",
                sa.String(160),
                sa.ForeignKey("pipeline_jobs.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("owner_id", sa.String(120), nullable=False),
            sa.Column("token", sa.String(120), nullable=False, unique=True),
            sa.Column("acquired_at", sa.Float(), nullable=False),
            sa.Column("heartbeat_at", sa.Float(), nullable=False),
            sa.Column("expires_at", sa.Float(), nullable=False),
        )
        op.create_index(
            "ix_pipeline_job_leases_expires_at",
            "pipeline_job_leases",
            ["expires_at"],
        )


def downgrade() -> None:
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "pipeline_job_leases" in tables:
        op.drop_table("pipeline_job_leases")
    if "pipeline_jobs" in tables:
        op.drop_table("pipeline_jobs")
    video_columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("video_assets")
    }
    if "generation_key" in video_columns:
        op.drop_column("video_assets", "generation_key")
