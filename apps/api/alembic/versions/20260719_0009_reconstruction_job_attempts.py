"""Add attempt accounting and a last-error column to reconstruction jobs.

A deterministically crashing child (OOM under CPU inference is realistic) used
to be rediscovered and respawned forever with no cap, no recorded failure text
and no way for the scheduler to give up. ``PipelineJobRow`` already carries
``attempts``/``error``; this brings the reconstruction control record to the
same contract.

Revision ID: 20260719_0009
Revises: 20260718_0008
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260719_0009"
down_revision = "20260718_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "reconstruction_jobs",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "reconstruction_jobs",
        sa.Column("error", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("reconstruction_jobs", "error")
    op.drop_column("reconstruction_jobs", "attempts")
