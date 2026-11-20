"""add append-only playback events

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-21 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "playback_events",
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("track_id", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("played_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("context", sa.String(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["track_id"], ["tracks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index("ix_playback_events_user_id", "playback_events", ["user_id"])
    op.create_index(
        "ix_playback_events_user_created", "playback_events", ["user_id", "created_at"]
    )
    op.create_index(
        "ix_playback_events_track_created", "playback_events", ["track_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_playback_events_track_created", table_name="playback_events")
    op.drop_index("ix_playback_events_user_created", table_name="playback_events")
    op.drop_index("ix_playback_events_user_id", table_name="playback_events")
    op.drop_table("playback_events")
