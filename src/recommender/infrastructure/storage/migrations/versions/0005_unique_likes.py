"""unique like per user and track

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-06 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Повторные лайки одного трека оставляем в одном экземпляре (самый ранний)
    op.execute(
        "DELETE FROM likes WHERE id NOT IN "
        "(SELECT keep FROM (SELECT MIN(id) AS keep FROM likes GROUP BY user_id, track_id) AS k)"
    )
    with op.batch_alter_table("likes", schema=None) as batch_op:
        batch_op.create_unique_constraint("uq_likes_user_track", ["user_id", "track_id"])


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("likes", schema=None) as batch_op:
        batch_op.drop_constraint("uq_likes_user_track", type_="unique")
