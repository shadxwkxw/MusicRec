"""track source: fma / upload / import

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-09 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Загрузка через API — «<uuid>_<имя>»: 8-4-4-4-12 символов и подчёркивание
UPLOAD_NAME = "________-____-____-____-____________!_%"


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("tracks", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("source", sa.String(), nullable=False, server_default="import")
        )
        batch_op.create_index(batch_op.f("ix_tracks_source"), ["source"], unique=False)
    op.execute("UPDATE tracks SET source = 'fma' WHERE filename LIKE 'fma!_%' ESCAPE '!'")
    op.execute(
        "UPDATE tracks SET source = 'upload' WHERE source = 'import' AND "
        f"(audio_path LIKE '%/uploads/%' OR filename LIKE '{UPLOAD_NAME}' ESCAPE '!')"
    )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("tracks", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_tracks_source"))
        batch_op.drop_column("source")
