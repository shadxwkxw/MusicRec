"""OpenAPI-схема сервиса в docs/openapi.json — контракт API для клиентов.

    python scripts/export_openapi.py          # записать схему
    python scripts/export_openapi.py --check  # упасть, если схема устарела (CI)

Схема строится с конфигом по умолчанию: значения из .env на неё не влияют.
"""

import difflib
import json
import os
import sys
from pathlib import Path

os.environ["ENV_FILE"] = ""  # как в тестах: без локального .env
for name in ("FEATURE_SOURCE", "API_KEY", "API_PROTECT_READS", "CORS_ORIGINS"):
    os.environ.pop(name, None)

from recommender.interfaces.online.main import app  # noqa: E402

TARGET = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"


def render() -> str:
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    fresh = render()
    if "--check" not in sys.argv:
        TARGET.parent.mkdir(exist_ok=True)
        TARGET.write_text(fresh)
        print(f"wrote {TARGET}")
        return 0
    current = TARGET.read_text() if TARGET.exists() else ""
    if current == fresh:
        print("docs/openapi.json is up to date")
        return 0
    sys.stdout.writelines(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            fresh.splitlines(keepends=True),
            "docs/openapi.json (committed)",
            "docs/openapi.json (from code)",
        )
    )
    print("\nAPI changed: run `make openapi` and commit docs/openapi.json", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
