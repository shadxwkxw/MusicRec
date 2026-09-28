"""CLI-входная точка batch-сервиса.

Subcommands:
    extract    — извлечь фичи из директории аудио и записать в БД
    recommend  — precompute top-N похожих для всех треков, выгрузить в файл

Examples:
    python services/batch/main.py extract --input-dir /data/audio_inbox
    python services/batch/main.py recommend --output artifacts/recs.parquet --top-n 20
"""

import argparse
import asyncio
from pathlib import Path

from recommender.application.batch_extract import run_batch_extract
from recommender.application.batch_recommend import run_batch_recommend
from recommender.infrastructure.storage.postgres import async_session, init_db


async def _extract(args: argparse.Namespace) -> None:
    await init_db()
    async with async_session() as session:
        stats = await run_batch_extract(
            input_dir=Path(args.input_dir),
            db=session,
            default_artist=args.artist,
        )
    print(
        f"Batch extract done: processed={stats.processed}, "
        f"skipped={stats.skipped}, failed={len(stats.failed)}"
    )
    for name, err in stats.failed:
        print(f"  FAIL {name}: {err}")


async def _recommend(args: argparse.Namespace) -> None:
    async with async_session() as session:
        result = await run_batch_recommend(
            db=session,
            output_path=Path(args.output),
            top_n=args.top_n,
        )
    print(
        f"Batch recommend done: tracks_scored={result.tracks_scored}, output={result.output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommender batch service")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract features from a directory")
    p_extract.add_argument("--input-dir", required=True, help="Directory with audio")
    p_extract.add_argument("--artist", default="Unknown", help="Default artist if unknown")
    p_extract.set_defaults(func=_extract)

    p_recommend = sub.add_parser(
        "recommend", help="Precompute top-N recommendations for all tracks"
    )
    p_recommend.add_argument("--output", required=True, help="Output path (.csv or .parquet)")
    p_recommend.add_argument("--top-n", type=int, default=10)
    p_recommend.set_defaults(func=_recommend)

    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
