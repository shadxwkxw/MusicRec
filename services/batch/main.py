"""CLI-входная точка batch-сервиса.

Subcommands:
    extract    — извлечь фичи из директории аудио и записать в БД
    import-fma — импортировать подмножество Free Music Archive с жанрами
    rebuild    — пересобрать индекс из всех треков БД (параметры тюнинга сохраняются)
    tune       — подобрать параметры Optuna по лайкам и пересобрать индекс
    evaluate   — оценить текущий индекс на отложенных лайках против бейзлайнов
    recommend  — precompute top-N похожих для всех треков, выгрузить в файл

rebuild и tune пишут индекс на диск; запущенный online-сервис подхватит его
после POST /index/reload.

Examples:
    python services/batch/main.py extract --input-dir /data/audio_inbox
    python services/batch/main.py import-fma --root data/fma --workers 8
    python services/batch/main.py rebuild
    python services/batch/main.py tune
    python services/batch/main.py evaluate
    python services/batch/main.py recommend --output artifacts/recs.parquet --top-n 20
"""

import argparse
import asyncio
from pathlib import Path

from recommender.application.batch_extract import (
    BatchExtractResult,
    run_batch_extract,
    run_batch_import,
)
from recommender.application.batch_recommend import run_batch_recommend
from recommender.application.index.build_index import NoTracksError, rebuild_index
from recommender.application.training.evaluation import evaluate_saved_index
from recommender.application.training.tune_recommender import (
    create_tuning_run,
    execute_tuning_run,
)
from recommender.config import settings
from recommender.infrastructure.storage.postgres import async_session, init_db


async def _extract(args: argparse.Namespace) -> None:
    await init_db()
    async with async_session() as session:
        stats = await run_batch_extract(
            input_dir=Path(args.input_dir),
            db=session,
            default_artist=args.artist,
            workers=args.workers,
        )
    _print_import_stats("Batch extract", stats)


async def _import_fma(args: argparse.Namespace) -> None:
    from recommender.infrastructure.datasets.fma import load_fma_items

    items = load_fma_items(Path(args.root), args.subset)
    if args.limit:
        items = items[: args.limit]
    print(
        f"FMA {args.subset}: {len(items)} tracks with audio, extracting with {args.workers} workers"
    )

    def progress(done: int, total: int) -> None:
        if done % 250 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    await init_db()
    async with async_session() as session:
        stats = await run_batch_import(items, session, workers=args.workers, progress=progress)
    _print_import_stats("FMA import", stats)
    print("Run `rebuild` (and `index-reload` for a running server) to make them searchable")


def _print_import_stats(title: str, stats: BatchExtractResult) -> None:
    print(
        f"{title} done: processed={stats.processed}, "
        f"skipped={stats.skipped}, failed={len(stats.failed)}"
    )
    for name, err in stats.failed:
        print(f"  FAIL {name}: {err}")


async def _rebuild(args: argparse.Namespace) -> None:
    await init_db()
    async with async_session() as session:
        try:
            result = await rebuild_index(session)
        except NoTracksError as e:
            raise SystemExit(f"Rebuild failed: {e}") from e
    print(
        f"Rebuild done: tracks_indexed={result.tracks_indexed}, "
        f"metric={result.engine.metric}, feature_dim={result.feature_dim}"
    )


async def _tune(args: argparse.Namespace) -> None:
    await init_db()
    async with async_session() as session:
        run_id = await create_tuning_run(session)
        try:
            result = await execute_tuning_run(session, run_id)
        except Exception as e:
            raise SystemExit(f"Tuning run {run_id} failed: {e}") from e
    train = ", ".join(f"{name}={value:.3f}" for name, value in result["train"].items())
    print(
        f"Tuning run {run_id} done ({result['objective']} objective): "
        f"best_score={result['best_score']:.3f} (train: {train})"
    )
    params = result["best_params"]
    print(
        f"  params: {params['metric']}, {params['norm_method']}, "
        f"boost={params['boost_weight']:.2f}, "
        + ", ".join(f"{k[2:]}={v:.2f}" for k, v in params.items() if k.startswith("w_"))
    )
    _print_report(result["holdout"], result["test_likes"])
    _print_table("Same-genre share on held-out artists", result["genre_test"], "no genre labels")


async def _evaluate(args: argparse.Namespace) -> None:
    await init_db()
    async with async_session() as session:
        try:
            report = await evaluate_saved_index(session)
        except FileNotFoundError as e:
            raise SystemExit("No saved index yet, run rebuild or tune first") from e
    _print_report(report["holdout"])
    _print_table("Same-genre share among nearest tracks", report["genre"], "no genre labels")


def _print_report(report: dict[str, dict[str, float]], test_likes: int | None = None) -> None:
    suffix = f", {test_likes} test likes" if test_likes is not None else ""
    _print_table(f"Holdout evaluation{suffix}", report, "no test likes (need users with >=2 likes)")


def _print_table(title: str, rows: dict[str, dict[str, float]], empty: str) -> None:
    if not rows:
        print(f"{title}: {empty}")
        return
    columns = list(next(iter(rows.values())))
    width = max(14, *(len(name) for name in rows))
    print(f"{title}:")
    print(f"  {'':{width}}" + "".join(f"{c:>15}" for c in columns))
    for name, metrics in rows.items():
        print(f"  {name:{width}}" + "".join(f"{metrics.get(c, 0.0):>15.3f}" for c in columns))


async def _recommend(args: argparse.Namespace) -> None:
    await init_db()
    async with async_session() as session:
        result = await run_batch_recommend(
            db=session,
            output_path=Path(args.output),
            top_n=args.top_n,
            use_likes=args.use_likes,
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
    p_extract.add_argument("--workers", type=int, default=1, help="Parallel extraction processes")
    p_extract.set_defaults(func=_extract)

    p_fma = sub.add_parser("import-fma", help="Import a Free Music Archive subset with genres")
    p_fma.add_argument("--root", default="data/fma", help="Folder with unpacked FMA archives")
    p_fma.add_argument("--subset", default="small", choices=["small", "medium", "large"])
    p_fma.add_argument("--workers", type=int, default=1, help="Parallel extraction processes")
    p_fma.add_argument("--limit", type=int, default=0, help="Import only the first N tracks")
    p_fma.set_defaults(func=_import_fma)

    p_rebuild = sub.add_parser("rebuild", help="Rebuild the index from all tracks in the DB")
    p_rebuild.set_defaults(func=_rebuild)

    p_tune = sub.add_parser("tune", help="Tune parameters on likes and rebuild the index")
    p_tune.set_defaults(func=_tune)

    p_evaluate = sub.add_parser(
        "evaluate", help="Evaluate the saved index on held-out likes against baselines"
    )
    p_evaluate.set_defaults(func=_evaluate)

    p_recommend = sub.add_parser(
        "recommend", help="Precompute top-N recommendations for all tracks"
    )
    p_recommend.add_argument("--output", required=True, help="Output path (.csv or .parquet)")
    p_recommend.add_argument("--top-n", type=int, default=settings.default_rec_limit)
    p_recommend.add_argument(
        "--use-likes", action="store_true", help="Apply co-like boost from user likes"
    )
    p_recommend.set_defaults(func=_recommend)

    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
