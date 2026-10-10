"""CLI-входная точка batch-сервиса.

Subcommands:
    extract    — извлечь фичи из директории аудио и записать в БД
    import-fma — импортировать подмножество Free Music Archive с жанрами
    import-s3  — импортировать аудио из S3 по префиксу (storage.s3_bucket)
    embed      — досчитать эмбеддинги модели features.embedding_model
    rebuild    — пересобрать индекс из всех треков БД (параметры тюнинга сохраняются)
    tune       — подобрать параметры Optuna по лайкам и пересобрать индекс
    evaluate   — оценить текущий индекс на отложенных лайках против бейзлайнов
    recommend  — precompute top-N похожих для всех треков, выгрузить в файл или S3
    migrate-s3 — перенести аудио из папки загрузок, индекс и выгрузки в S3
    upload-s3  — залить папку с новой музыкой в S3 (без дублей), дальше import-s3
    fill-genres — дописать жанр из тегов файла трекам без жанра

rebuild и tune пишут индекс на диск; запущенный online-сервис подхватит его
после POST /index/reload.

Examples:
    python services/batch/main.py extract --input-dir /data/audio_inbox
    python services/batch/main.py import-fma --root data/fma --workers 8
    python services/batch/main.py import-s3 --prefix music/ --workers 8
    python services/batch/main.py rebuild
    python services/batch/main.py tune
    python services/batch/main.py evaluate
    python services/batch/main.py recommend --output artifacts/recs.parquet --top-n 20
    python services/batch/main.py recommend --output s3://music/artifacts/recs.parquet
    python services/batch/main.py migrate-s3 --delete-local
    python services/batch/main.py upload-s3 --dir musicnew --prefix music/
"""

import argparse
import asyncio
from pathlib import Path

from recommender.application.batch_embed import count_missing_embeddings, run_batch_embed
from recommender.application.batch_extract import (
    BatchExtractResult,
    fill_genres_from_tags,
    run_batch_extract,
    run_batch_import,
    s3_import_items,
)
from recommender.application.batch_recommend import run_batch_recommend
from recommender.application.index.build_index import NoTracksError, rebuild_index
from recommender.application.migrate_s3 import (
    migrate_artifacts,
    migrate_audio,
    migrate_index,
    upload_directory,
)
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


async def _import_s3(args: argparse.Namespace) -> None:
    items = s3_import_items(args.prefix, args.artist)
    if args.limit:
        items = items[: args.limit]
    print(
        f"s3://{settings.s3_bucket}/{args.prefix}: {len(items)} audio files, "
        f"extracting with {args.workers} workers"
    )

    def progress(done: int, total: int) -> None:
        if done % 250 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    await init_db()
    async with async_session() as session:
        stats = await run_batch_import(items, session, workers=args.workers, progress=progress)
    _print_import_stats("S3 import", stats)
    print("Run `embed` (embedding mode) and `rebuild` to make them searchable")


def _print_import_stats(title: str, stats: BatchExtractResult) -> None:
    print(
        f"{title} done: processed={stats.processed}, "
        f"skipped={stats.skipped}, failed={len(stats.failed)}"
        + (f", audio paths filled={stats.paths_filled}" if stats.paths_filled else "")
    )
    for name, err in stats.failed:
        print(f"  FAIL {name}: {err}")


async def _embed(args: argparse.Namespace) -> None:
    await init_db()
    async with async_session() as session:
        missing = await count_missing_embeddings(session, settings.embedding_model)
    if not missing:
        print(f"Embed: all tracks already have {settings.embedding_model} embeddings")
        return

    try:
        from recommender.infrastructure.data_processing.embeddings import ClapEmbedder

        embedder = ClapEmbedder(
            model_name=settings.embedding_model,
            window_seconds=settings.embedding_window_seconds,
            max_windows=settings.embedding_max_windows,
            duration_limit=settings.duration_limit,
            batch_tracks=settings.embedding_batch_tracks,
            loaders=settings.embedding_loaders,
        )
    except ImportError as e:
        raise SystemExit(
            f"Embeddings need torch and transformers: make install-embeddings ({e})"
        ) from e
    print(f"Embedding {missing} tracks with {embedder.model_name} on {embedder.device}")

    def progress(done: int, total: int) -> None:
        if done % 500 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    async with async_session() as session:
        stats = await run_batch_embed(session, embedder, progress=progress)
    _print_import_stats("Embed", stats)
    if settings.feature_source != "embedding":
        print("features.source is librosa: set FEATURE_SOURCE=embedding to use them")
    print("Run `rebuild` (and `index-reload` for a running server) to make them searchable")


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
    weights = ", ".join(f"{k[2:]}={v:.2f}" for k, v in params.items() if k.startswith("w_"))
    print(
        f"  params: {params['metric']}, {params['norm_method']}, "
        f"boost={params['boost_weight']:.2f}" + (f", {weights}" if weights else "")
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
            output_path=args.output,
            top_n=args.top_n,
            use_likes=args.use_likes,
        )
    print(
        f"Batch recommend done: tracks_scored={result.tracks_scored}, output={result.output_path}"
    )


async def _migrate_s3(args: argparse.Namespace) -> None:
    if settings.storage_backend != "s3":
        raise SystemExit("Set AUDIO_STORAGE=s3 and S3_BUCKET before migrate-s3")

    def progress(done: int, total: int) -> None:
        print(f"  audio {done}/{total}", flush=True)

    await init_db()
    async with async_session() as session:
        audio = await migrate_audio(
            session, delete_local=args.delete_local, workers=args.workers, progress=progress
        )
    print(
        f"Audio: moved={audio.moved} to s3://{settings.s3_bucket}/{settings.s3_upload_prefix}, "
        f"deleted local={audio.deleted_local}, kept outside {settings.audio_dir}="
        f"{audio.outside_audio_dir}, missing={len(audio.missing)}, failed={len(audio.failed)}"
    )
    for path in audio.missing:
        print(f"  MISSING {path}")
    for path, err in audio.failed:
        print(f"  FAIL {path}: {err}")

    version = migrate_index()
    print(
        f"Index: s3://{settings.s3_bucket}/{settings.s3_index_prefix} current={version}"
        if version
        else "Index: no saved index yet"
    )

    uploaded = migrate_artifacts(Path(args.artifacts_dir))
    print(
        f"Artifacts: {len(uploaded)} files to "
        f"s3://{settings.s3_bucket}/{settings.s3_artifacts_prefix}"
    )


async def _upload_s3(args: argparse.Namespace) -> None:
    if settings.storage_backend != "s3":
        raise SystemExit("Set AUDIO_STORAGE=s3 and S3_BUCKET before upload-s3")
    result = upload_directory(Path(args.dir), args.prefix, workers=args.workers)
    print(
        f"Upload to s3://{settings.s3_bucket}/{args.prefix}: uploaded={len(result.uploaded)}, "
        f"already in bucket={len(result.duplicates)}, failed={len(result.failed)}"
    )
    for path in result.duplicates:
        print(f"  SKIP (already in bucket) {path}")
    for path, err in result.failed:
        print(f"  FAIL {path}: {err}")
    print(f"Next: import-s3 --prefix {args.prefix}, embed, rebuild")


async def _fill_genres(args: argparse.Namespace) -> None:
    def progress(done: int, total: int) -> None:
        print(f"  {done}/{total}", flush=True)

    await init_db()
    async with async_session() as session:
        result = await fill_genres_from_tags(session, args.source or None, progress=progress)
    print(
        f"Genres from tags: filled={result.filled}, no genre tag={result.no_tag}, "
        f"audio missing={len(result.missing)}"
    )
    for location in result.missing:
        print(f"  MISSING {location}")


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

    p_s3 = sub.add_parser("import-s3", help="Import audio from S3 under a prefix")
    p_s3.add_argument("--prefix", default="", help="Key prefix inside storage.s3_bucket")
    p_s3.add_argument("--artist", default="Unknown", help="Default artist")
    p_s3.add_argument("--workers", type=int, default=1, help="Parallel extraction processes")
    p_s3.add_argument("--limit", type=int, default=0, help="Import only the first N files")
    p_s3.set_defaults(func=_import_s3)

    p_embed = sub.add_parser("embed", help="Compute missing embeddings (features.embedding_model)")
    p_embed.set_defaults(func=_embed)

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
    p_recommend.add_argument(
        "--output", required=True, help="Output .csv or .parquet: a path or s3://bucket/key"
    )
    p_recommend.add_argument("--top-n", type=int, default=settings.default_rec_limit)
    p_recommend.add_argument(
        "--use-likes", action="store_true", help="Apply co-like boost from user likes"
    )
    p_recommend.set_defaults(func=_recommend)

    p_migrate = sub.add_parser(
        "migrate-s3", help="Move uploaded audio, the index and recommend outputs to S3"
    )
    p_migrate.add_argument(
        "--delete-local", action="store_true", help="Delete local audio after it is in S3"
    )
    p_migrate.add_argument("--workers", type=int, default=8, help="Parallel uploads")
    p_migrate.add_argument(
        "--artifacts-dir", default="artifacts", help="Folder with recommend outputs"
    )
    p_migrate.set_defaults(func=_migrate_s3)

    p_upload = sub.add_parser("upload-s3", help="Upload a folder of new music to S3, skip known")
    p_upload.add_argument("--dir", required=True, help="Local folder (searched recursively)")
    p_upload.add_argument("--prefix", default="music/", help="Key prefix in storage.s3_bucket")
    p_upload.add_argument("--workers", type=int, default=8, help="Parallel uploads")
    p_upload.set_defaults(func=_upload_s3)

    p_genres = sub.add_parser("fill-genres", help="Fill missing genres from audio file tags")
    p_genres.add_argument(
        "--source", action="append", help="Only these sources (upload, import, fma); repeatable"
    )
    p_genres.set_defaults(func=_fill_genres)

    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
