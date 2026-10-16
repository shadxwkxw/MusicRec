"""REST-роуты online-сервиса. Тонкий слой поверх application use cases."""

import json
import logging
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.covers import delete_cover
from recommender.application.features import (
    IndexSourceMismatchError,
    audio_path_of,
    check_index_source,
    uses_embeddings,
)
from recommender.application.index.build_index import NoTracksError, rebuild_index
from recommender.application.recommend import (
    FeaturesNotReadyError,
    NoLikedTracksError,
    TrackNotFoundError,
    recommend_by_track,
    recommend_for_user,
)
from recommender.application.search import (
    TextEncoder,
    TextSearchNotSupportedError,
    search_tracks,
)
from recommender.application.training.tune_recommender import (
    create_tuning_run,
    execute_tuning_run,
)
from recommender.application.visibility import hidden_track_ids
from recommender.config import settings
from recommender.infrastructure.data_processing.audio import get_duration
from recommender.infrastructure.data_processing.extract import (
    extract_features,
    features_to_bytes,
)
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.data_processing.tags import read_tags
from recommender.infrastructure.data_processing.text_worker import ProcessTextEncoder
from recommender.infrastructure.storage.artifacts import (
    IndexArtifacts,
    load_current,
    update_current,
)
from recommender.infrastructure.storage.audio_store import (
    is_s3,
    parse_s3_uri,
    store_for,
    upload_store,
)
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import (
    AutoMLRunORM,
    LikeORM,
    TrackEmbeddingORM,
    TrackORM,
    async_session,
    get_db,
    utcnow,
)
from recommender.interfaces.online.observability import count_recommendations
from recommender.interfaces.online.schemas import (
    AutoMLStatusResponse,
    LikeRequest,
    LikeResponse,
    RecommendationItem,
    RecommendationResponse,
    SearchResponse,
    TrackFeaturesResponse,
    TrackPopularity,
    TrackRegister,
    TrackResponse,
    TrackUpdate,
)
from recommender.interfaces.online.security import require_api_key, require_read_access

logger = logging.getLogger(__name__)
REGISTERED_SOURCE = "app"  # треки, файлы которых принадлежат приложению
# Чтение — по ключу, только если api.protect_reads; изменения — всегда по ключу
reads = APIRouter(dependencies=[Depends(require_read_access)])
writes = APIRouter(dependencies=[Depends(require_api_key)])


def _engine(request: Request) -> FaissRecommender:
    return request.app.state.engine


def _normalizer(request: Request) -> FeatureNormalizer:
    return request.app.state.normalizer


def _compatible(artifacts: IndexArtifacts) -> bool:
    return artifacts.engine.source == settings.feature_source_id


def _update_index(request: Request, change: Callable[[IndexArtifacts], bool]) -> IndexArtifacts:
    """Изменить индекс поверх актуальной версии на диске и переключить сервис на результат.

    Если batch уже опубликовал более новую версию, изменение применяется к ней,
    а не перетирает её устаревшим индексом из памяти.
    """
    return _update_state(request.app.state, change)


def _update_state(state, change: Callable[[IndexArtifacts], bool]) -> IndexArtifacts:
    artifacts = update_current(IndexArtifacts(state.engine, state.normalizer), change)
    if _compatible(artifacts):
        state.engine, state.normalizer = artifacts.engine, artifacts.normalizer
    return artifacts


# ──────────────────────────────────────────
# Tracks
# ──────────────────────────────────────────


def _copy_limited(source: IO[bytes], target: IO[bytes], max_bytes: int) -> int | None:
    """Скопировать не больше max_bytes; None, если источник больше."""
    copied = 0
    while chunk := source.read(1024 * 1024):
        copied += len(chunk)
        if copied > max_bytes:
            return None
        target.write(chunk)
    return copied


@writes.post("/tracks/upload", response_model=TrackResponse)
async def upload_track(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    artist: str = Form(default="Unknown"),
    genre: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Загрузить аудио, извлечь фичи, сохранить в хранилище (локально или S3), добавить в индекс."""
    track_id = str(uuid.uuid4())
    # Только имя файла от клиента, без каталогов
    filename = f"{track_id}_{Path(file.filename or 'audio').name}"

    max_bytes = int(settings.api_max_upload_mb * 1024 * 1024)
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(413, f"File is larger than {settings.api_max_upload_mb:g} MB")
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
        copied = _copy_limited(file.file, tmp, max_bytes)
    local_file = Path(tmp.name)
    if copied is None:
        local_file.unlink(missing_ok=True)
        raise HTTPException(413, f"File is larger than {settings.api_max_upload_mb:g} MB")

    try:
        features = extract_features(local_file)
        duration = get_duration(local_file)
        genre = genre or read_tags(local_file).genre  # жанр из тегов файла, если не передан
    except Exception as e:
        local_file.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to extract features: {e}") from e

    try:
        audio_path = await run_in_threadpool(upload_store().save_upload, local_file, filename)
    except Exception as e:
        local_file.unlink(missing_ok=True)
        logger.exception("Failed to store uploaded audio %s", filename)
        raise HTTPException(502, f"Failed to store audio: {e}") from e

    track = TrackORM(
        id=track_id,
        title=title,
        artist=artist,
        genre=genre,
        filename=filename,
        audio_path=audio_path,
        source="upload",
        duration=duration,
        feature_vector=features_to_bytes(features),
    )
    db.add(track)
    await db.commit()

    indexed = _index_new_track(request, background, track_id, features, audio_path)

    return TrackResponse(
        id=track_id,
        title=title,
        artist=artist,
        genre=genre,
        duration=duration,
        created_at=track.created_at,
        indexed=indexed,
        source="upload",
    )


def _adder(track_id: str, vector) -> Callable[[IndexArtifacts], bool]:
    """Изменение индекса: добавить трек, если индекс того же источника и нормализатор
    уже обучен (иначе сырые признаки смешаются с нормализованными)."""

    def add(artifacts: IndexArtifacts) -> bool:
        if not _compatible(artifacts) or not artifacts.normalizer.is_fitted:
            return False
        if track_id in artifacts.engine.track_ids:
            return False
        artifacts.engine.add_tracks([track_id], artifacts.normalizer.transform(vector))
        return True

    return add


def _index_new_track(
    request: Request, background: BackgroundTasks, track_id: str, features, location: str
) -> bool:
    """Новый трек — в индекс. librosa-признаки уже посчитаны: сразу. В режиме embedding
    эмбеддинг считается в фоне после ответа (api.embed_new_tracks), иначе его досчитает
    batch embed, и трек появится после rebuild. True — трек уже в индексе."""
    if not uses_embeddings():
        return track_id in _update_index(request, _adder(track_id, features)).engine.track_ids
    if settings.api_embed_new_tracks:
        background.add_task(_embed_new_track, request.app.state, track_id, location)
    return False


def _load_audio_encoder() -> TextEncoder:
    # В отдельном процессе, как текстовая модель: torch и faiss в одном процессе
    # на macOS конфликтуют. Интерфейс тот же: encode(пути файлов) → векторы
    return ProcessTextEncoder(
        "recommender.infrastructure.data_processing.embeddings:ClapFileEncoder"
    )


def _audio_encoder(state) -> TextEncoder:
    """Модель CLAP для аудио грузится при первом новом треке."""
    if getattr(state, "audio_encoder", None) is None:
        state.audio_encoder = _load_audio_encoder()
    return state.audio_encoder


async def _embed_new_track(state, track_id: str, location: str) -> None:
    """Фоновая задача: эмбеддинг трека → track_embeddings → индекс. Сбой не теряет трек:
    его досчитает batch embed."""
    try:
        encoder = await run_in_threadpool(_audio_encoder, state)
        if encoder.model_name != settings.embedding_model:
            raise RuntimeError(f"encoder model {encoder.model_name} != {settings.embedding_model}")

        def embed():
            with store_for(location).local_copy(location) as path:
                return encoder.encode([str(path)])[0]

        vector = await run_in_threadpool(embed)
        async with async_session() as db:
            if await db.get(TrackORM, track_id) is None:
                return  # трек успели удалить
            await db.merge(
                TrackEmbeddingORM(
                    track_id=track_id,
                    model=encoder.model_name,
                    vector=vector.astype("float32").tobytes(),
                    created_at=utcnow(),
                )
            )
            await db.commit()
        await run_in_threadpool(_update_state, state, _adder(track_id, vector.copy()))
        logger.info("Embedded and indexed new track %s", track_id)
    except ImportError:
        logger.warning(
            "Track %s: no torch/transformers for embeddings, run batch embed + rebuild", track_id
        )
    except Exception:
        logger.exception("Track %s: background embedding failed, run batch embed", track_id)


@writes.post(
    "/tracks/register",
    response_model=TrackResponse,
    responses={200: {"description": "Уже зарегистрирован"}, 201: {"description": "Создан"}},
    status_code=201,
)
async def register_track(
    request: Request,
    data: TrackRegister,
    response: Response,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Зарегистрировать трек, файл которого уже лежит в S3 (его загрузило приложение).

    Сервис читает файл, считает признаки, длительность и жанр из тегов, но файлом не
    владеет: при удалении трека объект в S3 остаётся. Разрешены только объекты своего
    бакета под storage.s3_register_prefixes. Повторная регистрация того же файла
    возвращает существующий трек (200).
    """
    if settings.storage_backend != "s3":
        raise HTTPException(409, "Registering S3 files needs AUDIO_STORAGE=s3")
    try:
        bucket, key = parse_s3_uri(data.location)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    prefixes = settings.s3_register_prefixes
    if bucket != settings.s3_bucket or not any(key.startswith(p) for p in prefixes):
        allowed = ", ".join(f"s3://{settings.s3_bucket}/{p}" for p in prefixes) or "nothing"
        raise HTTPException(422, f"Location must be under {allowed}")

    existing = (
        await db.execute(select(TrackORM).where(TrackORM.filename == key))
    ).scalar_one_or_none()
    if existing is not None:
        response.status_code = 200
        return _to_response(existing, set(_engine(request).track_ids))

    def read():
        with store_for(data.location).local_copy(data.location) as path:
            return extract_features(path), get_duration(path), read_tags(path).genre

    try:
        features, duration, tag_genre = await run_in_threadpool(read)
    except FileNotFoundError as e:
        raise HTTPException(404, f"Audio not found: {data.location}") from e
    except Exception as e:
        raise HTTPException(400, f"Failed to extract features: {e}") from e

    track = TrackORM(
        id=str(uuid.uuid4()),
        title=data.title,
        artist=data.artist,
        genre=data.genre or tag_genre,
        filename=key,  # как у import-s3: повторный импорт этого ключа будет пропущен
        audio_path=data.location,
        source=REGISTERED_SOURCE,
        duration=duration,
        feature_vector=features_to_bytes(features),
    )
    db.add(track)
    await db.commit()
    indexed = _index_new_track(request, background, track.id, features, data.location)
    return _to_response(track, {track.id} if indexed else set())


def _to_response(track: TrackORM, indexed_ids: set[str]) -> TrackResponse:
    return TrackResponse(
        id=track.id,
        title=track.title,
        artist=track.artist,
        genre=track.genre,
        duration=track.duration,
        created_at=track.created_at,
        indexed=track.id in indexed_ids,
        source=track.source,
        audio_location=track.audio_path if is_s3(track.audio_path or "") else None,
        cover_location=track.cover_path or None,
    )


@reads.get("/tracks/popularity", response_model=list[TrackPopularity])
async def get_track_popularity(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(1000, ge=1, le=10000),
):
    """Треки с лайками, самые лайкаемые первыми: число лайков от всех пользователей,
    включая плейлисты. Приложение сортирует по нему свои подборки."""
    likes = func.count(LikeORM.id)
    result = await db.execute(
        select(LikeORM.track_id, likes)
        .group_by(LikeORM.track_id)
        .order_by(likes.desc(), LikeORM.track_id)
        .limit(limit)
    )
    return [TrackPopularity(track_id=track_id, likes=count) for track_id, count in result.all()]


@reads.get("/tracks", response_model=list[TrackResponse])
async def get_all_tracks(
    request: Request,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(settings.api_tracks_page_size, ge=1, le=settings.api_tracks_page_max),
    offset: int = Query(0, ge=0),
    source: str | None = Query(
        None, description="Только эти источники через запятую: upload, import, app, fma"
    ),
):
    """Каталог, сначала новые; для синхронизации — с фильтром по источнику."""
    query = select(TrackORM).order_by(TrackORM.created_at.desc(), TrackORM.id)
    if source:
        query = query.where(
            TrackORM.source.in_([s.strip() for s in source.split(",") if s.strip()])
        )
    result = await db.execute(query.limit(limit).offset(offset))
    indexed_ids = set(_engine(request).track_ids)
    tracks: Sequence[TrackORM] = result.scalars().all()
    return [_to_response(t, indexed_ids) for t in tracks]


@writes.patch("/tracks/{track_id}", response_model=TrackResponse)
async def update_track(
    request: Request,
    track_id: str,
    data: TrackUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Исправить метаданные трека. Фичи и индекс не затрагиваются."""
    track = await db.get(TrackORM, track_id)
    if not track:
        raise HTTPException(404, "Track not found")

    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(track, field, value)
    await db.commit()

    return _to_response(track, set(_engine(request).track_ids))


@writes.delete("/tracks/{track_id}", status_code=204)
async def delete_track(request: Request, track_id: str, db: AsyncSession = Depends(get_db)):
    """Удалить трек вместе с лайками, эмбеддингами, аудиофайлом и записью в индексе."""
    track = await db.get(TrackORM, track_id)
    if not track:
        raise HTTPException(404, "Track not found")

    location = audio_path_of(track)
    cover = track.cover_path
    await db.execute(delete(LikeORM).where(LikeORM.track_id == track_id))
    await db.execute(delete(TrackEmbeddingORM).where(TrackEmbeddingORM.track_id == track_id))
    await db.delete(track)
    await db.commit()
    # Аудио удаляется, только если сервис сам его загрузил, а не импортированный каталог
    try:
        await run_in_threadpool(store_for(location).delete_upload, location)
    except Exception:
        logger.exception("Track %s deleted, but its audio %s was not", track_id, location)
    try:
        await run_in_threadpool(delete_cover, cover)  # обложками каталога владеет сервис
    except Exception:
        logger.exception("Track %s deleted, but its cover %s was not", track_id, cover)

    # Публикуем новую версию, иначе после рестарта удалённый трек вернётся из индекса на диске
    _update_index(request, lambda artifacts: bool(artifacts.engine.remove_tracks({track_id})))


@reads.get("/tracks/{track_id}/features", response_model=TrackFeaturesResponse)
async def get_track_features(track_id: str, db: AsyncSession = Depends(get_db)):
    from recommender.infrastructure.data_processing.extract import bytes_to_features

    track = await db.get(TrackORM, track_id)
    if not track or not track.feature_vector:
        raise HTTPException(404, "Track not found or features not extracted")

    features = bytes_to_features(track.feature_vector)
    return TrackFeaturesResponse(
        track_id=track_id,
        dimension=len(features),
        features=features.tolist(),
    )


# ──────────────────────────────────────────
# Recommendations
# ──────────────────────────────────────────


async def _enrich(db: AsyncSession, recs) -> list[RecommendationItem]:
    items: list[RecommendationItem] = []
    for rec in recs:
        t = await db.get(TrackORM, rec.track_id)
        if t:
            items.append(
                RecommendationItem(
                    track_id=rec.track_id,
                    title=t.title,
                    artist=t.artist,
                    genre=t.genre,
                    score=round(rec.score, 4),
                )
            )
    return items


def _load_text_encoder() -> TextEncoder:
    # В отдельном процессе: torch и faiss в одном процессе на macOS конфликтуют (OpenMP)
    return ProcessTextEncoder(
        "recommender.infrastructure.data_processing.text_encoder:ClapTextEncoder",
        settings.embedding_model,
    )


def _text_encoder(request: Request) -> TextEncoder:
    """Текстовая модель грузится при первом поиске: без поиска сервис остаётся лёгким."""
    state = request.app.state
    if getattr(state, "text_encoder", None) is None:
        try:
            state.text_encoder = _load_text_encoder()
        except ImportError as e:
            raise HTTPException(
                501,
                "Text search needs torch and transformers "
                "(make install-embeddings, or ONLINE_EXTRAS=embeddings for Docker)",
            ) from e
    return state.text_encoder


@reads.get("/search", response_model=SearchResponse)
async def search(
    request: Request,
    q: str = Query(min_length=1, max_length=500, description="Описание на английском"),
    limit: int = Query(settings.default_rec_limit, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Треки по текстовому описанию: «calm acoustic folk», «energetic hip-hop beat»."""
    try:
        engine = _engine(request)
        recs = search_tracks(
            q,
            engine,
            _normalizer(request),
            _text_encoder(request),
            limit=limit,
            hidden_ids=set(await hidden_track_ids(db, engine)),
        )
    except TextSearchNotSupportedError as e:
        raise HTTPException(409, f"{e} (set FEATURE_SOURCE=embedding and rebuild)") from e
    return SearchResponse(query=q, results=await _enrich(db, recs))


@reads.get("/recommendations/{track_id}", response_model=RecommendationResponse)
async def get_recommendations(
    request: Request,
    track_id: str,
    limit: int = Query(settings.default_rec_limit, ge=1, le=settings.api_max_rec_limit),
    use_likes: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """Рекомендации по треку: сходство + коллаборативный бустинг."""
    try:
        recs = await recommend_by_track(
            track_id,
            db,
            engine=_engine(request),
            normalizer=_normalizer(request),
            limit=limit,
            use_likes=use_likes,
        )
    except TrackNotFoundError:
        raise HTTPException(404, "Track not found") from None
    except FeaturesNotReadyError:
        raise HTTPException(
            409, f"Track has no {settings.feature_source} features yet (run batch embed)"
        ) from None

    count_recommendations(request, "track", "use_likes" if use_likes else "content")
    return RecommendationResponse(
        source_track_id=track_id,
        recommendations=await _enrich(db, recs),
    )


@reads.get("/recommendations/user/{user_id}", response_model=RecommendationResponse)
async def get_user_recommendations(
    request: Request,
    user_id: str,
    limit: int = Query(settings.default_rec_limit, ge=1, le=settings.api_max_rec_limit),
    use_likes: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """Персональные рекомендации по интересам из лайков; без лайков — популярное."""
    try:
        result = await recommend_for_user(
            user_id,
            db,
            engine=_engine(request),
            normalizer=_normalizer(request),
            limit=limit,
            use_likes=use_likes,
        )
    except NoLikedTracksError:
        raise HTTPException(404, "No liked tracks found for user") from None

    count_recommendations(request, "user", result.strategy)
    return RecommendationResponse(
        source_track_id=f"user:{user_id}",
        recommendations=await _enrich(db, result.items),
        strategy=result.strategy,
    )


# ──────────────────────────────────────────
# Likes
# ──────────────────────────────────────────


def _like_query(user_id: str, track_id: str):
    return select(LikeORM).where(LikeORM.user_id == user_id, LikeORM.track_id == track_id)


@writes.post("/likes", response_model=LikeResponse)
async def add_like(data: LikeRequest, db: AsyncSession = Depends(get_db)):
    """Поставить лайк. Повторный лайк того же трека ничего не меняет (status=exists)."""
    track = await db.get(TrackORM, data.track_id)
    if not track:
        raise HTTPException(404, "Track not found")

    status = "ok"
    if (await db.execute(_like_query(data.user_id, data.track_id))).first():
        status = "exists"
    else:
        db.add(LikeORM(user_id=data.user_id, track_id=data.track_id))
        try:
            await db.commit()
        except IntegrityError:  # тот же лайк пришёл параллельным запросом
            await db.rollback()
            status = "exists"

    return LikeResponse(status=status, user_id=data.user_id, track_id=data.track_id)


@writes.delete("/likes/{user_id}/{track_id}", status_code=204)
async def remove_like(user_id: str, track_id: str, db: AsyncSession = Depends(get_db)):
    like = (await db.execute(_like_query(user_id, track_id))).scalar_one_or_none()
    if like is None:
        raise HTTPException(404, "Like not found")
    await db.delete(like)
    await db.commit()


@reads.get("/users/{user_id}/likes", response_model=list[TrackResponse])
async def get_user_likes(
    request: Request,
    user_id: str,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(settings.api_tracks_page_size, ge=1, le=settings.api_tracks_page_max),
    offset: int = Query(0, ge=0),
):
    """Лайкнутые треки пользователя, сначала последние."""
    result = await db.execute(
        select(TrackORM)
        .join(LikeORM, LikeORM.track_id == TrackORM.id)
        .where(LikeORM.user_id == user_id)
        .order_by(LikeORM.created_at.desc(), LikeORM.id.desc())
        .limit(limit)
        .offset(offset)
    )
    tracks: Sequence[TrackORM] = result.scalars().all()
    indexed_ids = set(_engine(request).track_ids)
    return [_to_response(t, indexed_ids) for t in tracks]


# ──────────────────────────────────────────
# Tuning (Optuna)
# ──────────────────────────────────────────


@writes.post("/automl/train")
async def start_tuning(
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Запустить оптимизацию в фоне; по завершении сервис переключается на новый индекс."""
    run_id = await create_tuning_run(db)
    app_state = request.app.state

    async def _train() -> None:
        async with async_session() as session:
            try:
                await execute_tuning_run(session, run_id)
            except Exception:
                logger.exception("Tuning run %s failed", run_id)  # статус failed уже записан
                return
        artifacts = load_current()
        app_state.engine, app_state.normalizer = artifacts.engine, artifacts.normalizer

    background_tasks.add_task(_train)
    return {"run_id": run_id, "status": "started"}


@reads.get("/automl/status", response_model=list[AutoMLStatusResponse])
async def get_tuning_status(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AutoMLRunORM).order_by(AutoMLRunORM.id.desc()))
    runs: Sequence[AutoMLRunORM] = result.scalars().all()
    return [
        AutoMLStatusResponse(
            id=r.id,
            status=r.status,
            best_score=r.best_score,
            best_params=json.loads(r.best_params) if r.best_params else None,
            metrics=json.loads(r.metrics) if r.metrics else None,
            n_trials=r.n_trials,
            started_at=r.started_at,
            completed_at=r.completed_at,
        )
        for r in runs
    ]


# ──────────────────────────────────────────
# Index management
# ──────────────────────────────────────────


@writes.post("/index/rebuild")
async def rebuild_index_endpoint(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        result = await rebuild_index(db)
    except NoTracksError as e:
        raise HTTPException(400, str(e)) from e

    request.app.state.engine = result.engine
    request.app.state.normalizer = result.normalizer
    return {
        "status": "ok",
        "tracks_indexed": result.tracks_indexed,
        "feature_dim": result.feature_dim,
    }


@writes.post("/index/reload")
async def reload_index(request: Request):
    """Подхватить индекс и нормализатор с диска, например после batch rebuild / tune."""
    try:
        artifacts = load_current()
        engine, normalizer = artifacts.engine, artifacts.normalizer
    except FileNotFoundError as e:
        raise HTTPException(409, "No saved index yet, run /index/rebuild first") from e
    try:
        check_index_source(engine)
    except IndexSourceMismatchError as e:
        raise HTTPException(409, str(e)) from e

    request.app.state.engine = engine
    request.app.state.normalizer = normalizer
    return {
        "status": "ok",
        "tracks_indexed": engine.index.ntotal,
        "metric": engine.metric,
        "version": engine.version,
    }


router = APIRouter()
router.include_router(reads)
router.include_router(writes)
