"""REST-роуты online-сервиса. Тонкий слой поверх application use cases."""

import json
import logging
import shutil
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import librosa
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

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
from recommender.config import settings
from recommender.infrastructure.data_processing.extract import (
    extract_features,
    features_to_bytes,
)
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.data_processing.text_worker import ProcessTextEncoder
from recommender.infrastructure.storage.artifacts import (
    IndexArtifacts,
    load_current,
    update_current,
)
from recommender.infrastructure.storage.audio_store import store_for, upload_store
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import (
    AutoMLRunORM,
    LikeORM,
    TrackEmbeddingORM,
    TrackORM,
    async_session,
    get_db,
)
from recommender.interfaces.online.schemas import (
    AutoMLStatusResponse,
    LikeRequest,
    LikeResponse,
    RecommendationItem,
    RecommendationResponse,
    SearchResponse,
    TrackFeaturesResponse,
    TrackResponse,
    TrackUpdate,
)

logger = logging.getLogger(__name__)
router = APIRouter()


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
    state = request.app.state
    artifacts = update_current(IndexArtifacts(state.engine, state.normalizer), change)
    if _compatible(artifacts):
        state.engine, state.normalizer = artifacts.engine, artifacts.normalizer
    return artifacts


# ──────────────────────────────────────────
# Tracks
# ──────────────────────────────────────────


@router.post("/tracks/upload", response_model=TrackResponse)
async def upload_track(
    request: Request,
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

    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
    local_file = Path(tmp.name)

    try:
        features = extract_features(local_file)
        duration = librosa.get_duration(path=str(local_file))
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
        duration=duration,
        feature_vector=features_to_bytes(features),
    )
    db.add(track)
    await db.commit()

    # Сразу в индекс — только если индекс на librosa-признаках и нормализатор
    # уже обучен (иначе сырые признаки смешаются с нормализованными). В режиме
    # embedding эмбеддинг досчитает batch embed, трек появится после rebuild.
    indexed = False
    if not uses_embeddings():

        def add(artifacts: IndexArtifacts) -> bool:
            if not _compatible(artifacts) or not artifacts.normalizer.is_fitted:
                return False
            artifacts.engine.add_tracks([track_id], artifacts.normalizer.transform(features))
            return True

        indexed = track_id in _update_index(request, add).engine.track_ids

    return TrackResponse(
        id=track_id,
        title=title,
        artist=artist,
        genre=genre,
        duration=duration,
        created_at=track.created_at,
        indexed=indexed,
    )


def _to_response(track: TrackORM, indexed_ids: set[str]) -> TrackResponse:
    return TrackResponse(
        id=track.id,
        title=track.title,
        artist=track.artist,
        genre=track.genre,
        duration=track.duration,
        created_at=track.created_at,
        indexed=track.id in indexed_ids,
    )


@router.get("/tracks", response_model=list[TrackResponse])
async def get_all_tracks(
    request: Request,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(settings.api_tracks_page_size, ge=1, le=settings.api_tracks_page_max),
    offset: int = Query(0, ge=0),
):
    result = await db.execute(
        select(TrackORM).order_by(TrackORM.created_at.desc()).limit(limit).offset(offset)
    )
    indexed_ids = set(_engine(request).track_ids)
    tracks: Sequence[TrackORM] = result.scalars().all()
    return [_to_response(t, indexed_ids) for t in tracks]


@router.patch("/tracks/{track_id}", response_model=TrackResponse)
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


@router.delete("/tracks/{track_id}", status_code=204)
async def delete_track(request: Request, track_id: str, db: AsyncSession = Depends(get_db)):
    """Удалить трек вместе с лайками, эмбеддингами, аудиофайлом и записью в индексе."""
    track = await db.get(TrackORM, track_id)
    if not track:
        raise HTTPException(404, "Track not found")

    location = audio_path_of(track)
    await db.execute(delete(LikeORM).where(LikeORM.track_id == track_id))
    await db.execute(delete(TrackEmbeddingORM).where(TrackEmbeddingORM.track_id == track_id))
    await db.delete(track)
    await db.commit()
    # Аудио удаляется, только если сервис сам его загрузил, а не импортированный каталог
    try:
        await run_in_threadpool(store_for(location).delete_upload, location)
    except Exception:
        logger.exception("Track %s deleted, but its audio %s was not", track_id, location)

    # Публикуем новую версию, иначе после рестарта удалённый трек вернётся из индекса на диске
    _update_index(request, lambda artifacts: bool(artifacts.engine.remove_tracks({track_id})))


@router.get("/tracks/{track_id}/features", response_model=TrackFeaturesResponse)
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


@router.get("/search", response_model=SearchResponse)
async def search(
    request: Request,
    q: str = Query(min_length=1, max_length=500, description="Описание на английском"),
    limit: int = Query(settings.default_rec_limit, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Треки по текстовому описанию: «calm acoustic folk», «energetic hip-hop beat»."""
    try:
        recs = search_tracks(
            q, _engine(request), _normalizer(request), _text_encoder(request), limit=limit
        )
    except TextSearchNotSupportedError as e:
        raise HTTPException(409, f"{e} (set FEATURE_SOURCE=embedding and rebuild)") from e
    return SearchResponse(query=q, results=await _enrich(db, recs))


@router.get("/recommendations/{track_id}", response_model=RecommendationResponse)
async def get_recommendations(
    request: Request,
    track_id: str,
    limit: int = settings.default_rec_limit,
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

    return RecommendationResponse(
        source_track_id=track_id,
        recommendations=await _enrich(db, recs),
    )


@router.get("/recommendations/user/{user_id}", response_model=RecommendationResponse)
async def get_user_recommendations(
    request: Request,
    user_id: str,
    limit: int = settings.default_rec_limit,
    use_likes: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """Персональные рекомендации по лайкам пользователя."""
    try:
        recs = await recommend_for_user(
            user_id,
            db,
            engine=_engine(request),
            normalizer=_normalizer(request),
            limit=limit,
            use_likes=use_likes,
        )
    except NoLikedTracksError:
        raise HTTPException(404, "No liked tracks found for user") from None

    return RecommendationResponse(
        source_track_id=f"user:{user_id}",
        recommendations=await _enrich(db, recs),
    )


# ──────────────────────────────────────────
# Likes
# ──────────────────────────────────────────


@router.post("/likes", response_model=LikeResponse)
async def add_like(data: LikeRequest, db: AsyncSession = Depends(get_db)):
    track = await db.get(TrackORM, data.track_id)
    if not track:
        raise HTTPException(404, "Track not found")

    like = LikeORM(user_id=data.user_id, track_id=data.track_id)
    db.add(like)
    await db.commit()

    return LikeResponse(status="ok", user_id=data.user_id, track_id=data.track_id)


# ──────────────────────────────────────────
# Tuning (Optuna)
# ──────────────────────────────────────────


@router.post("/automl/train")
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


@router.get("/automl/status", response_model=list[AutoMLStatusResponse])
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


@router.post("/index/rebuild")
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


@router.post("/index/reload")
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
