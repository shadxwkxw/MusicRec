"""REST-роуты online-сервиса. Тонкий слой поверх application use cases."""

import datetime
import json
import shutil
import uuid

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
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.index.build_index import NoTracksError, rebuild_index
from recommender.application.recommend import (
    NoLikedTracksError,
    TrackNotFoundError,
    recommend_by_track,
    recommend_for_user,
)
from recommender.application.training.tune_recommender import run_tuning
from recommender.config import settings
from recommender.infrastructure.data_processing.extract import (
    extract_features,
    features_to_bytes,
)
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import (
    AutoMLRunORM,
    LikeORM,
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
    TrackFeaturesResponse,
    TrackResponse,
    TrackUpdate,
)

router = APIRouter()


def _engine(request: Request) -> FaissRecommender:
    return request.app.state.engine


def _normalizer(request: Request) -> FeatureNormalizer:
    return request.app.state.normalizer


# ──────────────────────────────────────────
# Tracks
# ──────────────────────────────────────────


@router.post("/tracks/upload", response_model=TrackResponse)
async def upload_track(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(...),
    artist: str = Form(default="Unknown"),
    db: AsyncSession = Depends(get_db),
):
    """Загрузить аудио, извлечь фичи, добавить в индекс."""
    track_id = str(uuid.uuid4())
    filename = f"{track_id}_{file.filename}"
    filepath = settings.audio_dir / filename

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        features = extract_features(filepath)
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to extract features: {e}") from e

    duration = librosa.get_duration(filename=str(filepath))

    track = TrackORM(
        id=track_id,
        title=title,
        artist=artist,
        filename=filename,
        duration=duration,
        feature_vector=features_to_bytes(features),
    )
    db.add(track)
    await db.commit()

    # Инкрементальное добавление в индекс возможно только если нормализатор
    # уже обучен — иначе сырые фичи смешаются с нормализованными и испортят
    # косинусное сходство. До первого /index/rebuild трек живёт в БД и
    # станет searchable после ребилда.
    engine = _engine(request)
    normalizer = _normalizer(request)
    indexed = normalizer.is_fitted
    if indexed:
        norm_features = normalizer.transform(features)
        engine.add_tracks([track_id], norm_features)
        engine.save()

    return TrackResponse(
        id=track_id,
        title=title,
        artist=artist,
        duration=duration,
        created_at=track.created_at,
        indexed=indexed,
    )


def _to_response(track: TrackORM, indexed_ids: set[str]) -> TrackResponse:
    return TrackResponse(
        id=track.id,
        title=track.title,
        artist=track.artist,
        duration=track.duration,
        created_at=track.created_at,
        indexed=track.id in indexed_ids,
    )


@router.get("/tracks", response_model=list[TrackResponse])
async def get_all_tracks(
    request: Request,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    result = await db.execute(
        select(TrackORM).order_by(TrackORM.created_at.desc()).limit(limit).offset(offset)
    )
    indexed_ids = set(_engine(request).track_ids)
    return [_to_response(t, indexed_ids) for t in result.scalars().all()]


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
    """Удалить трек вместе с его лайками, аудиофайлом и записью в индексе."""
    track = await db.get(TrackORM, track_id)
    if not track:
        raise HTTPException(404, "Track not found")

    filepath = settings.audio_dir / track.filename
    await db.execute(delete(LikeORM).where(LikeORM.track_id == track_id))
    await db.delete(track)
    await db.commit()
    filepath.unlink(missing_ok=True)

    engine = _engine(request)
    if engine.remove_tracks({track_id}):
        # Сохраняем, иначе после рестарта удалённый трек вернётся в индекс с диска.
        engine.save()


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
                    score=round(rec.score, 4),
                )
            )
    return items


@router.get("/recommendations/{track_id}", response_model=RecommendationResponse)
async def get_recommendations(
    request: Request,
    track_id: str,
    limit: int = 10,
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

    return RecommendationResponse(
        source_track_id=track_id,
        recommendations=await _enrich(db, recs),
    )


@router.get("/recommendations/user/{user_id}", response_model=RecommendationResponse)
async def get_user_recommendations(
    request: Request,
    user_id: str,
    limit: int = 10,
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
    """Запустить оптимизацию в фоне."""
    run = AutoMLRunORM(status="pending")
    db.add(run)
    await db.commit()
    await db.refresh(run)

    app_state = request.app.state

    async def _train(run_id: int):
        async with async_session() as session:
            try:
                await run_tuning(session, run_id)
                # Перезагрузить движок/нормализатор с новыми артефактами
                app_state.engine = FaissRecommender.load()
                app_state.normalizer = FeatureNormalizer.load()
            except Exception as e:
                r = await session.get(AutoMLRunORM, run_id)
                r.status = "failed"
                r.best_params = json.dumps({"error": str(e)})
                r.completed_at = datetime.datetime.utcnow()
                await session.commit()

    background_tasks.add_task(_train, run.id)
    return {"run_id": run.id, "status": "started"}


@router.get("/automl/status", response_model=list[AutoMLStatusResponse])
async def get_tuning_status(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AutoMLRunORM).order_by(AutoMLRunORM.id.desc()))
    runs = result.scalars().all()
    return [
        AutoMLStatusResponse(
            id=r.id,
            status=r.status,
            best_score=r.best_score,
            best_params=json.loads(r.best_params) if r.best_params else None,
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
