"""Use case: гиперпараметрическая оптимизация рекомендера через Optuna.

Подбирается:
- метод нормализации (standard / minmax / robust) и метрика (cosine / euclidean);
- групповые веса признаков (mfcc / chroma / contrast / ...);
- вес коллаборативного бустинга.

Цель подбора параметров признаков (tuning.objective):
- genre — доля того же жанра среди k ближайших с artist filter. Артисты
  делятся на подбор и проверку (tuning.test_fraction), индекс в каждой попытке
  строится только из треков подбора. Вес буста жанрами не оценить, поэтому
  он подбирается вторым шагом по лайкам при найденных параметрах признаков
  (или берётся из конфига, если лайков нет);
- likes — leave-one-out по обучающим лайкам (среднее MRR@k по двум
  прод-путям), буст подбирается вместе с остальным;
- auto — genre, если в каталоге хватает треков с жанром, иначе likes.

В обоих режимах лайки делятся на обучающие и отложенные, и в отчёт идёт оценка
на отложенных вместе с бейзлайнами. Итог: индекс и нормализатор пересобраны с
лучшими параметрами и сохранены на диск.
"""

import json
from collections.abc import Sequence

import numpy as np
import optuna
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.collaborative import load_user_likes
from recommender.application.training.evaluation import (
    evaluate,
    genre_report,
    holdout_report,
    leave_one_out_queries,
    objective_score,
    split_by_artist,
    split_likes,
)
from recommender.config import settings
from recommender.infrastructure.data_processing.extract import bytes_to_features
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import AutoMLRunORM, TrackORM, utcnow

# Индексы групп признаков в 82-мерном векторе
FEATURE_GROUPS = {
    "mfcc": (0, 26),
    "chroma": (26, 50),
    "contrast": (50, 64),
    "tonnetz": (64, 76),
    "tempo": (76, 77),
    "rms": (77, 78),
    "zcr": (78, 79),
    "spectral_stats": (79, 82),
}


def feature_weight_vector(weights: dict[str, float]) -> np.ndarray:
    """Развернуть групповые веса в per-dimension вектор."""
    dim = max(end for _, end in FEATURE_GROUPS.values())
    vec = np.ones(dim, dtype=np.float32)
    for group_name, (start, end) in FEATURE_GROUPS.items():
        vec[start:end] = weights.get(group_name, 1.0)
    return vec


def _objective_mode(genres: dict[str, str]) -> str:
    if settings.tuning_objective != "auto":
        return settings.tuning_objective
    enough = len(genres) >= settings.tuning_min_genre_tracks and len(set(genres.values())) >= 2
    return "genre" if enough else "likes"


async def run_tuning(db: AsyncSession, run_id: int) -> dict:
    """Подобрать параметры, оценить на отложенных данных, пересобрать индекс."""
    result = await db.execute(select(TrackORM).where(TrackORM.feature_vector.isnot(None)))
    tracks: Sequence[TrackORM] = result.scalars().all()

    if len(tracks) < 5:
        raise ValueError("Need at least 5 tracks with features to run tuning")

    track_ids = [t.id for t in tracks]
    raw_features = np.array([bytes_to_features(t.feature_vector) for t in tracks])
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    artists = {t.id: t.artist or "" for t in tracks}
    genres = {t.id: t.genre for t in tracks if t.genre}
    k = settings.tuning_eval_k

    train, test = split_likes(
        await load_user_likes(db), settings.tuning_test_fraction, settings.tuning_seed
    )
    has_likes = bool(leave_one_out_queries(train, set(track_ids)))
    mode = _objective_mode(genres)
    if mode == "likes" and not has_likes:
        raise ValueError(
            "Need users with >=2 liked tracks for evaluation (after holding out test likes)"
        )
    if mode == "genre" and len(set(genres.values())) < 2:
        raise ValueError("Genre objective needs tracks of at least 2 genres")
    tune_ids, test_ids = split_by_artist(
        list(genres), artists, settings.tuning_test_fraction, settings.tuning_seed
    )

    run = await db.get(AutoMLRunORM, run_id)
    if run is None:
        raise ValueError(f"AutoML run {run_id} not found")
    run.status = "running"
    run.started_at = utcnow()
    await db.commit()

    def build(
        params: dict, only: list[str] | None = None
    ) -> tuple[FeatureNormalizer, np.ndarray, FaissRecommender]:
        weights = {g: params[f"w_{g}"] for g in FEATURE_GROUPS}
        normalizer = FeatureNormalizer(
            method=params["norm_method"], weights=feature_weight_vector(weights)
        )
        normalized = normalizer.fit_transform(raw_features)
        engine = FaissRecommender(
            dimension=normalized.shape[1],
            metric=params["metric"],
            boost_weight=params.get("boost_weight", settings.default_boost_weight),
        )
        ids = track_ids if only is None else only
        engine.add_tracks(ids, normalized[[id_to_idx[t] for t in ids]].copy())
        return normalizer, normalized, engine

    def objective(trial: optuna.Trial) -> float:
        params = {
            "norm_method": trial.suggest_categorical("norm_method", settings.tuning_norm_methods),
            "metric": trial.suggest_categorical("metric", settings.tuning_metrics),
        }
        for group_name in FEATURE_GROUPS:
            params[f"w_{group_name}"] = trial.suggest_float(
                f"w_{group_name}", 0.0, settings.tuning_max_feature_weight
            )
        if mode == "likes":
            params["boost_weight"] = trial.suggest_float(
                "boost_weight", 0.0, settings.tuning_max_boost_weight
            )
            _, normalized, engine = build(params)
            metrics = evaluate(engine, normalized, id_to_idx, train)
            trial.set_user_attr("metrics", metrics)
            return objective_score(metrics)

        _, normalized, engine = build(params, only=tune_ids)
        report = genre_report(engine, normalized, id_to_idx, genres, artists, query_ids=tune_ids)
        trial.set_user_attr("metrics", report["all"])
        return report["all"][f"filtered@{k}"]

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=settings.tuning_seed)
    )
    study.optimize(objective, n_trials=settings.automl_n_trials, timeout=settings.automl_timeout)
    best = dict(study.best_params)

    normalizer, normalized, engine = build(best)
    if mode == "genre":
        best["boost_weight"] = (
            _tune_boost(engine, normalized, id_to_idx, train)
            if has_likes
            else settings.default_boost_weight
        )
        engine.boost_weight = best["boost_weight"]

    metrics = {
        "objective": mode,
        "train": study.best_trial.user_attrs["metrics"],
        "holdout": holdout_report(engine, normalized, id_to_idx, artists, train, test),
        "test_likes": sum(len(t) for t in test.values()),
        "genre_test": genre_report(
            engine, normalized, id_to_idx, genres, artists, query_ids=test_ids
        ),
    }

    run.status = "completed"
    run.best_score = study.best_value
    run.best_params = json.dumps(best)
    run.metrics = json.dumps(metrics)
    run.n_trials = len(study.trials)
    run.completed_at = utcnow()
    await db.commit()

    normalizer.save()
    engine.save()

    return {
        "best_score": study.best_value,
        "best_params": best,
        "n_trials": len(study.trials),
        **metrics,
    }


def _tune_boost(
    engine: FaissRecommender,
    normalized: np.ndarray,
    id_to_idx: dict[str, int],
    train: dict[str, set[str]],
) -> float:
    """Вес буста по leave-one-out на обучающих лайках при готовых параметрах признаков."""
    best_weight, best_score = 0.0, -1.0
    for weight in np.linspace(0.0, settings.tuning_max_boost_weight, 13):
        engine.boost_weight = float(weight)
        score = objective_score(evaluate(engine, normalized, id_to_idx, train))
        if score > best_score:  # при равенстве остаётся меньший вес
            best_weight, best_score = float(weight), score
    return best_weight


async def create_tuning_run(db: AsyncSession) -> int:
    """Зарегистрировать запуск тюнинга (status=pending), вернуть его id."""
    run = AutoMLRunORM(status="pending")
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run.id


async def execute_tuning_run(db: AsyncSession, run_id: int) -> dict:
    """Выполнить запуск. При ошибке помечает его failed с текстом ошибки и пробрасывает её."""
    try:
        return await run_tuning(db, run_id)
    except Exception as e:
        await db.rollback()
        run = await db.get(AutoMLRunORM, run_id)
        if run is not None:
            run.status = "failed"
            run.best_params = json.dumps({"error": str(e)})
            run.completed_at = utcnow()
            await db.commit()
        raise
