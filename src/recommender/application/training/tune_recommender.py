"""Use case: гиперпараметрическая оптимизация рекомендера через Optuna.

Оптимизируемое:
- метод нормализации (standard / minmax / robust)
- метрика расстояния (cosine / euclidean)
- групповые веса признаков (mfcc / chroma / contrast / ...)
- вес коллаборативного бустинга

Лайки делятся на обучающие и отложенные (tuning.test_fraction). Параметры
подбираются по leave-one-out на обучающих лайках (среднее MRR@k по двум
прод-путям, см. evaluation.py), а итоговая оценка вместе с бейзлайнами
считается на отложенных — их тюнинг не видел. Пространство поиска, k и seed
задаются в секции tuning конфига.

Итог: индекс и нормализатор пересобраны с лучшими параметрами и
сохранены на диск.
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
    holdout_report,
    leave_one_out_queries,
    objective_score,
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


def apply_feature_weights(features: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """Применить групповые веса к матрице признаков."""
    return features * feature_weight_vector(weights)


async def run_tuning(db: AsyncSession, run_id: int) -> dict:
    """Подобрать параметры на обучающих лайках, оценить на отложенных, пересобрать индекс."""
    result = await db.execute(select(TrackORM).where(TrackORM.feature_vector.isnot(None)))
    tracks: Sequence[TrackORM] = result.scalars().all()

    if len(tracks) < 5:
        raise ValueError("Need at least 5 tracks with features to run tuning")

    track_ids = [t.id for t in tracks]
    raw_features = np.array([bytes_to_features(t.feature_vector) for t in tracks])
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}

    train, test = split_likes(
        await load_user_likes(db), settings.tuning_test_fraction, settings.tuning_seed
    )
    if not leave_one_out_queries(train, set(track_ids)):
        raise ValueError(
            "Need users with >=2 liked tracks for evaluation (after holding out test likes)"
        )

    run = await db.get(AutoMLRunORM, run_id)
    if run is None:
        raise ValueError(f"AutoML run {run_id} not found")
    run.status = "running"
    run.started_at = utcnow()
    await db.commit()

    def build(params: dict) -> tuple[FeatureNormalizer, np.ndarray, FaissRecommender]:
        weights = {g: params[f"w_{g}"] for g in FEATURE_GROUPS}
        normalizer = FeatureNormalizer(
            method=params["norm_method"], weights=feature_weight_vector(weights)
        )
        normalized = normalizer.fit_transform(raw_features)
        engine = FaissRecommender(
            dimension=normalized.shape[1],
            metric=params["metric"],
            boost_weight=params["boost_weight"],
        )
        engine.add_tracks(track_ids, normalized.copy())
        return normalizer, normalized, engine

    def objective(trial: optuna.Trial) -> float:
        params = {
            "norm_method": trial.suggest_categorical("norm_method", settings.tuning_norm_methods),
            "metric": trial.suggest_categorical("metric", settings.tuning_metrics),
            "boost_weight": trial.suggest_float(
                "boost_weight", 0.0, settings.tuning_max_boost_weight
            ),
        }
        for group_name in FEATURE_GROUPS:
            params[f"w_{group_name}"] = trial.suggest_float(
                f"w_{group_name}", 0.0, settings.tuning_max_feature_weight
            )
        _, normalized, engine = build(params)
        metrics = evaluate(engine, normalized, id_to_idx, train)
        trial.set_user_attr("metrics", metrics)
        return objective_score(metrics)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=settings.tuning_seed)
    )
    study.optimize(objective, n_trials=settings.automl_n_trials, timeout=settings.automl_timeout)

    best = study.best_params
    normalizer, normalized, engine = build(best)
    artists = {t.id: t.artist or "" for t in tracks}
    holdout = holdout_report(engine, normalized, id_to_idx, artists, train, test)
    metrics = {
        "train": study.best_trial.user_attrs["metrics"],
        "holdout": holdout,
        "test_likes": sum(len(t) for t in test.values()),
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
