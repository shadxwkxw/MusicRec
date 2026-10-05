"""Настройки рекомендера: читает configs/config.yaml в типизированный Settings.

Поддерживает подстановку переменных окружения вида ${VAR} или ${VAR:-default}.
Перед этим подгружается .env из рабочей директории (путь меняется через
ENV_FILE, пустое значение отключает): переменные, уже заданные в окружении,
важнее файла. Путь к конфигу переопределяется через CONFIG_PATH.
"""

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

ENV_VAR_PATTERN = re.compile(r"^\$\{([^}]+)\}$")


def _resolve_env(value: Any) -> Any:
    """Рекурсивно подставляет ${VAR} / ${VAR:-default} в значениях."""
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    if not isinstance(value, str):
        return value
    match = ENV_VAR_PATTERN.match(value)
    if not match:
        return value
    expr = match.group(1)
    if ":-" in expr:
        name, default = expr.split(":-", 1)
    else:
        name, default = expr, None
    resolved = os.getenv(name)
    if resolved is None:
        if default is None:
            raise KeyError(f"Required env var '{name}' is not set")
        resolved = default
    return resolved


Metric = Literal["cosine", "euclidean"]
NormMethod = Literal["standard", "minmax", "robust"]


class Settings(BaseModel):
    # Paths
    data_dir: Path
    audio_dir: Path
    index_dir: Path
    models_dir: Path
    # Audio storage
    storage_backend: Literal["local", "s3"]
    s3_bucket: str | None
    s3_upload_prefix: str
    s3_index_prefix: str
    s3_artifacts_prefix: str
    s3_endpoint_url: str | None
    s3_region: str
    # Database
    db_url: str
    # Audio feature extraction
    sample_rate: int
    duration_limit: float
    n_mfcc: int
    n_chroma: int
    n_contrast_bands: int
    # Feature source
    feature_source: Literal["librosa", "embedding"]
    embedding_model: str
    embedding_window_seconds: float = Field(gt=0)
    embedding_max_windows: int = Field(ge=1)
    embedding_batch_tracks: int = Field(ge=1)
    embedding_loaders: int = Field(ge=1)
    # Text search
    search_prompt_templates: list[str] = Field(min_length=1)
    # Index
    index_keep_versions: int = Field(ge=2)
    # Recommendation
    default_rec_limit: int = Field(ge=1)
    candidate_multiplier: int = Field(ge=1)
    # Персональные рекомендации
    user_max_likes: int = Field(ge=1)
    user_max_interests: int = Field(ge=1)
    user_min_likes_per_interest: int = Field(ge=1)
    user_merge_similarity: float = Field(ge=-1, le=1)
    user_mean_share: float = Field(ge=0, le=1)
    user_max_per_artist: int = Field(ge=0)
    user_cold_start: bool
    default_metric: Metric
    default_norm_method: NormMethod
    default_boost_weight: float = Field(ge=0)
    # Tuning
    automl_n_trials: int = Field(ge=1)
    automl_timeout: int = Field(ge=1)
    tuning_eval_k: int = Field(ge=1)
    tuning_max_feature_weight: float = Field(gt=0)
    tuning_max_boost_weight: float = Field(ge=0)
    tuning_norm_methods: list[NormMethod] = Field(min_length=1)
    tuning_metrics: list[Metric] = Field(min_length=1)
    tuning_objective: Literal["auto", "genre", "likes"]
    tuning_min_genre_tracks: int = Field(ge=2)
    tuning_test_fraction: float = Field(ge=0, lt=1)
    tuning_seed: int
    # API
    api_host: str
    api_port: int
    api_tracks_page_size: int = Field(ge=1)
    api_tracks_page_max: int = Field(ge=1)

    @model_validator(mode="after")
    def _s3_needs_bucket(self) -> "Settings":
        if self.storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("storage.backend=s3 needs S3_BUCKET")
        return self

    @field_validator("search_prompt_templates")
    @classmethod
    def _templates_have_placeholder(cls, templates: list[str]) -> list[str]:
        bad = [t for t in templates if t.count("{}") != 1]
        if bad:
            raise ValueError(f"each prompt template needs exactly one {{}}: {bad}")
        return templates

    @property
    def feature_source_id(self) -> str:
        """Из чего строится индекс: librosa или embedding:<модель>."""
        if self.feature_source == "embedding":
            return f"embedding:{self.embedding_model}"
        return "librosa"

    @property
    def feature_dim(self) -> int:
        """Размерность итогового вектора признаков."""
        return (
            self.n_mfcc * 2  # mfcc mean + std
            + self.n_chroma * 2  # chroma mean + std
            + (self.n_contrast_bands + 1) * 2  # contrast: librosa возвращает n_bands+1 строк
            + 6 * 2  # tonnetz mean + std
            + 1  # tempo
            + 1  # rms mean
            + 1  # zcr mean
            + 3  # spectral centroid/bandwidth/rolloff means
        )


def _find_config_path() -> Path:
    """Порядок поиска:
    1. $CONFIG_PATH (явное указание)
    2. ./configs/config.yaml (рабочая директория)
    3. default_config.yaml, лежащий рядом с этим модулем (package default)
    """
    explicit = os.getenv("CONFIG_PATH")
    if explicit:
        return Path(explicit)
    cwd_path = Path("configs/config.yaml")
    if cwd_path.exists():
        return cwd_path
    return Path(__file__).parent / "default_config.yaml"


def load_env_file() -> Path | None:
    """Подгрузить .env (или $ENV_FILE), не перезаписывая заданные переменные."""
    path = Path(os.getenv("ENV_FILE", ".env"))
    if not os.getenv("ENV_FILE", ".env") or not path.is_file():
        return None
    load_dotenv(path, override=False)
    return path


def _load_config_dict() -> dict:
    with _find_config_path().open("r", encoding="utf-8") as f:
        return _resolve_env(yaml.safe_load(f))


def _build_settings(raw: dict) -> Settings:
    return Settings(
        data_dir=Path(raw["paths"]["data_dir"]),
        audio_dir=Path(raw["paths"]["audio_dir"]),
        index_dir=Path(raw["paths"]["index_dir"]),
        models_dir=Path(raw["paths"]["models_dir"]),
        storage_backend=raw["storage"]["backend"],
        s3_bucket=raw["storage"]["s3_bucket"] or None,
        s3_upload_prefix=raw["storage"]["s3_upload_prefix"],
        s3_index_prefix=raw["storage"]["s3_index_prefix"],
        s3_artifacts_prefix=raw["storage"]["s3_artifacts_prefix"],
        s3_endpoint_url=raw["storage"]["s3_endpoint_url"] or None,
        s3_region=raw["storage"]["s3_region"],
        db_url=raw["database"]["url"],
        sample_rate=raw["audio"]["sample_rate"],
        duration_limit=raw["audio"]["duration_limit"],
        n_mfcc=raw["audio"]["n_mfcc"],
        n_chroma=raw["audio"]["n_chroma"],
        n_contrast_bands=raw["audio"]["n_contrast_bands"],
        feature_source=raw["features"]["source"],
        embedding_model=raw["features"]["embedding_model"],
        embedding_window_seconds=raw["features"]["embedding_window_seconds"],
        embedding_max_windows=raw["features"]["embedding_max_windows"],
        embedding_batch_tracks=raw["features"]["embedding_batch_tracks"],
        embedding_loaders=raw["features"]["embedding_loaders"],
        search_prompt_templates=raw["search"]["prompt_templates"],
        index_keep_versions=raw["index"]["keep_versions"],
        default_rec_limit=raw["recommendation"]["default_limit"],
        candidate_multiplier=raw["recommendation"]["candidate_multiplier"],
        user_max_likes=raw["user_recommendation"]["max_likes"],
        user_max_interests=raw["user_recommendation"]["max_interests"],
        user_min_likes_per_interest=raw["user_recommendation"]["min_likes_per_interest"],
        user_merge_similarity=raw["user_recommendation"]["merge_similarity"],
        user_mean_share=raw["user_recommendation"]["mean_share"],
        user_max_per_artist=raw["user_recommendation"]["max_per_artist"],
        user_cold_start=raw["user_recommendation"]["cold_start"],
        default_metric=raw["recommendation"]["default_metric"],
        default_norm_method=raw["recommendation"]["default_norm_method"],
        default_boost_weight=raw["recommendation"]["default_boost_weight"],
        automl_n_trials=raw["tuning"]["n_trials"],
        automl_timeout=raw["tuning"]["timeout"],
        tuning_eval_k=raw["tuning"]["eval_k"],
        tuning_max_feature_weight=raw["tuning"]["max_feature_weight"],
        tuning_max_boost_weight=raw["tuning"]["max_boost_weight"],
        tuning_norm_methods=raw["tuning"]["norm_methods"],
        tuning_metrics=raw["tuning"]["metrics"],
        tuning_objective=raw["tuning"]["objective"],
        tuning_min_genre_tracks=raw["tuning"]["min_genre_tracks"],
        tuning_test_fraction=raw["tuning"]["test_fraction"],
        tuning_seed=raw["tuning"]["seed"],
        api_host=raw["api"]["host"],
        api_port=raw["api"]["port"],
        api_tracks_page_size=raw["api"]["tracks_page_size"],
        api_tracks_page_max=raw["api"]["tracks_page_max"],
    )


load_env_file()
settings = _build_settings(_load_config_dict())

for _d in (settings.data_dir, settings.audio_dir, settings.index_dir, settings.models_dir):
    _d.mkdir(parents=True, exist_ok=True)
