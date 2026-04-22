"""Настройки рекомендера: читает configs/config.yaml в типизированный Settings.

Поддерживает подстановку переменных окружения вида ${VAR} или ${VAR:-default}.
Путь к конфигу переопределяется через CONFIG_PATH.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


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


class Settings(BaseModel):
    # Paths
    data_dir: Path
    audio_dir: Path
    index_dir: Path
    models_dir: Path
    # Database
    db_url: str
    # Audio feature extraction
    sample_rate: int
    duration_limit: float
    n_mfcc: int
    n_chroma: int
    n_contrast_bands: int
    # Recommendation
    default_rec_limit: int
    faiss_nprobe: int
    # Tuning
    automl_n_trials: int
    automl_timeout: int
    # API
    api_host: str
    api_port: int

    @property
    def feature_dim(self) -> int:
        """Размерность итогового вектора признаков."""
        return (
            self.n_mfcc * 2                # mfcc mean + std
            + self.n_chroma * 2            # chroma mean + std
            + (self.n_contrast_bands + 1) * 2  # contrast: librosa возвращает n_bands+1 строк
            + 6 * 2                        # tonnetz mean + std
            + 1                            # tempo
            + 1                            # rms mean
            + 1                            # zcr mean
            + 3                            # spectral centroid/bandwidth/rolloff means
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


def _load_config_dict() -> dict:
    with _find_config_path().open("r", encoding="utf-8") as f:
        return _resolve_env(yaml.safe_load(f))


def _build_settings(raw: dict) -> Settings:
    return Settings(
        data_dir=Path(raw["paths"]["data_dir"]),
        audio_dir=Path(raw["paths"]["audio_dir"]),
        index_dir=Path(raw["paths"]["index_dir"]),
        models_dir=Path(raw["paths"]["models_dir"]),
        db_url=raw["database"]["url"],
        sample_rate=raw["audio"]["sample_rate"],
        duration_limit=raw["audio"]["duration_limit"],
        n_mfcc=raw["audio"]["n_mfcc"],
        n_chroma=raw["audio"]["n_chroma"],
        n_contrast_bands=raw["audio"]["n_contrast_bands"],
        default_rec_limit=raw["recommendation"]["default_limit"],
        faiss_nprobe=raw["recommendation"]["faiss_nprobe"],
        automl_n_trials=raw["tuning"]["n_trials"],
        automl_timeout=raw["tuning"]["timeout"],
        api_host=raw["api"]["host"],
        api_port=raw["api"]["port"],
    )


settings = _build_settings(_load_config_dict())

for _d in (settings.data_dir, settings.audio_dir, settings.index_dir, settings.models_dir):
    _d.mkdir(parents=True, exist_ok=True)
