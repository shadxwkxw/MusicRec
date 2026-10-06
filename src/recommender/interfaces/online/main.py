"""FastAPI online service bootstrap."""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from recommender.application.features import IndexSourceMismatchError, check_index_source
from recommender.config import settings
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.artifacts import load_current
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import init_db
from recommender.interfaces.online import health, observability
from recommender.interfaces.online.routes import router
from recommender.interfaces.online.security import API_KEY_HEADER, require_read_access

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown жизненный цикл."""
    await init_db()
    if settings.api_key is None:
        logger.warning(
            "API_KEY is not set: anyone can change tracks, likes and the index (dev only)"
        )

    try:
        artifacts = load_current()
        engine, normalizer = artifacts.engine, artifacts.normalizer
        check_index_source(engine)
        app.state.engine, app.state.normalizer = engine, normalizer
        logger.info(
            "Loaded index %s with %d tracks (%s)",
            engine.version,
            engine.index.ntotal,
            engine.source,
        )
    except IndexSourceMismatchError as e:
        app.state.engine, app.state.normalizer = FaissRecommender(), FeatureNormalizer()
        logger.warning("%s. Starting with empty index", e)
    except FileNotFoundError:
        app.state.engine, app.state.normalizer = FaissRecommender(), FeatureNormalizer()
        logger.info("No saved index yet. Starting with empty index")

    yield

    encoder = getattr(app.state, "text_encoder", None)
    if encoder is not None and hasattr(encoder, "close"):
        encoder.close()


def create_app() -> FastAPI:
    observability.configure_logging()
    application = FastAPI(
        title="Music Recommender API",
        description="Content-based music recommendation with hyperparameter tuning",
        version="1.0.0",
        lifespan=lifespan,
    )
    if settings.api_cors_origins:  # браузер пустит запросы только с этих доменов
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api_cors_origins,
            allow_methods=["*"],
            allow_headers=["Content-Type", API_KEY_HEADER],
        )
    observability.install(application)
    application.include_router(health.router)
    application.include_router(router)

    @application.get(
        "/metrics", include_in_schema=False, dependencies=[Depends(require_read_access)]
    )
    def metrics() -> Response:
        """Метрики Prometheus (ключ нужен, если api.protect_reads)."""
        registry = observability.metrics_of(application).registry
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
