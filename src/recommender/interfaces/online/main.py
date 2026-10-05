"""FastAPI online service bootstrap."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from recommender.application.features import IndexSourceMismatchError, check_index_source
from recommender.config import settings
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.artifacts import load_current
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import init_db
from recommender.interfaces.online.routes import router
from recommender.interfaces.online.security import API_KEY_HEADER


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown жизненный цикл."""
    await init_db()
    if settings.api_key is None:
        print("⚠ API_KEY is not set: anyone can change tracks, likes and the index (dev only)")

    try:
        artifacts = load_current()
        engine, normalizer = artifacts.engine, artifacts.normalizer
        check_index_source(engine)
        app.state.engine, app.state.normalizer = engine, normalizer
        print(
            f"✓ Loaded index {engine.version} with {engine.index.ntotal} tracks ({engine.source})"
        )
    except IndexSourceMismatchError as e:
        app.state.engine, app.state.normalizer = FaissRecommender(), FeatureNormalizer()
        print(f"⚠ {e}. Starting with empty index")
    except FileNotFoundError:
        app.state.engine, app.state.normalizer = FaissRecommender(), FeatureNormalizer()
        print("⚡ Starting with empty index")

    yield

    encoder = getattr(app.state, "text_encoder", None)
    if encoder is not None and hasattr(encoder, "close"):
        encoder.close()


def create_app() -> FastAPI:
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
    application.include_router(router)
    return application


app = create_app()
