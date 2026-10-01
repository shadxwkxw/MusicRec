"""FastAPI online service bootstrap."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from recommender.application.features import IndexSourceMismatchError, check_index_source
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import init_db
from recommender.interfaces.online.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown жизненный цикл."""
    await init_db()

    try:
        engine, normalizer = FaissRecommender.load(), FeatureNormalizer.load()
        check_index_source(engine)
        app.state.engine, app.state.normalizer = engine, normalizer
        print(f"✓ Loaded index with {engine.index.ntotal} tracks ({engine.source})")
    except IndexSourceMismatchError as e:
        app.state.engine, app.state.normalizer = FaissRecommender(), FeatureNormalizer()
        print(f"⚠ {e}. Starting with empty index")
    except FileNotFoundError:
        app.state.engine, app.state.normalizer = FaissRecommender(), FeatureNormalizer()
        print("⚡ Starting with empty index")

    yield


app = FastAPI(
    title="Music Recommender API",
    description="Content-based music recommendation with hyperparameter tuning",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
