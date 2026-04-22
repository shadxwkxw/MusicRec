"""FastAPI online service bootstrap."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import init_db
from recommender.interfaces.online.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown жизненный цикл."""
    await init_db()

    try:
        app.state.engine = FaissRecommender.load()
        app.state.normalizer = FeatureNormalizer.load()
        print(f"✓ Loaded index with {app.state.engine.index.ntotal} tracks")
    except Exception:
        app.state.engine = FaissRecommender()
        app.state.normalizer = FeatureNormalizer()
        print("⚡ Starting with empty index")

    yield


app = FastAPI(
    title="Music Recommender API",
    description="Content-based music recommendation with hyperparameter tuning",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
