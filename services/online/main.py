"""CLI-входная точка online-сервиса."""

import uvicorn

from recommender.config import settings
from recommender.interfaces.online.main import app


def main() -> None:
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
