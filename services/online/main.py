"""CLI-входная точка online-сервиса."""

import uvicorn

from recommender.interfaces.online.main import app


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
