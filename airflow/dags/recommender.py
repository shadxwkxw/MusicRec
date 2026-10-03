"""DAG'и рекомендера: задачи запускают batch-образ проекта через DockerOperator.

Логика остаётся в batch CLI (services/batch/main.py), DAG только упорядочивает
шаги. Окружение задаётся в docker-compose (профиль airflow):

    RECOMMENDER_HOST_DIR     абсолютный путь к проекту на хосте (bind mount data/)
    RECOMMENDER_BATCH_IMAGE  образ batch-сервиса
    RECOMMENDER_NETWORK      docker-сеть, где доступны postgres и recommender-online
    RECOMMENDER_API_URL      адрес online-сервиса для /index/reload
    RECOMMENDER_DOCKER_URL   Docker API (через docker-socket-proxy)
    DB_URL, FEATURE_SOURCE   передаются в batch-контейнеры как есть
"""

import json
import os
import urllib.error
import urllib.request

import pendulum
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import dag, task
from docker.types import Mount

HOST_DIR = os.getenv("RECOMMENDER_HOST_DIR", "/opt/recommender")
BATCH_IMAGE = os.getenv("RECOMMENDER_BATCH_IMAGE", "music-recommender-batch:latest")
NETWORK = os.getenv("RECOMMENDER_NETWORK", "recommender-net")
API_URL = os.getenv("RECOMMENDER_API_URL", "http://recommender-online:8000")
DOCKER_URL = os.getenv("RECOMMENDER_DOCKER_URL", "unix://var/run/docker.sock")
INBOX = "data/inbox"

DEFAULT_ARGS = {
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}


def batch(task_id: str, *command: str, **kwargs) -> DockerOperator:
    """Задача = одна команда batch CLI в отдельном контейнере."""
    return DockerOperator(
        task_id=task_id,
        image=BATCH_IMAGE,
        command=list(command),
        docker_url=DOCKER_URL,
        network_mode=NETWORK,
        mounts=[
            Mount(source=f"{HOST_DIR}/data", target="/app/data", type="bind"),
            Mount(source=f"{HOST_DIR}/artifacts", target="/app/artifacts", type="bind"),
            Mount(source=f"{HOST_DIR}/configs", target="/app/configs", type="bind", read_only=True),
            # веса моделей для embed скачиваются один раз и переживают контейнер
            Mount(source="recommender-hf-cache", target="/root/.cache/huggingface", type="volume"),
        ],
        environment={
            "DB_URL": os.environ.get("DB_URL", ""),
            "FEATURE_SOURCE": os.environ.get("FEATURE_SOURCE", "librosa"),
            "ENV_FILE": "",  # всё окружение задаётся здесь явно
            "PYTHONUNBUFFERED": "1",
        },
        mount_tmp_dir=False,
        auto_remove="success",
        **kwargs,
    )


@task
def reload_online_index() -> dict:
    """Переключить работающий online-сервис на опубликованную версию индекса."""
    request = urllib.request.Request(f"{API_URL}/index/reload", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"reload failed: {e.code} {e.read().decode()}") from e
    print(f"online service switched to index {result.get('version')}: {result}")
    return result


@task.branch
def choose_feature_step() -> str:
    """Эмбеддинги досчитываем только в режиме embedding (нужен образ с torch)."""
    return "embed" if os.environ.get("FEATURE_SOURCE") == "embedding" else "rebuild"


@dag(
    dag_id="recommender_daily",
    description="Импорт новых треков, признаки, пересборка индекса, batch-рекомендации",
    schedule="0 3 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["recommender"],
)
def recommender_daily():
    extract = batch("extract", "extract", "--input-dir", INBOX, "--workers", "4")
    embed = batch("embed", "embed")
    rebuild = batch("rebuild", "rebuild", trigger_rule="none_failed_min_one_success")
    recommend = batch(
        "recommend",
        "recommend",
        "--output",
        # run_after есть у любого запуска; у ручного в Airflow 3 нет logical_date и {{ ds }}
        "artifacts/recs_{{ dag_run.run_after | ds }}.parquet",
        "--use-likes",
    )

    extract >> choose_feature_step() >> [embed, rebuild]
    embed >> rebuild
    rebuild >> reload_online_index() >> recommend


@dag(
    dag_id="recommender_weekly_tuning",
    description="Подбор параметров по жанрам и лайкам, переключение сервиса, оценка",
    schedule="0 4 * * 0",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["recommender"],
)
def recommender_weekly_tuning():
    tune = batch("tune", "tune")
    evaluate = batch("evaluate", "evaluate")
    tune >> reload_online_index() >> evaluate


recommender_daily()
recommender_weekly_tuning()
