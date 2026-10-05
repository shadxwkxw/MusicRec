"""Общие настройки тестов: они не должны зависеть от локального .env и окружения.

Выполняется до импорта приложения (настройки строятся при импорте config).
"""

import os

os.environ["ENV_FILE"] = ""  # не читать .env разработчика
os.environ.pop("FEATURE_SOURCE", None)  # тесты рассчитаны на источник по умолчанию
for name in ("API_KEY", "API_PROTECT_READS", "CORS_ORIGINS"):  # доступ — как без настроек
    os.environ.pop(name, None)
