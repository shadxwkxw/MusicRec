"""Проверка DAG'ов Airflow: файлы импортируются без ошибок и нужные DAG'и на месте.

Запускается внутри образа Airflow (make airflow-check, CI):
    python scripts/check_dags.py <dag_id> ...
"""

import sys

from airflow.dag_processing.dagbag import DagBag

bag = DagBag(dag_folder="/opt/airflow/dags")
for path, error in bag.import_errors.items():
    print(f"IMPORT ERROR {path}:\n{error}")
print("dags:", ", ".join(sorted(bag.dag_ids)) or "none")

missing = set(sys.argv[1:]) - set(bag.dag_ids)
if bag.import_errors or missing:
    sys.exit(f"failed: {len(bag.import_errors)} import error(s), missing {sorted(missing)}")
