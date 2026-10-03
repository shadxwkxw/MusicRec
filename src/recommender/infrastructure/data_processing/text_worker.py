"""Текстовый кодировщик в отдельном процессе.

На macOS faiss, torch и scikit-learn везут каждый свою копию OpenMP (libomp),
и две инициализированные копии в одном процессе его роняют (OMP Error #15,
в сервисе — segfault). Поэтому модель живёт в отдельной программе
(`python -m recommender.infrastructure.data_processing.text_worker`), где нет
faiss, а сервис отправляет ей тексты через stdin/stdout.

Обычный multiprocessing не подходит: при запуске spawn дочерний процесс заново
импортирует главный модуль родителя — а вместе с ним и faiss.

Протокол: pickle-сообщения в обе стороны. Весь вывод библиотек в дочернем
процессе перенаправлен в stderr, чтобы не смешиваться с протоколом.
"""

import importlib
import os
import pickle
import select
import subprocess
import sys
import threading
from collections.abc import Sequence
from typing import IO, Any

import numpy as np


def _send(stream: IO[bytes], message: Any) -> None:
    pickle.dump(message, stream, protocol=pickle.HIGHEST_PROTOCOL)
    stream.flush()


def _serve(factory: str, args: list[str]) -> None:
    # stdout — только для протокола; print() библиотек уходит в stderr
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    protocol_in = sys.stdin.buffer
    try:
        module, name = factory.split(":")
        encoder = getattr(importlib.import_module(module), name)(*args)
    except Exception as e:
        _send(protocol_out, ("error", type(e).__name__, str(e)))
        return
    _send(protocol_out, ("ready", encoder.model_name, ""))
    while True:
        try:
            texts = pickle.load(protocol_in)
        except EOFError:
            return
        if texts is None:
            return
        try:
            _send(protocol_out, ("ok", encoder.encode(texts), ""))
        except Exception as e:
            _send(protocol_out, ("error", type(e).__name__, str(e)))


class ProcessTextEncoder:
    """TextEncoder, который считает в отдельном процессе.

    factory — "модуль:класс" кодировщика, args — строковые аргументы его
    конструктора. Если в дочернем процессе нет нужной библиотеки, конструктор
    поднимает ImportError, как при обычном импорте.
    """

    def __init__(self, factory: str, *args: str, timeout: float = 300):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        self._process = subprocess.Popen(
            [sys.executable, "-m", __name__, factory, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )
        self._lock = threading.Lock()
        self._timeout = timeout
        status, payload, detail = self._receive()
        if status == "error":
            self.close()
            error = (
                ImportError if payload in ("ImportError", "ModuleNotFoundError") else RuntimeError
            )
            raise error(f"text encoder failed to start: {payload}: {detail}")
        self.model_name: str = payload

    def _receive(self) -> tuple[str, Any, str]:
        assert self._process.stdout is not None
        ready, _, _ = select.select([self._process.stdout], [], [], self._timeout)
        if not ready:
            raise TimeoutError("text encoder process did not answer")
        try:
            return pickle.load(self._process.stdout)
        except EOFError as e:
            raise RuntimeError(
                f"text encoder process died (exit code {self._process.poll()})"
            ) from e

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        assert self._process.stdin is not None
        with self._lock:
            _send(self._process.stdin, list(texts))
            status, payload, detail = self._receive()
        if status == "error":
            raise RuntimeError(f"text encoder failed: {payload}: {detail}")
        return payload

    def close(self) -> None:
        if self._process.poll() is None and self._process.stdin is not None:
            try:
                _send(self._process.stdin, None)
                self._process.stdin.close()
                self._process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._process.kill()


if __name__ == "__main__":
    _serve(sys.argv[1], sys.argv[2:])
