"""Поддельные кодировщики для тестов ProcessTextEncoder (без torch)."""

import sys

import numpy as np


class EchoEncoder:
    """Вектор из длин текстов; пишет в stdout, как это делают настоящие библиотеки."""

    def __init__(self, model_name: str):
        print("loading weights... (noise on stdout)", flush=True)
        self.model_name = model_name

    def encode(self, texts):
        print("encoding", texts, flush=True)
        if any(t == "boom" for t in texts):
            raise ValueError("bad text")
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


class ModulesProbe:
    """Сообщает, загружен ли faiss в процессе кодировщика."""

    model_name = "probe"

    def encode(self, texts):
        return np.array([[float("faiss" in sys.modules)]], dtype=np.float32)


class MissingDependency:
    def __init__(self):
        import nonexistent_ml_library  # noqa: F401
