"""Текстовая часть CLAP: запрос → эмбеддинг в том же пространстве, что и аудио.

Модуль намеренно не импортирует faiss, librosa и scikit-learn: он загружается
в отдельном процессе (см. text_worker.py), где нет чужих копий OpenMP.
"""

from collections.abc import Sequence

import numpy as np


class ClapTextEncoder:
    """Грузит только текстовую башню (~125M параметров против ~153M у полной модели)."""

    def __init__(self, model_name: str = "laion/clap-htsat-unfused", device: str = "cpu"):
        import torch
        from transformers import ClapProcessor, ClapTextModelWithProjection

        self._torch = torch
        self.model_name = model_name
        self.device = device
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapTextModelWithProjection.from_pretrained(model_name)
        self.model.to(torch.device(device))
        self.model.eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) L2-нормализованных эмбеддингов текстов."""
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True).to(self.device)
        with self._torch.no_grad():
            embeds = self.model(**inputs).text_embeds.float().cpu().numpy()
        norms = np.maximum(np.linalg.norm(embeds, axis=1, keepdims=True), 1e-12)
        return (embeds / norms).astype(np.float32)
