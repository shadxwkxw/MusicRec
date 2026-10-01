"""Аудиоэмбеддинги предобученной моделью CLAP (Hugging Face transformers).

Модель принимает не больше window_seconds аудио и по умолчанию случайно
вырезает кусок из более длинного — эмбеддинг был бы недетерминированным.
Поэтому трек режется на окна сами: до max_windows окон, равномерно по треку;
эмбеддинги окон L2-нормализуются, усредняются и снова нормализуются.

torch и transformers — опциональные зависимости (make install-embeddings),
поэтому импортируются при создании ClapEmbedder.
"""

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 48_000
MIN_LAST_WINDOW_SECONDS = 3.0


def split_windows(y: np.ndarray, window: int, max_windows: int) -> list[np.ndarray]:
    """Окна по window отсчётов, равномерно по треку; короткий хвост отбрасывается."""
    if len(y) <= window:
        return [y]
    starts = list(range(0, len(y) - window + 1, window))
    tail = len(y) - (starts[-1] + window)
    if tail >= MIN_LAST_WINDOW_SECONDS * SAMPLE_RATE:
        starts.append(len(y) - window)
    if len(starts) > max_windows:
        picks = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in picks]
    return [y[s : s + window] for s in starts]


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class ClapEmbedder:
    def __init__(
        self,
        model_name: str = "laion/larger_clap_music",
        window_seconds: float = 10.0,
        max_windows: int = 6,
        duration_limit: float | None = None,
        device: str | None = None,
    ):
        import torch
        from transformers import ClapModel, ClapProcessor

        self._torch = torch
        self.model_name = model_name
        self.window = int(window_seconds * SAMPLE_RATE)
        self.max_windows = max_windows
        self.duration_limit = duration_limit
        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        self.device = device
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.model.to(torch.device(device))
        self.model.eval()
        self.dim = int(self.model.config.projection_dim)

    def load(self, path: str | Path) -> np.ndarray:
        y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=self.duration_limit)
        return y

    def embed_waveforms(
        self, waveforms: Sequence[np.ndarray], batch_windows: int = 64
    ) -> np.ndarray:
        """(n_tracks, dim): средний нормализованный эмбеддинг окон каждого трека."""
        windows, owner = [], []
        for i, y in enumerate(waveforms):
            for w in split_windows(y, self.window, self.max_windows):
                windows.append(w)
                owner.append(i)

        vectors = []
        for start in range(0, len(windows), batch_windows):
            inputs = self.processor(
                audio=windows[start : start + batch_windows],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.no_grad():
                out = self.model.get_audio_features(**inputs)
            pooled = out if self._torch.is_tensor(out) else out.pooler_output
            vectors.append(pooled.float().cpu().numpy())
        per_window = _normalize(np.concatenate(vectors))

        owners = np.array(owner)
        means = np.stack([per_window[owners == i].mean(axis=0) for i in range(len(waveforms))])
        return _normalize(means).astype(np.float32)

    def embed_files(
        self, paths: Sequence[str | Path], batch_tracks: int = 32, loaders: int = 8
    ) -> Iterator[tuple[int, np.ndarray | str]]:
        """(индекс пути, вектор или текст ошибки) пачками; аудио декодируется в потоках."""

        def safe_load(path: str | Path) -> np.ndarray | str:
            try:
                return self.load(path)
            except Exception as e:
                return f"{type(e).__name__}: {e}"

        with ThreadPoolExecutor(max_workers=loaders) as pool:
            for start in range(0, len(paths), batch_tracks):
                chunk = list(pool.map(safe_load, paths[start : start + batch_tracks]))
                ok = [(i, y) for i, y in enumerate(chunk) if not isinstance(y, str)]
                if ok:
                    embedded = self.embed_waveforms([y for _, y in ok])
                    for (i, _), vector in zip(ok, embedded, strict=True):
                        yield start + i, vector
                for i, y in enumerate(chunk):
                    if isinstance(y, str):
                        yield start + i, y
