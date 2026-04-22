"""Извлечение аудио-признаков через librosa.

58-мерный вектор: MFCC(26) + Chroma(24) + Spectral Contrast(14) + Tonnetz(12)
+ Tempo(1) + RMS(1) + ZCR(1) + spectral stats(3).
"""

from pathlib import Path

import librosa
import numpy as np

from recommender.config import settings


def extract_features(audio_path: str | Path) -> np.ndarray:
    """Извлечь вектор признаков из аудио-файла.

    Args:
        audio_path: путь к аудио (mp3, wav, flac, ogg, ...).

    Returns:
        1D numpy array формы (58,).
    """
    y, sr = librosa.load(
        str(audio_path),
        sr=settings.sample_rate,
        duration=settings.duration_limit,
        mono=True,
    )

    features: list[float] = []

    # 1. MFCC — timbral texture
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=settings.n_mfcc)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2. Chroma — harmonic / pitch content
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=settings.n_chroma)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # 3. Spectral Contrast — brightness per band
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_bands=settings.n_contrast_bands
    )
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # 4. Tonnetz — tonal centroid features
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    features.extend(np.mean(tonnetz, axis=1))
    features.extend(np.std(tonnetz, axis=1))

    # 5. Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]
    features.append(float(tempo))

    # 6. RMS Energy — average loudness
    rms = librosa.feature.rms(y=y)
    features.append(float(np.mean(rms)))

    # 7. Zero Crossing Rate — percussiveness
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(float(np.mean(zcr)))

    # 8. Spectral statistics
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(float(np.mean(centroid)))
    features.append(float(np.mean(bandwidth)))
    features.append(float(np.mean(rolloff)))

    return np.array(features, dtype=np.float32)


def features_to_bytes(features: np.ndarray) -> bytes:
    """Сериализовать вектор в bytes для хранения в БД."""
    return features.tobytes()


def bytes_to_features(data: bytes) -> np.ndarray:
    """Десериализовать вектор из bytes."""
    return np.frombuffer(data, dtype=np.float32)
