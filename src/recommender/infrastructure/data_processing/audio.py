"""Чтение аудио: сигнал и длительность.

librosa 1.0 читает только через soundfile (libsndfile): wav/flac/ogg/mp3.
Фолбэка на audioread в ней больше нет, поэтому m4a/AAC и прочее, что
libsndfile не знает, читаем через ffmpeg/ffprobe (они есть в Docker-образах).
"""

import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str | Path, sr: int, duration: float | None = None) -> np.ndarray:
    """Моно float32 с частотой sr, не длиннее duration секунд."""
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True, duration=duration)
    except sf.LibsndfileError as e:
        cmd = [_tool("ffmpeg", path, e), "-nostdin", "-v", "error", "-i", str(path)]
        if duration is not None:
            cmd += ["-t", str(duration)]
        # -vn: у mp3/m4a бывает обложка отдельным видеопотоком
        cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        y = np.frombuffer(_run(cmd, path, e), dtype=np.float32).copy()
        if y.size == 0:
            raise RuntimeError(f"ffmpeg decoded no audio from {path}") from e
    return y


def get_duration(path: str | Path) -> float:
    """Полная длительность трека в секундах."""
    try:
        return float(librosa.get_duration(path=str(path)))
    except sf.LibsndfileError as e:
        cmd = [_tool("ffprobe", path, e), "-v", "error", "-show_entries", "format=duration"]
        out = _run([*cmd, "-of", "default=noprint_wrappers=1:nokey=1", str(path)], path, e)
        try:
            return float(out.decode().strip())
        except ValueError:
            raise RuntimeError(f"ffprobe reported no duration for {path}") from e


def _tool(name: str, path: str | Path, cause: Exception) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError(f"Cannot read {path}: format needs {name}, not on PATH") from cause
    return exe


def _run(cmd: list[str], path: str | Path, cause: Exception) -> bytes:
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"{Path(cmd[0]).name} failed to read {path}: {detail}") from cause
    return proc.stdout
