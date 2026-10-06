"""m4a/AAC: libsndfile его не читает, декодирование идёт через ffmpeg."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from recommender.config import settings
from recommender.infrastructure.data_processing import audio
from recommender.infrastructure.data_processing.extract import extract_features

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not on PATH")


@pytest.fixture(scope="session")
def m4a_file(audio_files, tmp_path_factory) -> Path:
    if FFMPEG is None:
        pytest.skip("ffmpeg not on PATH")
    out = tmp_path_factory.mktemp("m4a") / "track.m4a"
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-i", str(audio_files[0])]
    subprocess.run([*cmd, "-vn", "-c:a", "aac", "-b:a", "128k", str(out)], check=True)
    return out


@needs_ffmpeg
def test_extract_features_from_m4a(m4a_file):
    features = extract_features(m4a_file)

    assert features.shape == (settings.feature_dim,) == (82,)
    assert np.isfinite(features).all()


@needs_ffmpeg
def test_m4a_respects_duration_limit(m4a_file):
    y = audio.load_audio(m4a_file, sr=8000, duration=1.0)

    assert y.dtype == np.float32
    assert len(y) == 8000


@needs_ffmpeg
async def test_upload_m4a(api, m4a_file):
    with m4a_file.open("rb") as f:
        resp = await api.client.post(
            "/tracks/upload",
            files={"file": ("song.m4a", f, "audio/mp4")},
            data={"title": "AAC", "artist": "Artist"},
        )
    assert resp.status_code == 200, resp.text
    track_id = resp.json()["id"]
    assert resp.json()["duration"] == pytest.approx(3.0, abs=0.1)

    resp = await api.client.get(f"/tracks/{track_id}/features")
    assert resp.status_code == 200, resp.text
    assert [p.suffix for p in settings.audio_dir.iterdir()] == [".m4a"]


def test_unsupported_format_without_ffmpeg_is_explicit(tmp_path, monkeypatch):
    path = tmp_path / "song.m4a"
    path.write_bytes(b"\x00\x00\x00\x18ftypM4A not really")
    monkeypatch.setattr(audio.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="needs ffmpeg"):
        audio.load_audio(path, sr=22050)
    with pytest.raises(RuntimeError, match="needs ffprobe"):
        audio.get_duration(path)


@needs_ffmpeg
def test_undecodable_file_fails_with_ffmpeg_error(tmp_path):
    path = tmp_path / "broken.m4a"
    path.write_bytes(b"not audio at all")

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        audio.load_audio(path, sr=22050)
    with pytest.raises(RuntimeError, match="ffprobe failed"):
        audio.get_duration(path)
