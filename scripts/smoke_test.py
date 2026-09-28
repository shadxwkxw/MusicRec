"""End-to-end smoke-тест: настоящий uvicorn, рестарт сервера и batch CLI.

Работает во временной папке со своим конфигом (CONFIG_PATH), рабочую data/
не трогает. Запуск: `make smoke` или `python scripts/smoke_test.py`.
"""

import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
SR = 22050
N_TRACKS = 6


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_config(work: Path) -> Path:
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    data = work / "data"
    config["paths"] = {
        "data_dir": str(data),
        "audio_dir": str(data / "audio"),
        "index_dir": str(data / "index"),
        "models_dir": str(data / "models"),
    }
    config["database"]["url"] = f"sqlite+aiosqlite:///{data / 'smoke.db'}"
    config["tuning"]["n_trials"] = 5
    path = work / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def make_audio(folder: Path) -> list[Path]:
    folder.mkdir()
    rng = np.random.default_rng(7)
    t = np.arange(SR * 3) / SR
    paths = []
    for i in range(N_TRACKS):
        y = 0.3 * np.sin(2 * np.pi * 110 * (1 + i * 0.4) * t)
        y += 0.02 * rng.standard_normal(len(t))
        path = folder / f"smoke_{i}.wav"
        sf.write(path, y.astype(np.float32), SR)
        paths.append(path)
    return paths


class Server:
    def __init__(self, env: dict[str, str], log: Path):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.env, self.log = env, log

    def __enter__(self) -> httpx.Client:
        self.log_file = self.log.open("a")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "recommender.interfaces.online.main:app",
             "--port", str(self.port)],
            env=self.env, stdout=self.log_file, stderr=subprocess.STDOUT,
        )  # fmt: skip
        self.client = httpx.Client(base_url=self.url, timeout=120)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early, see {self.log}")
            try:
                if self.client.get("/docs").status_code == 200:
                    return self.client
            except httpx.TransportError:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"server did not start, see {self.log}")

    def __exit__(self, *exc) -> None:
        self.client.close()
        self.proc.terminate()
        self.proc.wait(timeout=30)
        self.log_file.close()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="recommender-smoke-") as tmp:
        work = Path(tmp)
        env = {**os.environ, "CONFIG_PATH": str(write_config(work))}
        audio = make_audio(work / "src_audio")
        log = work / "server.log"

        try:
            print("server: first start")
            with Server(env, log) as api:
                ids = []
                for path in audio:
                    with path.open("rb") as f:
                        resp = api.post(
                            "/tracks/upload",
                            files={"file": (path.name, f, "audio/wav")},
                            data={"title": path.stem, "artist": "Smoke"},
                        )
                    resp.raise_for_status()
                    ids.append(resp.json()["id"])
                check(len(ids) == N_TRACKS, f"uploaded {N_TRACKS} tracks")

                rebuilt = api.post("/index/rebuild").json()
                check(rebuilt["tracks_indexed"] == N_TRACKS, "index rebuilt")

                recs = api.get(f"/recommendations/{ids[0]}").json()["recommendations"]
                check(len(recs) == N_TRACKS - 1, "recommendations by track")

                for tid in ids[:3]:
                    api.post("/likes", json={"user_id": "u1", "track_id": tid})
                for tid in ids[1:4]:
                    api.post("/likes", json={"user_id": "u2", "track_id": tid})
                user_recs = api.get("/recommendations/user/u1").json()["recommendations"]
                check(
                    {r["track_id"] for r in user_recs}.isdisjoint(ids[:3]), "user recs skip likes"
                )

                api.post("/automl/train")
                status = "pending"
                for _ in range(120):
                    status = api.get("/automl/status").json()[0]["status"]
                    if status in ("completed", "failed"):
                        break
                    time.sleep(1)
                check(status == "completed", "tuning completed")

                check(api.delete(f"/tracks/{ids[-1]}").status_code == 204, "track deleted")

            print("server: restart (index must load from disk)")
            with Server(env, log) as api:
                tracks = api.get("/tracks").json()
                check(len(tracks) == N_TRACKS - 1, "deleted track stays deleted")
                check(all(t["indexed"] for t in tracks), "all tracks indexed after restart")
                recs = api.get(f"/recommendations/{ids[0]}").json()["recommendations"]
                check(ids[-1] not in {r["track_id"] for r in recs}, "deleted track not recommended")

            print("batch CLI")
            batch = [sys.executable, str(ROOT / "services" / "batch" / "main.py")]
            inbox = work / "inbox"
            inbox.mkdir()
            for i in range(2):
                shutil.copy(audio[i], inbox / f"inbox_{i}.wav")
            db_path = work / "data" / "smoke.db"

            def track_count() -> int:
                with sqlite3.connect(db_path) as con:
                    return con.execute("select count(*) from tracks").fetchone()[0]

            before = track_count()
            subprocess.run([*batch, "extract", "--input-dir", str(inbox)], env=env, check=True)
            check(track_count() == before + 2, "batch extract imported 2 files")
            subprocess.run([*batch, "extract", "--input-dir", str(inbox)], env=env, check=True)
            check(track_count() == before + 2, "batch extract skips already imported files")

            out = work / "recs.parquet"
            subprocess.run(
                [*batch, "recommend", "--output", str(out), "--top-n", "3"], env=env, check=True
            )
            df = pd.read_parquet(out)
            check(len(df) == track_count() * 3, "batch wrote top-3 for every track")
            check((df.source_track_id != df.target_track_id).all(), "batch has no self-recs")
        except Exception:
            print(f"\n--- server log ---\n{log.read_text() if log.exists() else '(empty)'}")
            raise

    print("smoke test passed")


if __name__ == "__main__":
    main()
