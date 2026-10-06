"""E2E по docker compose: собранный образ online-сервиса + Postgres + API-ключ.

Только стандартная библиотека — запускается системным Python без установки
проекта. Две фазы, между ними CI перезапускает контейнер:

    python scripts/e2e_compose.py seed     # загрузка, индекс, лайки, проверки доступа
    docker compose restart recommender-online
    python scripts/e2e_compose.py verify   # после рестарта всё на месте

Окружение: API_URL (по умолчанию http://localhost:8000) и API_KEY.
"""

import io
import json
import math
import os
import random
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ["API_KEY"]
STATE = os.environ.get("E2E_STATE", "/tmp/recommender-e2e.json")
N_TRACKS = 6
SR = 22050


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}", flush=True)


def request(
    method: str,
    path: str,
    body: bytes | None = None,
    content_type: str | None = None,
    key: str | None = API_KEY,
) -> tuple[int, dict | list | None]:
    headers = {"Content-Type": content_type} if content_type else {}
    if key:
        headers["X-API-Key"] = key
    req = urllib.request.Request(f"{API_URL}{path}", data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, None


def post_json(path: str, data: dict, key: str | None = API_KEY):
    return request("POST", path, json.dumps(data).encode(), "application/json", key)


def tone(index: int) -> bytes:
    """3 секунды: аккорд и щелчки своего темпа — у треков разные признаки."""
    rng = random.Random(index)
    freq, bpm = 110 * (1 + index * 0.35), 70 + index * 12
    beat = int(SR * 60 / bpm)
    frames = bytearray()
    for n in range(SR * 3):
        t = n / SR
        value = 0.3 * math.sin(2 * math.pi * freq * t) + 0.1 * math.sin(3 * math.pi * freq * t)
        value += 0.5 * math.exp(-(((n % beat) / 60) ** 2)) + 0.02 * rng.gauss(0, 1)
        frames += struct.pack("<h", int(max(-1, min(1, value / 1.2)) * 32000))
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(frames))
    return out.getvalue()


def upload(index: int, key: str | None = API_KEY):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("title", f"E2E {index}"), ("artist", f"Artist {index % 3}")):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="e2e_{index}.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
        + tone(index)
        + b"\r\n"
    )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    return request("POST", "/tracks/upload", body, f"multipart/form-data; boundary={boundary}", key)


def wait_ready(timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{API_URL}/health/live", timeout=5):
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"{API_URL} did not start in {timeout:.0f}s")


def metrics() -> str:
    with urllib.request.urlopen(f"{API_URL}/metrics", timeout=30) as resp:
        return resp.read().decode()


def recommendations(path: str) -> dict:
    status, body = request("GET", path, key=None)
    check(status == 200, f"GET {path}" + ("" if status == 200 else f": {status} {body}"))
    assert isinstance(body, dict)
    return body


def seed() -> None:
    print("seed:")
    check(upload(0, key=None)[0] == 401, "upload without key is rejected")
    check(upload(0, key="wrong-key-wrong-key")[0] == 401, "upload with a wrong key is rejected")

    ids = []
    for i in range(N_TRACKS):
        status, body = upload(i)
        check(status == 200, f"upload track {i}" + ("" if status == 200 else f": {status} {body}"))
        assert isinstance(body, dict)
        ids.append(body["id"])

    status, body = request("POST", "/index/rebuild")
    check(status == 200 and body["tracks_indexed"] == N_TRACKS, f"rebuild indexed {N_TRACKS}")
    status, body = request("GET", "/health/ready", key=None)
    check(status == 200 and body["status"] == "ok", f"ready: database, storage, index ({body})")
    check(f"recommender_index_tracks {N_TRACKS}.0" in metrics(), "metrics: index size")

    for track_id in ids[:3]:
        check(post_json("/likes", {"user_id": "u1", "track_id": track_id})[0] == 200, "like")
    status, body = post_json("/likes", {"user_id": "u1", "track_id": ids[0]})
    check(status == 200 and body["status"] == "exists", "repeated like is not duplicated")
    check(
        post_json("/likes", {"user_id": "u1", "track_id": ids[3]}, key=None)[0] == 401,
        "like without key is rejected",
    )

    by_track = recommendations(f"/recommendations/{ids[0]}?limit=3")
    check(len(by_track["recommendations"]) == 3, "track recommendations")
    user = recommendations("/recommendations/user/u1?limit=3")
    check(user["strategy"] == "interests", "user recommendations by interests")
    check(not {r["track_id"] for r in user["recommendations"]} & set(ids[:3]), "likes excluded")
    newcomer = recommendations("/recommendations/user/newcomer?limit=3")
    check(newcomer["strategy"] == "popular", "cold start: popular")

    with open(STATE, "w") as f:
        json.dump({"ids": ids, "by_track": by_track["recommendations"]}, f)


def verify() -> None:
    print("verify after restart:")
    with open(STATE) as f:
        state = json.load(f)
    ids = state["ids"]

    by_track = recommendations(f"/recommendations/{ids[0]}?limit=3")
    check(by_track["recommendations"] == state["by_track"], "same recommendations after restart")
    status, likes = request("GET", "/users/u1/likes", key=None)
    check(status == 200 and isinstance(likes, list) and len(likes) == 3, "likes survived restart")

    check(request("DELETE", f"/tracks/{ids[5]}", key=None)[0] == 401, "delete without key")
    check(request("DELETE", f"/tracks/{ids[5]}")[0] == 204, "delete with key")
    status, tracks = request("GET", "/tracks", key=None)
    check(status == 200 and isinstance(tracks, list) and len(tracks) == N_TRACKS - 1, "deleted")
    check(request("POST", "/index/reload")[0] == 200, "reload with key")
    check(f"recommender_index_tracks {N_TRACKS - 1}.0" in metrics(), "metrics after delete")
    req = urllib.request.Request(f"{API_URL}/tracks", headers={"X-Request-ID": "e2e-trace-1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        check(resp.headers["X-Request-ID"] == "e2e-trace-1", "request id is echoed")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    if phase not in ("seed", "verify"):
        sys.exit("usage: e2e_compose.py seed|verify")
    wait_ready()
    seed() if phase == "seed" else verify()
    print(f"e2e {phase}: passed")
