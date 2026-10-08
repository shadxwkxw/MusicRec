"""Хранилище аудио в S3.

По умолчанию S3 подменяет moto внутри процесса: без сети и ключей. С
S3_TEST_ENDPOINT те же тесты идут на настоящем S3-совместимом сервере (в CI —
RustFS): ключи берутся из AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, бакет
очищается до и после каждого теста.
"""

import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import boto3
import pytest
from moto import mock_aws
from sqlalchemy import select

from recommender.application.batch_extract import ImportItem, run_batch_import, s3_import_items
from recommender.application.batch_recommend import run_batch_recommend
from recommender.application.migrate_s3 import migrate_artifacts, migrate_audio, migrate_index
from recommender.config import settings
from recommender.infrastructure.storage import artifacts as index_store
from recommender.infrastructure.storage import audio_store, index_mirror
from recommender.infrastructure.storage.audio_store import S3AudioStore
from recommender.infrastructure.storage.postgres import TrackORM
from tests.integration.test_embeddings import FakeEmbedder, _embed
from tests.unit.test_artifacts import _adder, _build

BUCKET = "music-bucket"
ENDPOINT = os.getenv("S3_TEST_ENDPOINT") or None


def _empty_bucket(client) -> None:
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            client.delete_objects(Bucket=BUCKET, Delete={"Objects": keys})


@pytest.fixture
def s3(monkeypatch):
    if ENDPOINT is None:
        for name, value in {
            "AWS_ACCESS_KEY_ID": "testing",
            "AWS_SECRET_ACCESS_KEY": "testing",
        }.items():
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setattr(settings, "storage_backend", "s3")
    monkeypatch.setattr(settings, "s3_bucket", BUCKET)
    monkeypatch.setattr(settings, "s3_upload_prefix", "uploads/")
    monkeypatch.setattr(settings, "s3_endpoint_url", ENDPOINT)
    audio_store._s3_client.cache_clear()
    with mock_aws() if ENDPOINT is None else nullcontext():
        client = boto3.client("s3", endpoint_url=ENDPOINT, region_name="us-east-1")
        if BUCKET not in {b["Name"] for b in client.list_buckets().get("Buckets", [])}:
            client.create_bucket(Bucket=BUCKET)
        _empty_bucket(client)
        yield client
        _empty_bucket(client)
    audio_store._s3_client.cache_clear()


def _keys(client, prefix: str = "") -> set[str]:
    listing = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return {o["Key"] for o in listing.get("Contents", [])}


# ── Хранилище ────────────────────────────────────────────────────


def test_store_saves_downloads_and_cleans_up(s3, tmp_path):
    store = S3AudioStore(BUCKET, "uploads/")
    local = tmp_path / "song.mp3"
    local.write_bytes(b"ID3 fake audio")

    location = store.save_upload(local, "abc_song.mp3")

    assert location == f"s3://{BUCKET}/uploads/abc_song.mp3"
    assert not local.exists()  # локальный файл перенесён в S3
    with store.local_copy(location) as copy:
        assert copy.suffix == ".mp3"  # расширение нужно декодерам
        assert copy.read_bytes() == b"ID3 fake audio"
        folder = copy.parent
    assert not folder.exists()  # временная копия удалена


def test_missing_object_is_file_not_found(s3):
    with (
        pytest.raises(FileNotFoundError),
        S3AudioStore(BUCKET).local_copy(f"s3://{BUCKET}/nope.mp3"),
    ):
        pass


def test_only_own_uploads_are_deleted(s3):
    for key in ("uploads/mine.mp3", "music/catalog.mp3"):
        s3.put_object(Bucket=BUCKET, Key=key, Body=b"x")
    store = S3AudioStore(BUCKET, "uploads/")

    assert store.delete_upload(f"s3://{BUCKET}/uploads/mine.mp3") is True
    assert store.delete_upload(f"s3://{BUCKET}/music/catalog.mp3") is False
    assert store.delete_upload("s3://other-bucket/uploads/x.mp3") is False

    assert _keys(s3) == {"music/catalog.mp3"}


def test_listing_skips_non_audio_and_own_uploads(s3):
    for key in (
        "music/a.mp3",
        "music/sub/b.FLAC",
        "music/cover.jpg",
        "uploads/u.mp3",
        "other/c.mp3",
    ):
        s3.put_object(Bucket=BUCKET, Key=key, Body=b"x")

    assert S3AudioStore(BUCKET, "uploads/").list_audio("") == [
        "music/a.mp3",
        "music/sub/b.FLAC",
        "other/c.mp3",
    ]
    assert S3AudioStore(BUCKET, "uploads/").list_audio("music/") == [
        "music/a.mp3",
        "music/sub/b.FLAC",
    ]


# ── API ──────────────────────────────────────────────────────────


async def test_upload_goes_to_s3_and_delete_removes_it(api, s3):
    track = await api.upload(0)

    async with api.sessions() as db:
        stored = await db.get(TrackORM, track["id"])
    assert stored.audio_path == f"s3://{BUCKET}/uploads/{track['id']}_track_0.wav"
    assert _keys(s3) == {f"uploads/{track['id']}_track_0.wav"}
    assert list(settings.audio_dir.iterdir()) == []  # на диске сервиса ничего не осталось

    assert (await api.client.delete(f"/tracks/{track['id']}")).status_code == 204
    assert _keys(s3) == set()


async def test_rejected_upload_leaves_nothing_in_s3(api, s3):
    resp = await api.client.post(
        "/tracks/upload",
        files={"file": ("broken.wav", b"not audio", "audio/wav")},
        data={"title": "Broken"},
    )

    assert resp.status_code == 400
    assert _keys(s3) == set()


async def test_client_filename_cannot_escape_the_upload_prefix(api, s3, audio_files):
    with audio_files[0].open("rb") as f:
        resp = await api.client.post(
            "/tracks/upload",
            files={"file": ("../../etc/evil.wav", f, "audio/wav")},
            data={"title": "Evil"},
        )

    assert resp.status_code == 200
    assert _keys(s3) == {f"uploads/{resp.json()['id']}_evil.wav"}


# ── Импорт каталога и эмбеддинги ─────────────────────────────────


async def test_import_from_s3_prefix_then_embed_and_delete_keeps_catalog(
    api, s3, audio_files, monkeypatch
):
    monkeypatch.setattr(settings, "feature_source", "embedding")
    monkeypatch.setattr(settings, "embedding_model", FakeEmbedder.model_name)
    for i in range(3):
        s3.upload_file(str(audio_files[i]), BUCKET, f"music/artist/song_{i}.wav")
    s3.put_object(Bucket=BUCKET, Key="music/readme.txt", Body=b"not audio")

    items = s3_import_items("music/", default_artist="Catalog")
    async with api.sessions() as db:
        stats = await run_batch_import(items, db)
        again = await run_batch_import(s3_import_items("music/"), db)
        tracks = (await db.execute(select(TrackORM))).scalars().all()

    assert (stats.processed, stats.failed) == (3, [])
    assert again.skipped == 3
    assert {t.audio_path for t in tracks} == {
        f"s3://{BUCKET}/music/artist/song_{i}.wav" for i in range(3)
    }
    assert {(t.title, t.artist) for t in tracks} == {(f"song_{i}", "Catalog") for i in range(3)}

    # embed скачивает аудио из S3; удалённый из бакета файл — в ошибках
    s3.delete_object(Bucket=BUCKET, Key="music/artist/song_2.wav")
    embed = await _embed(api)
    assert embed.processed == 2
    assert [err.split(":")[0] for _, err in embed.failed] == ["audio not found"]

    # удаление трека не трогает импортированный каталог
    first = next(t for t in tracks if t.audio_path.endswith("song_0.wav"))
    assert (await api.client.delete(f"/tracks/{first.id}")).status_code == 204
    assert "music/artist/song_0.wav" in _keys(s3)


async def test_embed_vectors_match_local_files(api, s3, audio_files, tmp_path, monkeypatch):
    """Эмбеддинг файла из S3 тот же, что у того же файла с диска."""
    from recommender.application.features import load_vectors

    monkeypatch.setattr(settings, "feature_source", "embedding")
    monkeypatch.setattr(settings, "embedding_model", FakeEmbedder.model_name)
    s3.upload_file(str(audio_files[4]), BUCKET, "music/x.wav")
    local = tmp_path / "x.wav"
    shutil.copy(audio_files[4], local)

    async with api.sessions() as db:
        await run_batch_import(s3_import_items("music/"), db)
    await _embed(api)
    async with api.sessions() as db:
        (vector,) = (await load_vectors(db)).values()

    ((_, expected),) = FakeEmbedder().embed_files([Path(local)])
    assert vector.tolist() == pytest.approx(expected.tolist())


# ── Версии индекса в бакете ──────────────────────────────────────


@pytest.fixture
def machines(s3, tmp_path, monkeypatch):
    """Две «машины» с разными локальными копиями индекса и общим бакетом."""
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    return tmp_path / "machine_a", tmp_path / "machine_b"


def test_published_version_reaches_another_machine_and_back(s3, machines):
    a, b = machines
    version = index_store.publish(*_build(["x", "y"]), root=a)

    files = {"faiss.index", "meta.joblib", "normalizer.joblib"}
    assert _keys(s3, "index/") == {f"index/versions/{version}/{f}" for f in files} | {
        "index/CURRENT"
    }
    on_b = index_store.load_current(root=b)  # пустая копия скачивает версию из бакета
    assert (on_b.engine.version, on_b.engine.track_ids) == (version, ["x", "y"])

    index_store.update_current(on_b, _adder("z"), root=b)  # изменение сервиса на машине b

    assert index_store.load_current(root=a).engine.track_ids == ["x", "y", "z"]


def test_failed_push_does_not_fail_publish_and_is_retried(s3, machines, monkeypatch):
    a, _ = machines

    def broken(self, version, folder):
        raise ConnectionError("S3 is down")

    with monkeypatch.context() as m:
        m.setattr(index_mirror.S3IndexMirror, "upload", broken)
        version = index_store.publish(*_build(["x"]), root=a)
    assert index_store.current_version(a) == version
    assert _keys(s3, "index/") == set()

    assert index_store.sync(a) == version  # локальная новее — уходит в бакет
    assert f"index/versions/{version}/faiss.index" in _keys(s3, "index/")


def test_unreachable_bucket_falls_back_to_local_copy(s3, machines, monkeypatch):
    a, _ = machines
    version = index_store.publish(*_build(["x"]), root=a)
    monkeypatch.setattr(settings, "s3_bucket", "no-such-bucket")

    assert index_store.load_current(root=a).engine.version == version
    with pytest.raises(Exception, match="NoSuchBucket"):
        index_store.sync(a)


def test_old_versions_are_pruned_in_bucket(s3, machines, monkeypatch):
    a, _ = machines
    monkeypatch.setattr(settings, "index_keep_versions", 2)

    versions = [index_store.publish(*_build(["x"]), root=a) for _ in range(4)]

    assert index_mirror.configured_mirror().versions() == versions[-2:]


# ── Перенос локальных данных и выгрузки в S3 ─────────────────────


async def test_migrate_moves_uploads_index_and_outputs(api, s3, audio_files, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_backend", "local")
    ids = await api.seed(3)  # загружены, пока хранилище было локальным
    lost = (await api.upload(3))["id"]
    (settings.audio_dir / f"{lost}_track_3.wav").unlink()
    catalog = tmp_path / "fma" / "song.wav"  # датасет вне папки загрузок
    catalog.parent.mkdir()
    shutil.copy(audio_files[5], catalog)
    async with api.sessions() as db:
        await run_batch_import([ImportItem(catalog, "song.wav", "Song", "FMA")], db)
    outputs = tmp_path / "artifacts"
    outputs.mkdir()
    (outputs / "recs.parquet").write_bytes(b"PAR1")
    (outputs / "notes.txt").write_text("not an output")
    monkeypatch.setattr(settings, "storage_backend", "s3")

    async with api.sessions() as db:
        audio = await migrate_audio(db, delete_local=True)
        again = await migrate_audio(db)
        tracks = {t.id: t for t in (await db.execute(select(TrackORM))).scalars()}

    assert (audio.moved, audio.deleted_local, audio.outside_audio_dir) == (3, 3, 1)
    assert audio.missing == [str(settings.audio_dir / f"{lost}_track_3.wav")]
    assert again.moved == 0
    for i, track_id in enumerate(ids):
        key = f"uploads/{track_id}_track_{i}.wav"
        assert tracks[track_id].audio_path == f"s3://{BUCKET}/{key}"
        assert key in _keys(s3, "uploads/")
    assert tracks[lost].audio_path == str(settings.audio_dir / f"{lost}_track_3.wav")
    assert {t.audio_path for t in tracks.values() if t.title == "Song"} == {str(catalog)}
    assert list(settings.audio_dir.iterdir()) == []
    assert catalog.exists()

    version = migrate_index()
    assert version == index_store.current_version()
    assert f"index/versions/{version}/faiss.index" in _keys(s3, "index/")
    assert migrate_artifacts(outputs) == [f"s3://{BUCKET}/artifacts/recs.parquet"]

    # перенесённая загрузка — снова «своя»: удаление трека удаляет объект
    assert (await api.client.delete(f"/tracks/{ids[0]}")).status_code == 204
    assert f"uploads/{ids[0]}_track_0.wav" not in _keys(s3, "uploads/")


async def test_migrate_requires_s3_backend(api, monkeypatch):
    monkeypatch.setattr(settings, "storage_backend", "local")
    async with api.sessions() as db:
        with pytest.raises(ValueError, match="AUDIO_STORAGE=s3"):
            await migrate_audio(db)


async def test_batch_recommend_writes_to_s3(api, s3):
    await api.seed(3)

    async with api.sessions() as db:
        await run_batch_recommend(db, f"s3://{BUCKET}/artifacts/recs.csv", top_n=2)

    body = s3.get_object(Bucket=BUCKET, Key="artifacts/recs.csv")["Body"].read().decode()
    lines = body.splitlines()
    assert lines[0] == "source_track_id,rank,target_track_id,score"
    assert len(lines) == 1 + 3 * 2


def test_s3_import_takes_artist_and_title_from_key(s3):
    for key in ("music/Heronwater - Мяу.mp3", "music/sub/untitled.mp3"):
        s3.put_object(Bucket=BUCKET, Key=key, Body=b"x")

    items = {i.filename: (i.artist, i.title) for i in s3_import_items("music/", "Catalog")}

    assert items == {
        "music/Heronwater - Мяу.mp3": ("Heronwater", "Мяу"),
        "music/sub/untitled.mp3": ("Catalog", "untitled"),
    }


def test_upload_directory_skips_tracks_already_in_bucket(s3, tmp_path):
    import unicodedata

    from recommender.application.migrate_s3 import upload_directory

    s3.put_object(
        Bucket=BUCKET, Key="uploads/0b6f1d2e-1111-4222-8333-944455556666_A - One.mp3", Body=b"x"
    )
    s3.put_object(Bucket=BUCKET, Key="music/Album/B - Два.mp3", Body=b"x")
    folder = tmp_path / "new"
    (folder / "Album 1").mkdir(parents=True)
    (folder / "Album 2").mkdir()
    (folder / "A - One.mp3").write_bytes(b"dup of upload")
    nfd = unicodedata.normalize("NFD", "B - Два.mp3")  # «й/ё»-подобные имена с macOS
    (folder / "Album 1" / nfd).write_bytes(b"dup of catalog")
    (folder / "Album 1" / "C - Йога.mp3").write_bytes(b"new")
    (folder / "Album 2" / "C - Йога.mp3").write_bytes(b"same track, other album")
    (folder / "Album 2" / "a - ONE.mp3").write_bytes(b"same as upload, other case")
    (folder / "Album 1" / "cover.jpg").write_bytes(b"not audio")

    result = upload_directory(folder, "music/")

    assert result.uploaded == ["music/Album 1/C - Йога.mp3"]
    assert len(result.duplicates) == 4
    assert _keys(s3, "music/") == {"music/Album/B - Два.mp3", "music/Album 1/C - Йога.mp3"}
