"""Жанр и другие теги аудиофайлов: приведение к общим названиям, импорт, загрузка через API."""

import shutil

import pytest
from mutagen.wave import WAVE
from sqlalchemy import select

from recommender.application.batch_extract import run_batch_extract
from recommender.infrastructure.data_processing.tags import canonical_genre, read_tags
from recommender.infrastructure.storage.postgres import TrackORM


@pytest.mark.parametrize(
    ("raw", "genre"),
    [
        ("Рэп", "Hip-Hop"),
        ("Хип-хоп", "Hip-Hop"),
        ("Хип-хоп;Рэп", "Hip-Hop"),
        ("hip hop", "Hip-Hop"),
        ("Поп", "Pop"),
        ("Поп-рок", "Pop"),
        ("Pop/Rock", "Pop"),
        ("Альтернатива", "Rock"),
        ("Электронная", "Electronic"),
        ("R&B", "Soul-RnB"),
        ("Шансон", "Шансон"),  # незнакомый — как есть
        ("  ", None),
        (None, None),
    ],
)
def test_canonical_genre(raw, genre):
    assert canonical_genre(raw) == genre


def _tagged(source, target, **tags):
    shutil.copy(source, target)
    audio = WAVE(str(target))
    audio.add_tags()
    from mutagen.id3 import TCON, TIT2, TPE1

    frames = {"genre": TCON, "title": TIT2, "artist": TPE1}
    for key, value in tags.items():
        audio.tags.add(frames[key](encoding=3, text=value))
    audio.save()
    return target


def test_read_tags(audio_files, tmp_path):
    path = _tagged(audio_files[0], tmp_path / "a.wav", genre="Хип-хоп;Рэп", artist="LIZER, FLESH")

    tags = read_tags(path)

    assert (tags.genre, tags.artist) == ("Hip-Hop", "LIZER, FLESH")
    assert read_tags(audio_files[1]).genre is None  # без тегов
    (tmp_path / "junk.mp3").write_bytes(b"not audio")
    assert read_tags(tmp_path / "junk.mp3").genre is None


async def test_import_takes_genre_from_tags(api, audio_files, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _tagged(audio_files[0], inbox / "Big Baby Tape - ACAB.wav", genre="Рэп")
    _tagged(audio_files[1], inbox / "MORGENSHTERN - Флаг.wav", genre="Рок")
    shutil.copy(audio_files[2], inbox / "No Tags - Track.wav")

    async with api.sessions() as db:
        await run_batch_extract(inbox, db)
        tracks = (await db.execute(select(TrackORM))).scalars().all()

    assert {(t.artist, t.title, t.genre) for t in tracks} == {
        ("Big Baby Tape", "ACAB", "Hip-Hop"),
        ("MORGENSHTERN", "Флаг", "Rock"),
        ("No Tags", "Track", None),
    }


async def test_api_upload_reads_genre_unless_given(api, audio_files, tmp_path):
    path = _tagged(audio_files[0], tmp_path / "t.wav", genre="Поп")

    async def upload(**data):
        with path.open("rb") as f:
            resp = await api.client.post(
                "/tracks/upload", files={"file": ("t.wav", f, "audio/wav")}, data=data
            )
        assert resp.status_code == 200, resp.text
        return resp.json()["genre"]

    assert await upload(title="A") == "Pop"
    assert await upload(title="B", genre="Indie") == "Indie"


async def test_fill_genres_from_tags(api, audio_files, tmp_path):
    from recommender.application.batch_extract import fill_genres_from_tags

    tagged = _tagged(audio_files[0], tmp_path / "rap.wav", genre="Рэп")
    plain = tmp_path / "plain.wav"
    shutil.copy(audio_files[1], plain)
    rows = [
        ("t1", str(tagged), None, "upload"),
        ("t2", str(plain), None, "upload"),
        ("t3", str(tmp_path / "gone.wav"), None, "upload"),
        ("t4", str(tagged), "Indie", "upload"),  # жанр уже есть — не трогаем
        ("t5", str(tagged), None, "fma"),  # другой источник
    ]
    async with api.sessions() as db:
        for track_id, path, genre, source in rows:
            db.add(
                TrackORM(
                    id=track_id,
                    title=track_id,
                    filename=f"{track_id}.wav",
                    audio_path=path,
                    genre=genre,
                    source=source,
                )
            )
        await db.commit()

        result = await fill_genres_from_tags(db, sources=["upload"], chunk=2)
        genres = dict((await db.execute(select(TrackORM.id, TrackORM.genre))).all())

    assert (result.filled, result.no_tag, result.missing) == (1, 1, [str(tmp_path / "gone.wav")])
    assert genres == {"t1": "Hip-Hop", "t2": None, "t3": None, "t4": "Indie", "t5": None}
