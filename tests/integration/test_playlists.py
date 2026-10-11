"""Плейлисты как лайки: разбор форматов, сопоставление с каталогом, синхронизация."""

import pytest
from sqlalchemy import select

from recommender.application.playlists import (
    CatalogMatcher,
    PlaylistEntry,
    import_playlists,
    normalize_title,
    parse_playlist,
)
from recommender.infrastructure.storage.postgres import LikeORM


@pytest.mark.parametrize(
    ("title", "normalized"),
    [
        ("Мяу (prod. by Heronwater, Rallex)", "мяу"),
        ("Не везёт", "не везет"),
        ("Как дела Syringe_", "как дела syringe"),
        ("Как дела Syringe?", "как дела syringe"),
        ("1000 Shells (feat. Loco OG Rocka)", "1000 shells"),
        ("Song feat. Someone", "song"),
        ("Track [Official Video]", "track"),
        ("(I Just) Died In Your Arms", "i just died in your arms"),  # не хвост — оставляем
    ],
)
def test_normalize_title(title, normalized):
    assert normalize_title(title) == normalized


def test_parse_txt_m3u_csv():
    txt, bad = parse_playlist("a.txt", "﻿A — One\nB - Two\n\nno separator\nC – Three\n")
    assert txt == [
        PlaylistEntry("A", "One"),
        PlaylistEntry("B", "Two"),
        PlaylistEntry("C", "Three"),
    ]
    assert bad == ["no separator"]

    m3u = "#EXTM3U\n#EXTINF:210,A - One\n/music/A - One.mp3\nC:\\Music\\B - Two.flac\n#comment\n"
    assert parse_playlist("b.m3u8", m3u)[0] == [
        PlaylistEntry("A", "One"),
        PlaylistEntry("B", "Two"),
    ]

    with_header = "title,artist,album\nOne,A,X\nTwo,B,Y\n,,\n"
    assert parse_playlist("c.csv", with_header)[0] == [
        PlaylistEntry("A", "One"),
        PlaylistEntry("B", "Two"),
    ]
    single = "A - One\nB - Two\n"
    assert parse_playlist("d.csv", single)[0] == [
        PlaylistEntry("A", "One"),
        PlaylistEntry("B", "Two"),
    ]


def test_matcher_needs_a_shared_artist():
    matcher = CatalogMatcher(
        [
            ("t1", "LIZER, FLESH", "Kids"),
            ("t2", "Heronwater", "Мяу (prod. by Heronwater, Rallex)"),
            ("t3", "Другой", "Kids"),
        ]
    )

    assert matcher.find(PlaylistEntry("Flesh", "kids")) == "t1"  # соавтор
    assert matcher.find(PlaylistEntry("heronwater", "Мяу")) == "t2"
    assert matcher.find(PlaylistEntry("Nobody", "Kids")) is None  # только по названию — нет
    assert matcher.find(PlaylistEntry("Heronwater", "Пугали")) is None


async def test_import_playlists_syncs_likes(api, tmp_path):
    ids = {}
    for i, (artist, title) in enumerate(
        [("Heronwater", "Мяу"), ("LIZER, FLESH", "Kids"), ("ЗоХа", "Танки"), ("Lizer", "Улица")]
    ):
        ids[title] = (await api.upload(i, title=title, artist=artist))["id"]
    folder = tmp_path / "playlists"
    folder.mkdir()
    (folder / "вечер.txt").write_text("Heronwater - мяу\nFLESH - Kids\nСкриптонит - Положение\n")
    (folder / "rap.m3u").write_text("#EXTINF:1,ЗоХа - Танки\nx.mp3\nLizer - Улица.mp3\n")
    (folder / "notes.md").write_text("ignored")

    async with api.sessions() as db:
        results = await import_playlists(db, folder)
    by_name = {r.name: r for r in results}

    assert set(by_name) == {"вечер.txt", "rap.m3u"}
    assert (by_name["вечер.txt"].matched, by_name["вечер.txt"].total) == (2, 3)
    assert by_name["вечер.txt"].unmatched == ["Скриптонит - Положение"]
    assert by_name["вечер.txt"].user_id == "playlist:вечер"

    (folder / "вечер.txt").write_text("Heronwater - Мяу\n")  # трек убрали из плейлиста
    async with api.sessions() as db:
        await import_playlists(db, folder)
        likes = {
            (u, t) for u, t in (await db.execute(select(LikeORM.user_id, LikeORM.track_id))).all()
        }

    assert likes == {
        ("playlist:вечер", ids["Мяу"]),
        ("playlist:rap", ids["Танки"]),
        ("playlist:rap", ids["Улица"]),
    }


async def test_playlist_user_gets_recommendations(api, tmp_path):
    seeded = await api.seed(5)
    tracks = (await api.client.get("/tracks")).json()
    title = {t["id"]: t["title"] for t in tracks}
    folder = tmp_path / "pl"
    folder.mkdir()
    (folder / "mix.txt").write_text(f"Artist - {title[seeded[0]]}\nArtist - {title[seeded[1]]}\n")
    async with api.sessions() as db:
        await import_playlists(db, folder)

    resp = await api.client.get("/recommendations/user/playlist:mix")

    assert resp.json()["strategy"] == "interests"
    assert {r["track_id"] for r in resp.json()["recommendations"]} == set(seeded[2:])


async def test_missing_folder_is_an_error(api, tmp_path):
    async with api.sessions() as db:
        with pytest.raises(ValueError, match="Not a directory"):
            await import_playlists(db, tmp_path / "nope")
