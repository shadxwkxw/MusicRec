"""Имена артистов: соавторы, регистр, группы связанных артистов, лимит."""

import pytest

from recommender.domain.artists import ArtistCap, artist_groups, artist_names, names_of


@pytest.mark.parametrize(
    ("artist", "names"),
    [
        ("Lizer", {"lizer"}),
        ("LIZER, FLESH", {"lizer", "flesh"}),
        ("FLESH,LIZER", {"lizer", "flesh"}),
        ("Whole Lotta Swag feat. VERi RERi", {"whole lotta swag", "veri reri"}),
        ("Kai Angel x 9mice", {"kai angel", "9mice"}),
        ("A ft B featuring C & D; E vs. F", {"a", "b", "c", "d", "e", "f"}),
        ("Yanix, OG Buda, 163ONMYNECK", {"yanix", "og buda", "163onmyneck"}),
        ("  Heronwater ,  BUSHIDO   ZHO ", {"heronwater", "bushido zho"}),
        ("Malcolm X", {"malcolm x"}),  # x на конце — часть имени, не разделитель
        ("Xzibit", {"xzibit"}),
        ("Unknown", set()),
        ("", set()),
        (None, set()),
    ],
)
def test_artist_names(artist, names):
    assert artist_names(artist) == names


def test_collaborations_link_artists_into_one_group():
    groups = artist_groups(
        {
            "a": "Lizer",
            "b": "LIZER, FLESH",
            "c": "Flesh",
            "d": "Heronwater",
            "e": "Unknown",
            "f": "Unknown",
        }
    )

    assert groups["a"] == groups["b"] == groups["c"]  # Lizer — FLESH через коллаборацию
    assert groups["d"] != groups["a"]
    assert groups["e"] != groups["f"]  # без артиста — не «один и тот же артист»


def test_cap_counts_each_coauthor_and_skips_liked():
    cap = ArtistCap(2, liked=names_of({"t": "Lizer"}, ["t"]))
    lizer_flesh, flesh = artist_names("LIZER, FLESH"), artist_names("Flesh")

    for _ in range(2):
        assert cap.allows(lizer_flesh)
        cap.add(lizer_flesh)

    assert not cap.allows(flesh)  # FLESH уже дважды — как соавтор
    assert cap.allows(artist_names("Lizer"))  # Lizer лайкнут — без лимита
    assert cap.allows(lizer_flesh)  # и его коллаборации тоже
    assert not cap.allows(artist_names("Flesh feat. Kai Angel"))
    assert ArtistCap(0).allows(flesh)
