"""Версии индекса на диске: публикация, очистка, legacy-формат, изменения поверх новых версий."""

import threading

import numpy as np
import pytest

from recommender.config import settings
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage import artifacts as store
from recommender.infrastructure.storage.faiss_index import FaissRecommender

DIM = 4


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "index_dir", tmp_path / "index")
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    (tmp_path / "models").mkdir()
    return tmp_path / "index"


def _build(ids: list[str]) -> tuple[FaissRecommender, FeatureNormalizer]:
    vectors = np.random.default_rng(len(ids)).standard_normal((max(len(ids), 2), DIM))
    normalizer = FeatureNormalizer("standard").fit(vectors.astype(np.float32))
    engine = FaissRecommender(dimension=DIM, metric="cosine")
    engine.add_tracks(ids, normalizer.transform(vectors[: len(ids)].astype(np.float32)))
    return engine, normalizer


def _adder(track_id: str):
    def add(artifacts: store.IndexArtifacts) -> bool:
        artifacts.engine.add_tracks([track_id], np.ones((1, DIM), np.float32))
        return True

    return add


def test_publish_writes_complete_version_and_pointer(root):
    engine, normalizer = _build(["a", "b"])

    version = store.publish(engine, normalizer)

    folder = root / "versions" / version
    assert {p.name for p in folder.iterdir()} == {"faiss.index", "meta.joblib", "normalizer.joblib"}
    assert (root / "CURRENT").read_text() == version
    loaded = store.load_current()
    assert loaded.engine.version == version == engine.version
    assert loaded.engine.track_ids == ["a", "b"]
    assert loaded.normalizer.is_fitted


def test_old_versions_are_pruned_but_current_and_recent_kept(root, monkeypatch):
    monkeypatch.setattr(settings, "index_keep_versions", 2)
    leftover = root / "versions" / ".tmp-crashed"
    leftover.mkdir(parents=True)

    versions = [store.publish(*_build(["a"])) for _ in range(4)]

    assert store.list_versions() == versions[-2:]
    assert store.current_version() == versions[-1]
    assert not leftover.exists()


def test_no_index_raises_file_not_found(root):
    with pytest.raises(FileNotFoundError):
        store.load_current()


def test_legacy_layout_is_read_and_replaced_on_first_publish(root):
    engine, normalizer = _build(["old"])
    root.mkdir()
    engine.save(root)
    normalizer.save(settings.models_dir / "normalizer.joblib")

    assert store.current_version() == store.LEGACY
    assert store.load_current().engine.track_ids == ["old"]

    store.publish(*_build(["new"]))

    assert not (root / "faiss.index").exists()
    assert not (settings.models_dir / "normalizer.joblib").exists()
    assert store.load_current().engine.track_ids == ["new"]


def test_change_from_stale_base_is_applied_on_top_of_newer_version(root):
    store.publish(*_build(["a"]))
    stale = store.load_current()  # сервис загрузил версию с одним треком
    store.publish(*_build(["a", "b", "c"]))  # batch опубликовал новую сборку

    result = store.update_current(stale, _adder("upload"))

    assert sorted(result.engine.track_ids) == ["a", "b", "c", "upload"]
    assert store.load_current().engine.track_ids == result.engine.track_ids
    assert store.current_version() == result.engine.version


def test_no_change_means_no_new_version(root):
    store.publish(*_build(["a"]))
    before = store.list_versions()

    store.update_current(store.load_current(), lambda artifacts: False)

    assert store.list_versions() == before


def test_concurrent_changes_from_one_stale_base_are_all_kept(root, monkeypatch):
    monkeypatch.setattr(settings, "index_keep_versions", 20)
    store.publish(*_build(["a"]))
    stale = store.load_current()

    threads = [
        threading.Thread(target=store.update_current, args=(stale, _adder(f"t{i}")))
        for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    final = store.load_current().engine.track_ids
    assert sorted(final) == ["a", *sorted(f"t{i}" for i in range(8))]
