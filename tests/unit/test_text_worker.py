"""Текстовый кодировщик в отдельном процессе (поддельные кодировщики, без torch)."""

import numpy as np
import pytest

from recommender.infrastructure.data_processing.text_worker import ProcessTextEncoder


@pytest.fixture
def echo():
    encoder = ProcessTextEncoder("tests.fake_text_encoders:EchoEncoder", "fake/model")
    yield encoder
    encoder.close()


def test_vectors_round_trip_despite_library_stdout_noise(echo):
    assert echo.model_name == "fake/model"
    np.testing.assert_array_equal(echo.encode(["ab", "abcd"]), [[2, 1], [4, 1]])


def test_encoder_error_is_reported_and_process_survives(echo):
    with pytest.raises(RuntimeError, match="ValueError: bad text"):
        echo.encode(["boom"])
    np.testing.assert_array_equal(echo.encode(["xyz"]), [[3, 1]])


def test_child_process_does_not_load_faiss():
    import faiss  # noqa: F401 — в родителе faiss есть, в процессе кодировщика его быть не должно

    probe = ProcessTextEncoder("tests.fake_text_encoders:ModulesProbe")
    try:
        assert probe.encode(["x"])[0, 0] == 0.0
    finally:
        probe.close()


def test_missing_library_raises_import_error():
    with pytest.raises(ImportError, match="nonexistent_ml_library"):
        ProcessTextEncoder("tests.fake_text_encoders:MissingDependency")


def test_close_stops_the_process(echo):
    echo.close()
    assert echo._process.poll() is not None
