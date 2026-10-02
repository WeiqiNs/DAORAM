import os

import pytest

from oblivlib.dependency import AesGcm, InteractLocalServer

DEFAULT_NUM_DATA = 2**12


@pytest.fixture
def num_data() -> int:
    return int(os.environ.get("NUM_DATA", DEFAULT_NUM_DATA))


@pytest.fixture
def test_file(tmp_path):
    return tmp_path / "test.bin"


@pytest.fixture
def client():
    return InteractLocalServer()


@pytest.fixture
def encryptor():
    return AesGcm()
