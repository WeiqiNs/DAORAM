import os

import pytest

from oblivlib.dependency import AesGcm, InteractLocalServer

# soram is opt-in (slow); run it with `pytest tests/soram`.
collect_ignore = ["soram"]

# Default dataset size for the round-trip suites; override per run with the NUM_DATA env var.
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
