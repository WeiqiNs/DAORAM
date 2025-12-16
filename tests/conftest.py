import pytest


@pytest.fixture
def test_file(tmp_path):
    """Provide a temp file path under pytest's temp directory."""
    return tmp_path / "test.bin"
