import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--num-data",
        action="store",
        default=1024,
        type=int,
        help="Number of data points for ORAM tests (default: 1024)"
    )


@pytest.fixture
def num_data(request):
    """Get the number of data points from command line option."""
    return request.config.getoption("--num-data")


@pytest.fixture
def test_file(tmp_path):
    """Provide a temp file path under pytest's temp directory."""
    return tmp_path / "test.bin"
